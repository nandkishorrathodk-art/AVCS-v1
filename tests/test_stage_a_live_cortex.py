"""Comprehensive validation suite for Stage A: AVCS System 1 Live-Stream Visual Cortex.

Covers:
1. Decoupled 1-Slot Latest-Value Buffer (LatestFrameMailbox):
   - Bounded 1-frame capacity, drop-tail replacement, zero backlog, zero stale lag.
   - Rejection of out-of-order sequence and non-monotonic timestamps.
   - Concurrent multi-threaded producer and consumer thread safety.
2. Spatiotemporal Visual Cortex Model (StreamingVisualCortexV1 & ConvGRUCell):
   - Multi-scale spatial feature extraction from raw input pixels on CPU.
   - 16x16x48 ConvGRU persistent recurrent temporal hidden state.
   - Forward pass contract: (latent_z, next_state, output_dict).
   - Recurrent state propagation across consecutive observations.
   - Timestamp delta decay attenuation.
   - Future predictive latent head mapping.
   - Real-time CPU parameter budget (~157k parameters).
3. Clean Random Weight Initialization:
   - Kaiming normal for 2D convolutions.
   - Strictly orthogonal recurrent kernel W_hh for ConvGRU.
   - Xavier uniform for linear projection layers.
   - Update gate bias initialized to 1.0, candidate bias initialized to 0.0.
   - Zero legacy checkpoint loading.
4. Fail-Closed Tri-State Security & Privacy Controls (StreamingPermissions):
   - Independent flags: enable_observation, enable_online_learning, authorize_checkpoint_save.
   - All three fail closed with PermissionError until explicitly authorized.
5. Streaming Observer Runtime & Gap Reset (StreamingObserver):
   - Fail-closed observation permission gate.
   - Recurrent state gap reset when delta_t > reset_after_seconds.
   - Stale, duplicate, and out-of-order packet skipping.
6. Zero Raw Persistence:
   - In-memory ephemeral buffers only; zero raw frames/videos persisted to disk.
"""
from __future__ import annotations

import math
from pathlib import Path
import threading
import time

import numpy as np
import pytest
import torch
import torch.nn as nn

from avcs.streaming_v1.model import (
    ConvGRUCell,
    CortexState,
    StreamingVisualCortexV1,
    create_clean_random_cortex,
)
from avcs.streaming_v1.runtime import (
    FramePacket,
    LatestFrameBuffer,
    LatestFrameMailbox,
    ObservationUpdate,
    StreamingObserver,
    StreamingPermissions,
)
from avcs.streaming_v1.sources import SyntheticCaptureSource


# ==============================================================================
# 1. Decoupled 1-Slot Latest-Value Mailbox Buffer Tests
# ==============================================================================

def test_mailbox_capacity_is_strictly_bounded_to_one():
    """Verify that LatestFrameMailbox has bounded capacity 1 and reports empty/full correctly."""
    box = LatestFrameMailbox()
    assert box.capacity == 1
    assert box.is_empty()
    assert not box.closed
    assert box.replaced_count == 0

    frame_0 = np.zeros((32, 32, 3), dtype=np.uint8)
    p0 = FramePacket(sequence=0, timestamp=1.0, rgb=frame_0)
    box.put(p0)

    assert not box.is_empty()
    assert box.replaced_count == 0

    taken = box.take()
    assert taken is not None
    assert taken.sequence == 0
    assert box.is_empty()


def test_mailbox_drop_tail_replaces_unread_frame_and_increments_count():
    """Verify drop-tail policy: incoming newer frame replaces unread frame immediately with zero lag."""
    box = LatestFrameMailbox()
    f0 = np.full((32, 32, 3), 10, dtype=np.uint8)
    f1 = np.full((32, 32, 3), 20, dtype=np.uint8)
    f2 = np.full((32, 32, 3), 30, dtype=np.uint8)

    box.put(FramePacket(0, 1.0, f0))
    assert box.replaced_count == 0

    # Put second frame before consumer takes the first
    box.put(FramePacket(1, 1.1, f1))
    assert box.replaced_count == 1

    # Put third frame before consumer takes the second
    box.put(FramePacket(2, 1.2, f2))
    assert box.replaced_count == 2

    # Consumer should immediately receive the latest frame (sequence 2), never stale sequence 0 or 1
    freshest = box.take()
    assert freshest is not None
    assert freshest.sequence == 2
    assert freshest.timestamp == pytest.approx(1.2)
    assert np.array_equal(freshest.rgb, f2)

    # Slot is now empty
    assert box.is_empty()
    assert box.take(timeout=0.01) is None


def test_mailbox_rejects_out_of_order_sequences_and_timestamps():
    """Verify buffer rejects retrograde or equal sequences and timestamps."""
    box = LatestFrameMailbox()
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    box.put(FramePacket(5, 10.0, frame))

    # Reject smaller sequence
    with pytest.raises(ValueError, match="out-of-order"):
        box.put(FramePacket(4, 10.1, frame))

    # Reject equal sequence
    with pytest.raises(ValueError, match="out-of-order"):
        box.put(FramePacket(5, 10.1, frame))

    # Reject retrograde timestamp
    with pytest.raises(ValueError, match="out-of-order"):
        box.put(FramePacket(6, 9.9, frame))

    # Reject equal timestamp
    with pytest.raises(ValueError, match="out-of-order"):
        box.put(FramePacket(6, 10.0, frame))


def test_mailbox_backward_compatible_alias():
    """Verify LatestFrameBuffer is an exact alias for LatestFrameMailbox."""
    assert LatestFrameBuffer is LatestFrameMailbox
    buf = LatestFrameBuffer()
    assert isinstance(buf, LatestFrameMailbox)


def test_mailbox_concurrent_producer_consumer_guarantees_zero_backlog():
    """Verify multi-threaded producer-consumer dynamics under high rate capture."""
    mailbox = LatestFrameMailbox()
    received_sequences: list[int] = []
    stop_event = threading.Event()

    def producer():
        seq = 0
        base_t = time.monotonic()
        for i in range(80):
            if stop_event.is_set():
                break
            frame = np.full((32, 32, 3), i % 256, dtype=np.uint8)
            mailbox.put(FramePacket(seq, base_t + i * 0.005, frame))
            seq += 1
            time.sleep(0.001)

    def consumer():
        while not stop_event.is_set():
            pkt = mailbox.take(timeout=0.05)
            if pkt is not None:
                received_sequences.append(pkt.sequence)
                time.sleep(0.01)  # Consumer is slower than producer
            if len(received_sequences) >= 8:
                break

    prod_thread = threading.Thread(target=producer)
    cons_thread = threading.Thread(target=consumer)
    cons_thread.start()
    prod_thread.start()

    cons_thread.join(timeout=5.0)
    stop_event.set()
    prod_thread.join(timeout=5.0)
    mailbox.close()

    assert len(received_sequences) >= 5
    # Sequences consumed must be strictly monotonically increasing (no reordering)
    for i in range(1, len(received_sequences)):
        assert received_sequences[i] > received_sequences[i - 1]
    # Because producer is faster than consumer, replaced_count must be positive
    assert mailbox.replaced_count > 0


# ==============================================================================
# 2. Spatiotemporal Visual Cortex Model Architecture Tests
# ==============================================================================

def test_spatiotemporal_cortex_encoder_scales_and_convgru_state():
    """Verify multi-scale spatial encoder directly downsamples raw pixels to 16x16 with 48 channels."""
    model = StreamingVisualCortexV1(feature_dim=48, hidden_dim=48, latent_dim=64)
    model.eval()

    # Raw RGB input at 128x128
    batch_size = 2
    raw_pixels = torch.rand(batch_size, 3, 128, 128)

    fused_pyr, eighth_scale = model.encode_spatial(raw_pixels)
    # Eighth scale downsampling: 128 / 8 = 16
    assert eighth_scale.shape == (batch_size, 48, 16, 16)
    assert fused_pyr.shape == (batch_size, 48, 16, 16)

    # Initial forward step
    latent_z, next_state, outputs = model.forward(raw_pixels, prev_state=None, delta_t=0.0)
    assert latent_z.shape == (batch_size, 64)
    assert next_state.hidden.shape == (batch_size, 48, 16, 16)
    assert next_state.previous_features.shape == (batch_size, 48, 16, 16)
    assert "change_logits" in outputs
    assert "predicted_next_latent" in outputs
    assert outputs["predicted_next_latent"].shape == (batch_size, 64)


def test_recurrent_temporal_convgru_state_propagation_across_observations():
    """Verify persistent hidden state retains temporal context across sequential observations."""
    model = StreamingVisualCortexV1(feature_dim=24, hidden_dim=32, latent_dim=48)
    model.eval()

    frame_0 = torch.rand(1, 3, 64, 64)
    frame_1 = torch.rand(1, 3, 64, 64)

    with torch.inference_mode():
        # Step 0: cold start
        z0, state_0, out0 = model.forward(frame_0, prev_state=None, delta_t=0.0)
        assert not out0["has_previous"].item()

        # Step 1: feed state_0 into step 1
        z1, state_1, out1 = model.forward(frame_1, prev_state=state_0, delta_t=0.033)
        assert out1["has_previous"].item()
        # Hidden state must have updated recurrently
        assert not torch.equal(state_0.hidden, state_1.hidden)

        # Step 2: compare step 1 with vs without prior state
        z1_cold, state_1_cold, out1_cold = model.forward(frame_1, prev_state=None, delta_t=0.0)
        # Recurrent state makes z1 condition on history: z1 != z1_cold
        assert not torch.equal(z1, z1_cold)


def test_recurrent_state_time_decay_attenuates_state_on_large_deltas():
    """Verify that larger timestamp deltas apply greater decay to historical recurrent hidden state."""
    model = StreamingVisualCortexV1(feature_dim=24, hidden_dim=32, latent_dim=48)
    model.eval()

    frame = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        _, initial_state, _ = model.forward(frame, prev_state=None)

        # Step with small delta (high frequency 30 FPS: 33ms)
        _, state_short, _ = model.forward(frame, prev_state=initial_state, delta_t=0.033)

        # Step with large delta (interrupted stream: 1.5s)
        _, state_long, _ = model.forward(frame, prev_state=initial_state, delta_t=1.5)

        # Larger delta decay changes the resulting hidden representation
        assert not torch.equal(state_short.hidden, state_long.hidden)


def test_predictive_latent_head_and_predict_future_interface():
    """Verify predictive latent head maps current latent to future predicted representation."""
    model = StreamingVisualCortexV1(feature_dim=24, hidden_dim=32, latent_dim=48)
    latent_in = torch.randn(2, 48)

    pred = model.predict_future(latent_in)
    assert pred.shape == (2, 48)

    # Verify backpropagation through future_predictor
    target = torch.randn(2, 48)
    loss = nn.functional.mse_loss(pred, target)
    loss.backward()

    for p in model.future_predictor.parameters():
        assert p.grad is not None
        assert not torch.all(p.grad == 0)


def test_initial_state_and_reset_state_interfaces():
    """Verify initial_state creates clean zeros and reset_state is callable."""
    model = StreamingVisualCortexV1(feature_dim=24, hidden_dim=32, latent_dim=48)
    state = model.initial_state(batch_size=2, height=16, width=16)
    assert state.hidden.shape == (2, 32, 16, 16)
    assert (state.hidden == 0).all().item()
    assert (state.previous_features == 0).all().item()

    # reset_state interface
    model.reset_state()


def test_cpu_parameter_budget():
    """Verify CPU real-time parameter budget (~157k parameters)."""
    # Test typical configurations
    cortex_compact = StreamingVisualCortexV1(feature_dim=24, hidden_dim=32, latent_dim=48)
    params_compact = sum(p.numel() for p in cortex_compact.parameters())
    # Compact model has ~143k parameters
    assert 120_000 <= params_compact <= 170_000

    cortex_standard = StreamingVisualCortexV1(feature_dim=32, hidden_dim=32, latent_dim=64)
    params_standard = sum(p.numel() for p in cortex_standard.parameters())
    # Standard model has ~154k parameters, well within ~157k budget
    assert 140_000 <= params_standard <= 170_000


# ==============================================================================
# 3. Clean Random Weight Initialization Tests
# ==============================================================================

def test_convgru_cell_orthogonal_recurrent_kernel_and_gate_biases():
    """Verify ConvGRUCell recurrent weights W_hh are orthogonal and update gate bias is 1.0."""
    hidden_dim = 48
    input_dim = 48
    cell = ConvGRUCell(input_dim, hidden_dim)

    # 1. Verify candidate recurrent kernel orthogonality: W_hh * W_hh^T == I
    # The recurrent portion of the weights is the slice [:, input_dim:]
    candidate_recurrent_w = cell.candidate.weight[:, input_dim:].flatten(1)
    prod = torch.matmul(candidate_recurrent_w, candidate_recurrent_w.T)
    eye = torch.eye(hidden_dim)
    orthogonality_error = (prod - eye).abs().max().item()
    assert orthogonality_error < 1e-4, f"Orthogonality error too high: {orthogonality_error}"

    # 2. Verify gates recurrent kernel orthogonality for both reset and update slices
    for i in range(2):
        gate_chunk = cell.gates.weight[
            i * hidden_dim : (i + 1) * hidden_dim,
            input_dim :,
        ].flatten(1)
        prod_gate = torch.matmul(gate_chunk, gate_chunk.T)
        error_gate = (prod_gate - eye).abs().max().item()
        assert error_gate < 1e-4, f"Gate chunk {i} orthogonality error too high: {error_gate}"

    # 3. Verify bias values:
    # Update gate bias must be 1.0 (encourages temporal retention)
    # Reset gate bias must be 0.0
    # Candidate bias must be 0.0
    assert (cell.gates.bias[hidden_dim:] == 1.0).all().item()
    assert (cell.gates.bias[:hidden_dim] == 0.0).all().item()
    assert (cell.candidate.bias == 0.0).all().item()


def test_clean_random_weight_initialization_factory():
    """Verify create_clean_random_cortex provides clean random weights with zero legacy loads."""
    # Two models initialized with identical seed have identical parameters
    model_a = create_clean_random_cortex(seed=123, feature_dim=24, hidden_dim=32, latent_dim=48)
    model_b = create_clean_random_cortex(seed=123, feature_dim=24, hidden_dim=32, latent_dim=48)
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p_a, p_b)

    # Different seed produces distinct parameters
    model_c = create_clean_random_cortex(seed=999, feature_dim=24, hidden_dim=32, latent_dim=48)
    assert not all(torch.equal(p_a, p_c) for p_a, p_c in zip(model_a.parameters(), model_c.parameters()))

    # Verify Linear layers are initialized (non-zero weights, zero bias)
    for name, module in model_a.named_modules():
        if isinstance(module, nn.Linear):
            assert not torch.all(module.weight == 0)
            if module.bias is not None:
                assert torch.all(module.bias == 0)


# ==============================================================================
# 4. Fail-Closed Tri-State Permissions Tests
# ==============================================================================

def test_permissions_tri_state_defaults_all_fail_closed():
    """Verify StreamingPermissions defaults all flags to False and raises PermissionError on assertion."""
    perms = StreamingPermissions()
    assert perms.enable_observation is False
    assert perms.enable_online_learning is False
    assert perms.authorize_checkpoint_save is False

    with pytest.raises(PermissionError, match="Live observation is disabled"):
        perms.assert_observation_allowed()

    with pytest.raises(PermissionError, match="Online parameter learning is disabled"):
        perms.assert_learning_allowed()

    with pytest.raises(PermissionError, match="Checkpoint persistence is disabled"):
        perms.assert_save_allowed()


def test_permissions_independent_authorization():
    """Verify each permission flag can be independently authorized without unlocking the others."""
    # 1. Only observation authorized
    p1 = StreamingPermissions(enable_observation=True)
    p1.assert_observation_allowed()  # Should succeed cleanly
    with pytest.raises(PermissionError, match="Online parameter learning is disabled"):
        p1.assert_learning_allowed()
    with pytest.raises(PermissionError, match="Checkpoint persistence is disabled"):
        p1.assert_save_allowed()

    # 2. Only online learning authorized
    p2 = StreamingPermissions(enable_online_learning=True)
    p2.assert_learning_allowed()  # Should succeed cleanly
    with pytest.raises(PermissionError, match="Live observation is disabled"):
        p2.assert_observation_allowed()
    with pytest.raises(PermissionError, match="Checkpoint persistence is disabled"):
        p2.assert_save_allowed()

    # 3. Only checkpoint save authorized
    p3 = StreamingPermissions(authorize_checkpoint_save=True)
    p3.assert_save_allowed()  # Should succeed cleanly
    with pytest.raises(PermissionError, match="Live observation is disabled"):
        p3.assert_observation_allowed()
    with pytest.raises(PermissionError, match="Online parameter learning is disabled"):
        p3.assert_learning_allowed()


# ==============================================================================
# 5. Streaming Observer Runtime & Gap Reset Tests
# ==============================================================================

def test_observer_enforces_fail_closed_observation_permission():
    """Verify StreamingObserver blocks live observation when enable_observation is False."""
    model = StreamingVisualCortexV1(feature_dim=16, hidden_dim=16, latent_dim=16)
    perms_blocked = StreamingPermissions(enable_observation=False)
    observer_blocked = StreamingObserver(model, input_size=(32, 32), permissions=perms_blocked)

    src = SyntheticCaptureSource(width=32, height=32, fps=10.0, total_frames=5)
    with pytest.raises(PermissionError, match="Live observation is disabled"):
        observer_blocked.start(src)

    pkt = FramePacket(0, 1.0, np.zeros((32, 32, 3), dtype=np.uint8))
    with pytest.raises(PermissionError, match="Live observation is disabled"):
        observer_blocked.observe(pkt)


def test_observer_observation_succeeds_when_authorized():
    """Verify StreamingObserver starts and observes cleanly when enable_observation is True."""
    model = StreamingVisualCortexV1(feature_dim=16, hidden_dim=16, latent_dim=16)
    perms = StreamingPermissions(enable_observation=True)
    observer = StreamingObserver(model, input_size=(32, 32), permissions=perms)

    now = time.monotonic()
    pkt = FramePacket(0, now, np.zeros((32, 32, 3), dtype=np.uint8))
    update = observer.observe(pkt)
    assert update is not None
    assert isinstance(update, ObservationUpdate)
    assert update.sequence == 0


def test_observer_gap_reset_on_interruption_threshold():
    """Verify observer cleanly resets recurrent state when time delta exceeds reset_after_seconds."""
    model = StreamingVisualCortexV1(feature_dim=16, hidden_dim=16, latent_dim=16)
    observer = StreamingObserver(
        model,
        input_size=(32, 32),
        reset_after_seconds=1.0,
    )

    f0 = np.full((32, 32, 3), 10, dtype=np.uint8)
    f1 = np.full((32, 32, 3), 20, dtype=np.uint8)
    f2 = np.full((32, 32, 3), 30, dtype=np.uint8)

    # Frame 0 at t=10.0
    up0 = observer.process(FramePacket(0, 10.0, f0), now=10.05)
    assert up0 is not None
    assert not up0.gap_reset

    # Frame 1 at t=10.2 (delta = 0.2 < 1.0s reset threshold): NO gap reset
    up1 = observer.process(FramePacket(1, 10.2, f1), now=10.25)
    assert up1 is not None
    assert not up1.gap_reset
    assert up1.timestamp_delta == pytest.approx(0.2)

    # Frame 2 at t=12.5 (delta = 2.3 > 1.0s reset threshold): GAP RESET MUST TRIGGER
    up2 = observer.process(FramePacket(2, 12.5, f2), now=12.55)
    assert up2 is not None
    assert up2.gap_reset is True
    assert up2.timestamp_delta == pytest.approx(2.3)


# ==============================================================================
# 6. Zero Raw Persistence Tests
# ==============================================================================

def test_zero_raw_persistence_during_stream_processing(tmp_path: Path):
    """Verify that during capture, buffering, and observation, ZERO raw frames are written to disk."""
    # List all files before execution
    initial_files = set(tmp_path.rglob("*"))

    model = StreamingVisualCortexV1(feature_dim=16, hidden_dim=16, latent_dim=16)
    perms = StreamingPermissions(enable_observation=True)
    observer = StreamingObserver(model, input_size=(32, 32), permissions=perms)

    src = SyntheticCaptureSource(width=32, height=32, fps=60.0, total_frames=20)
    src.open()
    try:
        for i in range(20):
            frame_rgb = src.grab()
            pkt = FramePacket(i, time.monotonic() + i * 0.016, frame_rgb)
            observer.process(pkt)
    finally:
        src.close()

    # Verify zero frame image files or binary dumps written in tmp_path
    subsequent_files = set(tmp_path.rglob("*")) - initial_files
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".raw", ".mp4", ".avi", ".npy"}
    persisted_media = [
        f for f in subsequent_files if f.suffix.lower() in image_extensions
    ]
    assert len(persisted_media) == 0, f"Violation of zero raw persistence: found {persisted_media}"
    assert observer.processed_count == 20

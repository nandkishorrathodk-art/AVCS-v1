"""Optional ONNX Runtime accelerator for the HumanEye cortical forward pass.

This is an opt-in performance path, not a replacement. It exports the exact
same computation the eager path runs -- cortical backbone, ventral semantic
stream and the affordance head -- into a single fused graph, and runs that
graph instead of three separate ``nn.Module`` calls per frame. The win comes
from removing per-frame Python/torch dispatch overhead, not from changing what
the model computes.

Semantics are pinned by a parity check at export time: the accelerator runs
both the eager modules and the exported graph on the same input and refuses to
build if they disagree beyond ``atol``. A graph that does not reproduce the
eager outputs is rejected rather than silently used.

Requires the optional ``onnx`` and ``onnxruntime`` packages.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


class OnnxCortexAccelerator:
    """Runs backbone + ventral + affordance head as one fused ONNX graph."""

    def __init__(self, session: Any, input_name: str, output_names: Sequence[str]) -> None:
        self._sess = session
        self._input_name = input_name
        self._output_names = list(output_names)

    # -- construction -----------------------------------------------------
    @classmethod
    def from_eye(
        cls,
        eye: torch.nn.Module,
        input_size: Tuple[int, int] = (128, 128),
        atol: float = 1e-4,
        path: Optional[str] = None,
        num_threads: int = 1,
    ) -> "OnnxCortexAccelerator":
        """Exports ``eye``'s cortical forward pass and verifies output parity.

        Raises RuntimeError if the exported graph disagrees with the eager
        modules on a probe input, so a broken export can never be used.
        """
        try:
            import onnx  # noqa: F401
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - exercised by callers
            raise RuntimeError(
                "The ONNX accelerator requires the optional 'onnx' and "
                "'onnxruntime' packages. Install with: "
                "pip install 'avcs-sdk[onnx]'"
            ) from exc

        import tempfile

        fused = _FusedCortex(eye.backbone, eye.ventral, eye.affordance_head)
        fused.eval()

        if path is None:
            path = tempfile.mktemp(suffix=".onnx", prefix="avcs_cortex_")

        dummy = torch.randn(1, 3, *input_size)
        with torch.inference_mode():
            torch.onnx.export(
                fused,
                dummy,
                path,
                input_names=["pixels"],
                output_names=["category_logits", "affordance_logits", "features"],
                opset_version=17,
                dynamic_axes={
                    "pixels": {2: "h", 3: "w"},
                    "category_logits": {0: "batch"},
                    "affordance_logits": {0: "batch"},
                    "features": {0: "batch"},
                },
                dynamo=False,
            )

        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            path, options, providers=["CPUExecutionProvider"]
        )

        accel = cls(
            session,
            session.get_inputs()[0].name,
            [o.name for o in session.get_outputs()],
        )

        # -- parity gate: refuse a graph that does not reproduce eager output
        probe = torch.randn(1, 3, *input_size)
        with torch.inference_mode():
            eager_cat, eager_aff, eager_feat = fused(probe)
        onnx_cat, onnx_aff, onnx_feat = accel.run_torch(probe)
        for name, eager, got in (
            ("category_logits", eager_cat, onnx_cat),
            ("affordance_logits", eager_aff, onnx_aff),
            ("features", eager_feat, onnx_feat),
        ):
            if eager.shape != got.shape or not torch.allclose(
                eager, got, atol=atol, rtol=0.0
            ):
                raise RuntimeError(
                    f"ONNX export failed parity check on '{name}': "
                    f"max abs diff "
                    f"{float((eager - got).abs().max()) if eager.shape == got.shape else 'shape mismatch'}"
                )

        return accel

    # -- inference --------------------------------------------------------
    def run(self, pixels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``pixels``: NCHW float32 numpy in [0,1].

        Returns (category_logits, affordance_logits, features).
        """
        cat, aff, feat = self._sess.run(self._output_names, {self._input_name: pixels})
        return cat, aff, feat

    def run_torch(self, pixels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Torch-in/torch-out wrapper used for the parity check and for callers
        that already hold a normalized tensor."""
        arr = pixels.detach().cpu().numpy()
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        cat, aff, feat = self.run(arr)
        return torch.from_numpy(cat), torch.from_numpy(aff), torch.from_numpy(feat)


class _FusedCortex(torch.nn.Module):
    """Backbone -> ventral category logits + affordance logits, as one graph."""

    def __init__(self, backbone: torch.nn.Module, ventral: torch.nn.Module, affordance_head: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.ventral = ventral
        self.affordance_head = affordance_head

    def forward(self, pixels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.backbone(pixels)
        category_logits = self.ventral(feat)["category_logits"]
        affordance_logits = self.affordance_head(feat)
        return category_logits, affordance_logits, feat

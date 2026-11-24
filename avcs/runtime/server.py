"""AVCS Low-Latency VLA Inference Server Module.

Re-exports the optimized production implementation from `avcs.runtime.vla_server`
to ensure complete backward compatibility with all existing callers and tests.
"""
from __future__ import annotations

from avcs.runtime.vla_server import (
    ASSISTANT_PREFILL,
    ActRequest,
    ActResponse,
    AsyncInferencePipeline,
    app,
    get_assistant_prefill,
    get_fast_engine,
    http_act,
    load_vla_model,
    run_inference_with_prefill,
    set_cuda_graph_runner,
    set_fast_engine,
    set_inference_handler,
    websocket_act,
)
from avcs.runtime.turbo_jpeg import (
    decode_turbo_jpeg,
    decode_turbo_jpeg_to_numpy,
    encode_turbo_jpeg,
)

__all__ = [
    "ASSISTANT_PREFILL",
    "ActRequest",
    "ActResponse",
    "AsyncInferencePipeline",
    "app",
    "decode_turbo_jpeg",
    "decode_turbo_jpeg_to_numpy",
    "encode_turbo_jpeg",
    "get_assistant_prefill",
    "get_fast_engine",
    "http_act",
    "load_vla_model",
    "run_inference_with_prefill",
    "set_cuda_graph_runner",
    "set_fast_engine",
    "set_inference_handler",
    "websocket_act",
]

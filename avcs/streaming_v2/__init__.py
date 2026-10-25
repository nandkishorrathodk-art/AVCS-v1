"""StreamingVisualCortexV2: End-to-end Pure-Pixel Spatiotemporal UI Perception Model.

Exposes the 4-channel spatial preprocessor, multi-scale temporal ConvGRU backbone,
and 3 specialized perception heads (Surprise/Spinner, 5-Class UI Detection, and
Visual Ripple Action Response).
"""

from avcs.streaming_v2.preprocessor import (
    EdgeLuminancePreprocessor,
    compute_edge_map,
    compute_edge_iou,
    photometric_inversion,
)
from avcs.streaming_v2.heads import (
    TemporalChangeSpinnerHead,
    GenericUIRegionDetectorHead,
    PostActionVisualResponseHead,
)
from avcs.streaming_v2.model import (
    SpatialPyramidStem,
    ConvGRUCell,
    StreamingVisualCortexV2,
    CortexV2Output,
    CortexTemporalState,
    System2SymbolicSummary,
)
from avcs.streaming_v2.integration import (
    StreamingV2Adapter,
    StreamingV2Observation,
    load_streaming_v2_weights,
)

__all__ = [
    "EdgeLuminancePreprocessor",
    "compute_edge_map",
    "compute_edge_iou",
    "photometric_inversion",
    "TemporalChangeSpinnerHead",
    "GenericUIRegionDetectorHead",
    "PostActionVisualResponseHead",
    "SpatialPyramidStem",
    "ConvGRUCell",
    "StreamingVisualCortexV2",
    "CortexV2Output",
    "CortexTemporalState",
    "System2SymbolicSummary",
    "StreamingV2Adapter",
    "StreamingV2Observation",
    "load_streaming_v2_weights",
]


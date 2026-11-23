"""Low-Latency Turbo JPEG Compression for AVCS Thin-Client Streaming.

Optimized for:
1. Encoding speed: <15ms on 1280x800 desktop frames.
2. Compressed payload: <35KB for real-time cloud dispatch.
3. Single-pass 4:2:0 chroma subsampling for low transmission latency.
"""
from __future__ import annotations

import io
from typing import Optional, Union
import numpy as np
from PIL import Image

try:
    import simplejpeg
except ImportError:
    simplejpeg = None


def encode_turbo_jpeg(
    frame: Union[np.ndarray, Image.Image],
    quality: int = 75,
) -> bytes:
    """Encodes an RGB frame to JPEG bytes using fast single-pass Turbo quantization.

    Args:
        frame: RGB image as numpy uint8 array (H, W, 3) or PIL Image.
        quality: JPEG compression quality (default 75, yields <35KB for 1280x800).

    Returns:
        Raw JPEG bytes.
    """
    if simplejpeg is not None and isinstance(frame, np.ndarray):
        try:
            return simplejpeg.encode_jpeg(
                frame,
                quality=quality,
                colorspace="RGB",
                fastdct=True,
            )
        except Exception:
            pass

    if isinstance(frame, np.ndarray):
        img = Image.fromarray(frame)
    else:
        img = frame

    buffer = io.BytesIO()
    # Using optimize=False, subsampling=2 (4:2:0) activates single-pass libjpeg-turbo encoding in Pillow
    img.save(buffer, format="JPEG", quality=quality, optimize=False, subsampling=2)
    return buffer.getvalue()


def decode_turbo_jpeg(jpeg_bytes: bytes) -> Image.Image:
    """Fast decoding of JPEG bytes to PIL RGB Image."""
    if simplejpeg is not None:
        try:
            arr = simplejpeg.decode_jpeg(jpeg_bytes, colorspace="RGB", fastdct=True)
            return Image.fromarray(arr)
        except Exception:
            pass
    return Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")

# Department: Protocol (`avcs.protocol`)

## 1. Responsibilities & Boundaries
- Defines standard wire formats, schemas, action specifications (`Action`, `ActionKind`), and coordinate conversion logic.
- Maintains canonical coordinate representations (e.g. 1000-grid normalized vs pixel coordinates) and HID tables.
- Pure protocol data contracts; encapsulates serialization, deserialization, and boundary validation.

## 2. Allowed Imports
- Standard Library ONLY (`typing`, `dataclasses`, `enum`, `json`, `math`, `re`, `pathlib`).
- Pure Python; zero operating system actuation or heavy ML framework dependencies.

## 3. Forbidden Imports
- NEVER import Windows OS actuation modules (`win32gui`, `win32api`, `ctypes.windll`, `avcs.os_io`).
- NEVER import deep learning frameworks (`torch`, `torchvision`, `avcs.models`, `avcs.train`).
- NEVER import runtime orchestration engines (`avcs.runtime`).

## 4. Sub-Agent Contract
- Single source of truth for action payloads, coordinate formats, and wire schemas.
- All departments (`runtime`, `data`, `models`, `eval`) MUST consume `avcs.protocol` data structures rather than redefining actions or schemas locally.

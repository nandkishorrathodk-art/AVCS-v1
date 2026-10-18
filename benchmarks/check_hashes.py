import hashlib
import torch
from pathlib import Path

files = [
    "runs/weights/candidates/candidate_m3_highres_v2.pt",
    "runs/weights/candidates/candidate_m3_reranker_v1.pt",
    "runs/weights/candidates/candidate_m3_foveal_v1.pt",
]

print("=== CHECKPOINT VERIFICATION LEDGER ===")
for f in files:
    p = Path(f)
    if not p.exists():
        print(f"MISSING: {f}")
        continue
    data = p.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    ckpt = torch.load(f, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        params = sum(t.numel() for t in ckpt["model_state_dict"].values())
        schema = ckpt.get("schema_version", "legacy")
    elif isinstance(ckpt, dict) and "reranker_state_dict" in ckpt:
        params = sum(t.numel() for t in ckpt["reranker_state_dict"].values())
        schema = ckpt.get("schema_version", "legacy")
    elif isinstance(ckpt, dict):
        params = sum(t.numel() for t in ckpt.values() if isinstance(t, torch.Tensor))
        schema = ckpt.get("schema_version", "legacy")
    else:
        params = 0
        schema = "unknown"
    print(f"File: {f}")
    print(f"  Size: {len(data)} bytes")
    print(f"  SHA-256: {sha}")
    print(f"  Parameters: {params:,}")
    print(f"  Schema: {schema}")

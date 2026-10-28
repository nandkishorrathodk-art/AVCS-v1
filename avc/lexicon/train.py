"""Train the shared identity space on CPU.

The objective is symmetric InfoNCE between the two branches:

    patch -> its label's text embedding      (vision learns to meet language)
    word  -> its own patch embedding         (language learns to meet vision)

Both directions are needed. Training only one leaves the space asymmetric, and
the inverse direction (patch -> name) is the whole point of the stream.

Reported metrics are deliberately split:

- ``closed_vocab`` on held-out *samples* of seen fonts
- ``held_out_font`` on a font family never trained on

The second number is the one that matters. Text recognition that only works on
the training font is a template, not recognition, and this repo has already been
burned once by a metric that flattered a model that had memorised its generator.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from avc.lexicon.encoder import LexiconEncoder
from avc.lexicon.render import make_corpus, split_fonts
from avc.lexicon.vocab import LABELS

DEFAULT_OUT = "runs/weights/avc_lexicon.pt"


def _split_corpus(
    labels: Sequence[str], fonts: Sequence[str], per_label: int, seed: int, augment: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    patches, targets = make_corpus(
        labels=labels, fonts=fonts, per_label=per_label, seed=seed, augment=augment
    )
    return torch.from_numpy(patches), torch.from_numpy(targets)


def _accuracy(model: LexiconEncoder, patches: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        emb = model.encode_patches(patches)
        sims = emb @ model.prototypes.t()
        top1 = sims.argmax(dim=1)
        acc = float((top1 == targets).float().mean())
        # Rejection view: how often the *correct* label is at least the floor.
        best = sims.gather(1, targets[:, None]).squeeze(1)
        accepted = float((best >= 0.55).float().mean())
    return {"top1": acc, "accepted": accepted, "n": int(patches.shape[0])}


def train(
    steps: int = 600,
    per_label: int = 6,
    batch_size: int = 64,
    lr: float = 3e-3,
    seed: int = 7,
    temperature: float = 0.07,
    out: str = DEFAULT_OUT,
    verbose: bool = True,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_fonts, held_fonts = split_fonts()
    if not train_fonts:
        raise RuntimeError("no TrueType fonts available to render a training corpus")

    labels = list(LABELS)
    model = LexiconEncoder(labels)
    model.rebuild_prototypes()

    x_train, y_train = _split_corpus(labels, train_fonts, per_label, seed, augment=True)
    x_val, y_val = _split_corpus(labels, train_fonts, 2, seed + 1)
    x_font, y_font = (
        _split_corpus(labels, held_fonts, 2, seed + 2) if held_fonts else (None, None)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, steps))
    rng = np.random.default_rng(seed)

    started = time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        idx = rng.integers(0, x_train.shape[0], size=min(batch_size, x_train.shape[0]))
        patches = x_train[idx]
        words = [labels[int(t)] for t in y_train[idx]]

        patch_emb = model.encode_patches(patches)
        text_emb = model.encode_words(words)
        # Anchor the prototypes to the labels the batch actually uses.
        targets = torch.tensor([int(t) for t in y_train[idx]], dtype=torch.long)

        logits_p2t = patch_emb @ text_emb.t() / temperature
        logits_t2p = text_emb @ patch_emb.t() / temperature
        labels_batch = torch.arange(patches.shape[0])
        loss = 0.5 * (
            F.cross_entropy(logits_p2t, labels_batch)
            + F.cross_entropy(logits_t2p, labels_batch)
        )
        # Keep the patch embedding near its own class prototype so the readout
        # at inference (which uses prototypes, not batch text) is calibrated.
        proto = model.prototypes[targets]
        loss = loss + 0.5 * (1.0 - (patch_emb * proto).sum(dim=1)).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if verbose and (step % max(1, steps // 6) == 0 or step == steps):
            print(f"  step {step:4d}/{steps}  loss={float(loss):.4f}")

    model.eval()
    model.rebuild_prototypes()
    report: Dict[str, object] = {
        "steps": steps,
        "labels": len(labels),
        "train_fonts": [Path(f).name for f in train_fonts],
        "held_out_fonts": [Path(f).name for f in held_fonts],
        "closed_vocab": _accuracy(model, x_val, y_val),
        "wall_seconds": round(time.perf_counter() - started, 2),
    }
    if x_font is not None:
        report["held_out_font"] = _accuracy(model, x_font, y_font)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": labels,
            "metrics": report,
        },
        out,
    )
    report["checkpoint"] = out
    if verbose:
        print(json.dumps(report, indent=2))
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Train the AVC Lexicon shared identity space")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--per-label", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    train(
        steps=args.steps,
        per_label=args.per_label,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        out=args.out,
    )


if __name__ == "__main__":
    main()

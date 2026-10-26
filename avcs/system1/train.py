"""Train the one model.

One loop, one optimizer, one loss, all tasks together. That is the whole point:
the gradients from every task meet in the shared trunk, so the trunk cannot
specialise into seven private front-ends.

    python3 -m avcs.system1.train --steps 400 --out runs/weights/system1_one_model.pt
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from avcs.system1.dataset import System1OneModelDataset, collate
from avcs.system1.losses import OneModelLoss
from avcs.system1.model import OneModel, OneModelConfig


def _seed(seed: int) -> None:
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model: OneModel, loader: DataLoader, device: Optional[torch.device] = None) -> dict:
    dev = device or next(model.parameters()).device
    model.eval()
    loss_fn = OneModelLoss()
    cat_ok = cat_n = 0
    point_err = []
    presence_ok = presence_n = 0
    kind_ok = kind_n = 0
    total = 0.0
    batches = 0
    for batch in loader:
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch["image"], batch["query"])
        _, br = loss_fn(out, batch)
        total += br.total
        batches += 1

        cat_ok += int((out.category_index() == batch["category"]).sum())
        cat_n += batch["category"].numel()

        occupied = batch["objectness"] > 0.5
        if occupied.any():
            kind_pred = out.kind_logits.argmax(dim=-1)
            kind_ok += int((kind_pred[occupied] == batch["kind"][occupied]).sum())
            kind_n += int(occupied.sum())

        pred_present = out.presence_logits.squeeze(-1) > 0
        true_present = batch["present"] > 0.5
        presence_ok += int((pred_present == true_present).sum())
        presence_n += true_present.numel()

        if true_present.any():
            p = out.point_norm()[true_present]
            g = model.config.grid
            ty = (batch["point_cell"][true_present] // g).float() / (g - 1)
            tx = (batch["point_cell"][true_present] % g).float() / (g - 1)
            err = torch.sqrt((p[:, 0] - tx) ** 2 + (p[:, 1] - ty) ** 2)
            point_err.extend(err.tolist())
    model.train()
    point_err.sort()
    return {
        "loss": total / max(1, batches),
        "category_acc": cat_ok / max(1, cat_n),
        "kind_acc": kind_ok / max(1, kind_n),
        "presence_acc": presence_ok / max(1, presence_n),
        "point_median_err": point_err[len(point_err) // 2] if point_err else float("nan"),
        "point_within_0.1": (
            sum(1 for e in point_err if e <= 0.1) / len(point_err) if point_err else float("nan")
        ),
        "n_point_eval": len(point_err),
    }


def train(
    steps: int = 400,
    batch_size: int = 8,
    lr: float = 2e-3,
    seed: int = 7,
    out: str = "runs/weights/system1_one_model.pt",
    log_every: int = 50,
    width: int = 64,
    latent_dim: int = 128,
    device: Optional[str] = None,
    amp: bool = True,
    corpus_root: Optional[str] = None,
) -> dict:
    _seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = OneModelConfig(width=width, latent_dim=latent_dim)
    model = OneModel(cfg).to(dev)
    loss_fn = OneModelLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    use_amp = bool(amp and dev.type == "cuda")

    root = corpus_root
    train_ds = System1OneModelDataset(root=root, split="train", grid=cfg.grid, image_size=cfg.input_size)
    val_ds = System1OneModelDataset(root=root, split="validation", grid=cfg.grid, image_size=cfg.input_size)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate,
        num_workers=2, drop_last=True, persistent_workers=True,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    print(f"device={dev.type}  amp={use_amp}")
    print(f"one model: {model.total_parameters:,} params  (width={width}, latent={latent_dim})")
    print(f"train frames={len(train_ds)}  val frames={len(val_ds)}")

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    step = 0
    t0 = time.perf_counter()
    last = {}
    while step < steps:
        for batch in train_loader:
            if step >= steps:
                break
            batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(batch["image"], batch["query"])
                loss, br = loss_fn(pred, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            last = br.as_dict()
            if step % log_every == 0 or step == 1:
                dt = time.perf_counter() - t0
                print(
                    f"  step {step:>5}/{steps}  loss={br.total:6.3f}  "
                    f"cat={br.category:5.3f} obj={br.objectness:5.3f} "
                    f"kind={br.kind:5.3f} word={br.word:5.3f} "
                    f"pt={br.point:5.3f} pres={br.presence:5.3f}  "
                    f"({dt:.1f}s, {step/dt:.1f} step/s)",
                    flush=True,
                )

    metrics = evaluate(model, val_loader, dev)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "steps": steps,
            "val_metrics": metrics,
            "final_train_loss": last,
        },
        out,
    )
    print("\nval metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nsaved -> {out}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="runs/weights/system1_one_model.pt")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--corpus-root", default=None)
    a = ap.parse_args()
    train(
        a.steps, a.batch_size, a.lr, a.seed, a.out, a.log_every,
        width=a.width, latent_dim=a.latent_dim, device=a.device,
        amp=not a.no_amp, corpus_root=a.corpus_root,
    )


if __name__ == "__main__":
    main()

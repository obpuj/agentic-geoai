"""SegFormer Bengaluru adaptation — the Gate C fallback (and possibly winner).

Run on Kaggle GPU:
    python train_segformer.py --chips /kaggle/input/auie-chips/chips \
        --out /kaggle/working/segformer_blr --epochs 40 --bands all

Design decisions:
  * MiT-B0 pretrained; first patch-embed conv adapted from 3 to 7 channels
    by averaging pretrained RGB kernels into every new channel (standard
    multispectral adaptation; preserves pretrained scale).
  * Weighted CE from the manifest (anti-collapse for the 70/25/2/2 skew).
  * Metric = per-class IoU + mIoU on val, computed on deterministic tiles.
    Best checkpoint by val mIoU. Pixel accuracy is reported but NEVER used
    for selection — 70% built-up makes it meaningless.
  * Same dataset/normalization/crops as the Prithvi run: Gate C fairness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from dataset import (ALL_BANDS, IGNORE_INDEX, NUM_CLASSES, RGB_BANDS,
                     ChipDataset, class_weights_from_manifest)

PRETRAINED = "nvidia/mit-b0"


def build_model(in_channels: int) -> SegformerForSemanticSegmentation:
    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)
    if in_channels != 3:
        old = model.segformer.encoder.patch_embeddings[0].proj
        new = nn.Conv2d(in_channels, old.out_channels,
                        kernel_size=old.kernel_size, stride=old.stride,
                        padding=old.padding)
        with torch.no_grad():
            mean_kernel = old.weight.mean(dim=1, keepdim=True)  # avg RGB
            new.weight.copy_(mean_kernel.repeat(1, in_channels, 1, 1))
            if old.bias is not None:
                new.bias.copy_(old.bias)
        model.segformer.encoder.patch_embeddings[0].proj = new
        model.config.num_channels = in_channels
    return model


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    inter = np.zeros(NUM_CLASSES)
    union = np.zeros(NUM_CLASSES)
    correct = total = 0
    for img, mask in loader:
        img = img.to(device)
        logits = model(pixel_values=img).logits
        logits = nn.functional.interpolate(
            logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        pred = logits.argmax(1).cpu().numpy()
        gt = mask.numpy()
        valid = gt != IGNORE_INDEX
        correct += (pred[valid] == gt[valid]).sum()
        total += valid.sum()
        for c in range(NUM_CLASSES):
            p, g = (pred == c) & valid, (gt == c) & valid
            inter[c] += (p & g).sum()
            union[c] += (p | g).sum()
    iou = inter / np.maximum(union, 1)
    return {"per_class_iou": iou.round(4).tolist(),
            "miou": round(float(iou.mean()), 4),
            "pixel_acc_do_not_select_on_this": round(float(correct / max(total, 1)), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--bands", choices=["all", "rgb"], default="all")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    chips, out = Path(args.chips), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bands = ALL_BANDS if args.bands == "all" else RGB_BANDS
    stats = chips / "stats.json"

    train_ds = ChipDataset(chips, "train", stats, bands, augment=True,
                           samples_per_chip=6)
    val_ds = ChipDataset(chips, "val", stats, bands, augment=False)
    train = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                       num_workers=2, drop_last=True)
    val = DataLoader(val_ds, batch_size=args.batch, num_workers=2)

    model = build_model(len(bands)).to(device)
    weights = torch.tensor(
        class_weights_from_manifest(chips / "manifest.json")).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE_INDEX)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    print(f"device={device} bands={args.bands} train_samples={len(train_ds)} "
          f"val_samples={len(val_ds)} class_weights={weights.cpu().numpy().round(2)}")

    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for img, mask in train:
            img, mask = img.to(device), mask.to(device)
            logits = model(pixel_values=img).logits
            logits = nn.functional.interpolate(
                logits, size=mask.shape[-2:], mode="bilinear",
                align_corners=False)
            loss = criterion(logits, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()
        metrics = evaluate(model, val, device)
        metrics["epoch"], metrics["train_loss"] = epoch, round(running / len(train), 4)
        history.append(metrics)
        print(metrics)
        if metrics["miou"] > best:
            best = metrics["miou"]
            model.save_pretrained(out / "best")
            (out / "best_metrics.json").write_text(json.dumps(metrics, indent=1))
    (out / "history.json").write_text(json.dumps(history, indent=1))
    print(f"best val mIoU: {best}  -> {out/'best'}")


if __name__ == "__main__":
    main()

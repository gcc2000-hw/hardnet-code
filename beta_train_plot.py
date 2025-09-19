#!/usr/bin/env python3
"""
Parses lines like:
[2025-09-03 15:50:51] INFO: Epoch 67: Train Loss=0.0603, Val Loss=0.0606, Constraint Loss=0.0200, Train Sat=0.7103, Val Sat=0.7116, ...

Outputs:
  - epoch_losses.png
  - epoch_satisfaction.png
  - bounds_batches.png (only if bounds tokens found)
"""
import re, math, argparse
from pathlib import Path
from typing import List, Dict, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPOCH_LINE = re.compile(r"\bEpoch\s+(\d+)\s*:\s*(.*)$", re.IGNORECASE)
KV        = re.compile(r"\s*([^=,]+?)\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*")
BOUND_TOKEN = re.compile(
    r"\b(?:(?:Lower\s*Bound)|LB|lower|min[_\s]*bound|min|lo)\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"|"
    r"\b(?:(?:Upper\s*Bound)|UB|upper|max[_\s]*bound|max|hi)\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    re.IGNORECASE
)

def mfloat(s: str) -> Optional[float]:
    try: return float(s)
    except: return None

def norm(k: str) -> str:
    k = k.strip().lower()
    mapping = {
        "train loss":"train_loss", "val loss":"val_loss", "validation loss":"val_loss",
        "constraint loss":"constraint_loss",
        "train sat":"train_sat", "val sat":"val_sat", "validation sat":"val_sat",
        "lr":"lr"
    }
    return mapping.get(k, k.replace(" ","_"))

def parse_epoch_tail(rest: str) -> Dict[str, float]:
    out={}
    for tok in rest.split(","):
        m = KV.match(tok)
        if not m: continue
        key = norm(m.group(1))
        val = mfloat(m.group(2))
        if val is not None:
            out[key]=val
    return out

def parse_file(path: Path):
    epochs: List[int] = []
    by_ep: Dict[int, Dict[str,float]] = {}

    b_idx: List[int] = []
    lb_vals: List[float] = []
    ub_vals: List[float] = []

    batch_i = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            em = EPOCH_LINE.search(line)
            if em:
                ep = int(em.group(1))
                rest = em.group(2) or ""
                metrics = parse_epoch_tail(rest)
                if metrics:
                    by_ep.setdefault(ep, {}).update(metrics)
                    if ep not in epochs: epochs.append(ep)

            # bounds anywhere
            hit=False; lb=None; ub=None
            for bm in BOUND_TOKEN.finditer(line):
                if bm.group(1) is not None: lb = mfloat(bm.group(1)); hit=True
                if bm.group(2) is not None: ub = mfloat(bm.group(2)); hit=True
            if hit and (lb is not None or ub is not None):
                b_idx.append(batch_i)
                lb_vals.append(lb if lb is not None else math.nan)
                ub_vals.append(ub if ub is not None else math.nan)
                batch_i += 1

    return sorted(set(epochs)), by_ep, b_idx, lb_vals, ub_vals

def series(epochs, by_ep, keys):
    s={k:[] for k in keys}
    for ep in epochs:
        d = by_ep.get(ep, {})
        for k in keys:
            s[k].append(d.get(k, float("nan")))
    return s

def roll(values, w: int):
    if w<=1: return values[:]
    out=[]; q=[]; s=0.0
    for v in values:
        q.append(v)
        if len(q)>w: q.pop(0)
        nn=[x for x in q if not math.isnan(x)]
        out.append(sum(nn)/len(nn) if nn else math.nan)
    return out

def style_ax(ax, title, ylab, xlab="Epoch"):
    ax.set_title(title, fontsize=18, fontweight="bold", pad=10)
    ax.set_xlabel(xlab, fontsize=12)
    ax.set_ylabel(ylab, fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)

def main():
    ap = argparse.ArgumentParser(description="Styled plots for epoch metrics + optional per-batch bounds.")
    ap.add_argument("logfile", type=Path)
    ap.add_argument("--outdir", type=Path, default=Path("."))
    ap.add_argument("--bounds-window", type=int, default=200)
    args = ap.parse_args()

    epochs, by_ep, b_idx, lb, ub = parse_file(args.logfile)
    args.outdir.mkdir(parents=True, exist_ok=True)

    keys = ["train_loss","val_loss","constraint_loss","train_sat","val_sat"]
    s = series(epochs, by_ep, keys)

    # Losses
    fig1, ax1 = plt.subplots(figsize=(10.5,7.5))
    ax1.plot(epochs, s["train_loss"], "-o", linewidth=2, markersize=4, label="Train Loss")
    ax1.plot(epochs, s["val_loss"], "-o", linewidth=2, markersize=4, label="Val Loss")
    ax1.plot(epochs, s["constraint_loss"], "-o", linewidth=2, markersize=4, label="Constraint Loss")
    style_ax(ax1, "Stage 2 — Losses", "Loss")
    ax1.legend(frameon=False)
    fig1.tight_layout(); fig1.savefig(args.outdir/"epoch_losses.png", dpi=180)

    # Satisfaction
    fig2, ax2 = plt.subplots(figsize=(10.5,7.5))
    ax2.plot(epochs, s["train_sat"], "-o", linewidth=2, markersize=4, label="Train Sat")
    ax2.plot(epochs, s["val_sat"], "-o", linewidth=2, markersize=4, label="Val Sat")
    style_ax(ax2, "Stage 2 — Constraint Satisfaction", "Satisfaction")
    ax2.legend(frameon=False)
    fig2.tight_layout(); fig2.savefig(args.outdir/"epoch_satisfaction.png", dpi=180)

    # Bounds (if found)
    if b_idx:
        fig3, ax3 = plt.subplots(figsize=(10.5,7.5))
        ax3.scatter(b_idx, lb, s=8, alpha=0.35, label="Lower Bound (batch)")
        ax3.scatter(b_idx, ub, s=8, alpha=0.35, label="Upper Bound (batch)")
        if args.bounds_window >= 2:
            ax3.plot(b_idx, roll(lb, args.bounds_window), linewidth=2, label=f"LB (rolling {args.bounds_window})")
            ax3.plot(b_idx, roll(ub, args.bounds_window), linewidth=2, label=f"UB (rolling {args.bounds_window})")
        style_ax(ax3, "Bounds — Batches", "Bound value", xlab="Batch index (appearance order)")
        ax3.legend(frameon=False)
        fig3.tight_layout(); fig3.savefig(args.outdir/"bounds_batches.png", dpi=180)
    else:
        print("No bounds found; skipped bounds plot.")

    print(f"Saved plots to {args.outdir.resolve()}")

if __name__ == "__main__":
    main()

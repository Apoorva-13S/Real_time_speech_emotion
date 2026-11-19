"""Confusion matrix, per-class bars and training curves.

matplotlib is optional: everything degrades to printed text if it is missing,
because a training run should never fail just because a plot could not be made.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import metrics


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def confusion(root: Path, y_true: np.ndarray, y_pred: np.ndarray,
              classes: list[str]) -> Path | None:
    plt = _plt()
    m = metrics.confusion_matrix(y_true, y_pred, classes)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (held-out actors)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if m[i, j]:
                ax.text(j, i, str(m[i, j]), ha="center", va="center",
                        fontsize=8,
                        color="white" if m[i, j] > m.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out = root / "reports" / "figures" / "confusion_matrix.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def per_class_bars(root: Path, res: dict, classes: list[str]) -> Path | None:
    plt = _plt()
    recalls = [res["per_class"][c]["recall"] for c in classes]
    f1s = [res["per_class"][c]["f1"] for c in classes]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(classes))
    ax.bar(x - 0.2, recalls, 0.4, label="recall")
    ax.bar(x + 0.2, f1s, 0.4, label="F1")
    ax.set_xticks(x, classes, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.axhline(res["macro_f1"], ls="--", lw=1, color="grey",
               label=f"macro F1 {res['macro_f1']:.2f}")
    ax.set_title("Per-class recall / F1 on held-out actors")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = root / "reports" / "figures" / "per_class.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def curves(root: Path, result: dict) -> Path | None:
    hist = result.get("history") or []
    if not hist:
        return None
    plt = _plt()
    epochs = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [h["train_loss"] for h in hist], "o-", label="train loss")
    ax.plot(epochs, [h["val_macro_f1"] for h in hist], "s-", label="val macro F1")
    ax.plot(epochs, [h["val_accuracy"] for h in hist], "^-", label="val accuracy")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Training history (selection metric: val macro F1)")
    fig.tight_layout()
    out = root / "reports" / "figures" / "training_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def save_all(root: Path, result: dict, y_true: np.ndarray, y_pred: np.ndarray,
             classes: list[str]) -> list[str]:
    root = Path(root)
    made = []
    for fn in (lambda: confusion(root, y_true, y_pred, classes),
               lambda: per_class_bars(root, result["test"], classes),
               lambda: curves(root, result)):
        try:
            p = fn()
            if p:
                made.append(str(p))
        except Exception as exc:                          # noqa: BLE001
            print(f"  plot skipped: {type(exc).__name__}: {exc}")
    if made:
        print("figures: " + ", ".join(made))
    return made

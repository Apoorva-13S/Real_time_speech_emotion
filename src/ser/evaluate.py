"""Evaluate a saved checkpoint on the held-out actors.

Never selects anything: this is the one honest read of the model. If it is
also used to choose a checkpoint, it stops being a test set.

Usage:
  python -m ser.evaluate --model runs/local/best --config configs/local_train.yaml
  python -m ser.evaluate --model runs/local/best --json reports/eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForAudioClassification

from . import config as cfgmod
from . import metrics, splits
from .data import load_any, to_arrays
from .labels import names_for
from .train import make_loader, pick_device


def evaluate(model_dir: str | Path, cfg: dict) -> dict:
    device = pick_device(cfgmod.get(cfg, "run.device", "auto"))
    model_dir = Path(model_dir)
    out_root = Path(cfgmod.get(cfg, "run.output_dir", "runs/local"))

    meta_path = model_dir.parent / "labels.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        classes = meta["classes"]
        window_s = float(meta["window_s"])
    else:
        classes = names_for(int(cfgmod.get(cfg, "model.n_classes", 8)))
        window_s = float(cfgmod.get(cfg, "data.window_s", 4.0))

    print(f"model={model_dir}")
    print(f"classes={classes}  window={window_s}s  device={device}")

    clips = load_any(cfgmod.get(cfg, "data.source", "xbgoose/ravdess"),
                     limit=cfgmod.get(cfg, "data.limit"))
    x, y = to_arrays(clips, classes, window_s)
    actors = np.asarray([c.actor for c in clips], dtype=np.int32)

    sp_path = Path(cfgmod.get(cfg, "data.splits_file",
                              str(out_root / "data/splits.json")))
    if not sp_path.is_file():
        raise FileNotFoundError(
            f"no split file at {sp_path}. Run training first, or point "
            f"data.splits_file at an existing splits.json. Evaluating on the "
            f"training actors would be meaningless.")
    actor_split = splits.load(sp_path)
    idx = splits.indices_for(actors, actor_split["test"])
    if not len(idx):
        raise RuntimeError("test split is empty for this dataset")

    model = AutoModelForAudioClassification.from_pretrained(
        model_dir).to(device).eval()
    loader = make_loader(x[idx], y[idx], int(cfgmod.get(cfg, "train.batch_size", 4)),
                         False)
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(input_values=batch["input_values"].to(device)).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
            trues.extend(batch["labels"].tolist())

    res = metrics.summary(np.asarray(trues), np.asarray(preds), classes)
    print("\n" + metrics.format_report(res, classes))
    print(f"\nmajority-class baseline: {metrics.majority_baseline(np.asarray(trues)):.4f}")
    if res["accuracy"] <= metrics.majority_baseline(np.asarray(trues)):
        print("WARNING: the model does not beat always predicting the most "
              "common class. Do not ship this checkpoint.")
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    ap.add_argument("--model", required=True, help="checkpoint folder")
    ap.add_argument("--config", default=None)
    ap.add_argument("--json", default=None, help="write results here")
    ap.add_argument("--min-macro-f1", type=float, default=None,
                    help="exit 1 if macro F1 is below this (CI gate)")
    args = ap.parse_args(argv)

    cfg = cfgmod.resolve(args.config)
    res = evaluate(args.model, cfg)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, indent=2))
        print(f"wrote {args.json}")

    if args.min_macro_f1 is not None:
        ok, reasons = metrics.gate(res, min_macro_f1=args.min_macro_f1)
        if not ok:
            print("\nCI GATE FAILED:")
            for r in reasons:
                print(f"  - {r}")
            return 1
        print(f"\nCI gate passed (macro F1 {res['macro_f1']:.4f} >= "
              f"{args.min_macro_f1:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

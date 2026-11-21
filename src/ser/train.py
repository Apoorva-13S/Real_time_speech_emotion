"""Fine-tune wav2vec2 for speech emotion recognition.

Designed for three environments with the same code:
  * Google Colab free GPU  (configs/colab_train.yaml)
  * a laptop CPU           (configs/local_train.yaml -- works, slowly)
  * GitHub Actions         (configs/ci_smoke.yaml -- 3 minutes, 12 clips)

Fixes carried over from the original notebook:
  * LR warmup + cosine decay (no more raw AdamW at a fixed LR)
  * GradScaler for fp16 (silent underflow previously wasted epochs)
  * class weights + per-class copies for the starved classes
  * checkpoint saved every epoch, optimiser state saved once (Drive quota)
  * evaluation reports macro F1 and per-class recall, never weighted F1 alone
  * resumable: --resume continues from last.pt
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from . import config as cfgmod
from . import metrics
from .labels import names_for


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def pick_device(pref: str = "auto") -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "cuda":
        print("CUDA requested but unavailable -> falling back to CPU")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class WaveDataset(Dataset):
    """(N, T) float32 waveforms + int labels. Fast, all in RAM (~180 MB)."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        return {"input_values": torch.from_numpy(self.x[i]),
                "labels": int(self.y[i])}


def class_weights(y: np.ndarray, n_classes: int, mode: str = "balanced") -> torch.Tensor:
    """Inverse-frequency weights. mode: 'balanced' | 'sqrt' | 'none'."""
    if mode == "none":
        return torch.ones(n_classes)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    if mode == "sqrt":
        w = np.sqrt(w)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def make_loader(x, y, batch_size: int, shuffle: bool, weights=None,
                num_workers: int = 0, seed: int = 0) -> DataLoader:
    ds = WaveDataset(x, y)
    sampler = None
    if weights is not None and shuffle:
        per_clip = np.asarray(weights)[y]
        sampler = WeightedRandomSampler(per_clip, num_samples=len(y),
                                        replacement=True)
        shuffle = False
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      sampler=sampler, num_workers=num_workers,
                      drop_last=False, generator=gen)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(model_name: str, classes: list[str], device: torch.device,
                freeze_feature_extractor: bool = True):
    from transformers import (AutoFeatureExtractor,
                              AutoModelForAudioClassification)

    id2label = {i: c for i, c in enumerate(classes)}
    label2id = {c: i for i, c in enumerate(classes)}
    model = AutoModelForAudioClassification.from_pretrained(
        model_name,
        num_labels=len(classes),
        id2label=id2label,
        label2id=label2id,
    )
    if freeze_feature_extractor and hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()      # standard for small SER datasets
    fe = AutoFeatureExtractor.from_pretrained(model_name)
    return model.to(device), fe


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    for batch in loader:
        x = batch["input_values"].to(device)
        logits = model(input_values=x).logits
        preds.extend(logits.argmax(-1).cpu().tolist())
        trues.extend(batch["labels"].tolist())
    return np.asarray(trues), np.asarray(preds)


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train(cfg: dict) -> dict:
    seed = int(cfgmod.get(cfg, "run.seed", 42))
    seed_everything(seed)
    device = pick_device(cfgmod.get(cfg, "run.device", "auto"))
    out_root = Path(cfgmod.get(cfg, "run.output_dir", "runs/local"))
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    classes = names_for(int(cfgmod.get(cfg, "model.n_classes", 8)))
    window_s = float(cfgmod.get(cfg, "data.window_s", 4.0))
    model_name = cfgmod.get(cfg, "model.name", "facebook/wav2vec2-base")

    print(f"device={device}  classes={len(classes)}  window={window_s}s")
    if device.type == "cpu":
        print("NOTE: training on CPU. Use configs/ci_smoke.yaml for a quick "
              "smoke test, or run training in Colab (see process.txt).")

    # ---- data (load -> split -> augment -> cache, in ser.prepare) -------- #
    from .prepare import prepare

    prep = prepare(cfg, verbose=True)
    classes = prep.classes
    window_s = prep.window_s
    x_tr, y_tr = prep.x_train, prep.y_train
    x_va, y_va = prep.x_val, prep.y_val
    x_te, y_te = prep.x_test, prep.y_test
    actor_split = prep.actor_split

    # ---- model / optimiser ---------------------------------------------- #
    model, feature_extractor = build_model(
        model_name, classes, device,
        freeze_feature_extractor=bool(cfgmod.get(
            cfg, "model.freeze_feature_extractor", True)))
    feature_extractor.save_pretrained(out_root / "preprocessor")
    (out_root / "labels.json").write_text(json.dumps({
        "classes": classes, "label2id": {c: i for i, c in enumerate(classes)},
        "window_s": window_s, "sample_rate": 16000,
        "model_name": model_name}, indent=2))

    epochs = int(cfgmod.get(cfg, "train.epochs", 3))
    bs = int(cfgmod.get(cfg, "train.batch_size", 4))
    accum = max(1, int(cfgmod.get(cfg, "train.grad_accum", 2)))
    lr = float(cfgmod.get(cfg, "train.lr", 3e-5))
    wd = float(cfgmod.get(cfg, "train.weight_decay", 0.01))
    warmup_ratio = float(cfgmod.get(cfg, "train.warmup_ratio", 0.1))
    wmode = cfgmod.get(cfg, "train.class_weight_mode", "sqrt")
    amp = bool(cfgmod.get(cfg, "train.fp16", True)) and device.type == "cuda"
    clip_norm = float(cfgmod.get(cfg, "train.max_grad_norm", 1.0))

    weights = class_weights(y_tr, len(classes), wmode)
    print(f"class weights ({wmode}): "
          f"{ {c: round(float(weights[i]), 3) for i, c in enumerate(classes)} }")

    train_loader = make_loader(
        x_tr, y_tr, bs, True,
        weights=(weights.numpy() if wmode != "none" else None),
        num_workers=int(cfgmod.get(cfg, "run.num_workers", 0)), seed=seed)
    val_loader = make_loader(x_va, y_va, bs, False)
    test_loader = make_loader(x_te, y_te, bs, False)

    steps_per_epoch = max(1, math.ceil(len(train_loader) / accum))
    total_steps = steps_per_epoch * epochs
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    from transformers import get_cosine_schedule_with_warmup

    sched = get_cosine_schedule_with_warmup(
        optim, int(total_steps * warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if hasattr(torch, "amp") \
        else torch.cuda.amp.GradScaler(enabled=amp)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device)) if wmode != "none" \
        else nn.CrossEntropyLoss()

    start_epoch = 0
    best = {"macro_f1": -1.0, "epoch": -1}
    resume_path = ckpt_dir / "last.pt"
    if cfgmod.get(cfg, "train.resume", False) and resume_path.is_file():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optim"])
        sched.load_state_dict(state["sched"])
        start_epoch = state["epoch"] + 1
        best = state.get("best", best)
        print(f"resumed from {resume_path} at epoch {start_epoch}")

    history = []
    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        running, seen, step = 0.0, 0, 0
        optim.zero_grad(set_to_none=True)
        for batch in train_loader:
            xb = batch["input_values"].to(device, non_blocking=True)
            yb = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type,
                                dtype=torch.float16, enabled=amp):
                logits = model(input_values=xb).logits
                loss = loss_fn(logits, yb) / accum
            scaler.scale(loss).backward()
            running += float(loss.item()) * accum
            seen += 1
            step += 1
            if step % accum == 0:
                scaler.unscale_(optim)
                gn = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_norm))
                scaler.step(optim)
                scaler.update()
                sched.step()
                optim.zero_grad(set_to_none=True)
                if seen % 20 == 0:
                    print(f"  epoch {epoch + 1} step {seen}/{len(train_loader)} "
                          f"loss {running / seen:.4f} grad {gn:.2f} "
                          f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
        if step % accum:                                    # flush tail
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optim)
            scaler.update()
            sched.step()
            optim.zero_grad(set_to_none=True)

        yt, yp = predict(model, val_loader, device)
        res = metrics.summary(yt, yp, classes)
        rec = {
            "epoch": epoch + 1,
            "train_loss": round(running / max(seen, 1), 4),
            "val_accuracy": res["accuracy"],
            "val_macro_f1": res["macro_f1"],
            "val_weighted_f1": res["weighted_f1"],
            "seconds": round(time.time() - t0, 1),
        }
        history.append(rec)
        print(f"epoch {epoch + 1}/{epochs}: loss={rec['train_loss']} "
              f"val_acc={res['accuracy']:.4f} val_macroF1={res['macro_f1']:.4f} "
              f"val_wF1={res['weighted_f1']:.4f} ({rec['seconds']}s)")

        model.save_pretrained(ckpt_dir / f"epoch_{epoch + 1}")
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optim": optim.state_dict(), "sched": sched.state_dict(),
                    "best": best}, resume_path)

        if res["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": res["macro_f1"], "epoch": epoch + 1,
                    "val_accuracy": res["accuracy"],
                    "val_weighted_f1": res["weighted_f1"]}
            model.save_pretrained(out_root / "best")
            feature_extractor.save_pretrained(out_root / "best")
            print(f"  new best (macro F1 {res['macro_f1']:.4f}) -> "
                  f"{out_root / 'best'}")

    # ---- test with the selected checkpoint ------------------------------- #
    from transformers import AutoModelForAudioClassification

    best_dir = out_root / "best"
    if best_dir.is_dir():
        model = AutoModelForAudioClassification.from_pretrained(
            best_dir).to(device)
        print(f"evaluating best checkpoint (epoch {best['epoch']}) on test")
    yt, yp = predict(model, val_loader, device)
    val_res = metrics.summary(yt, yp, classes)
    yt, yp = predict(model, test_loader, device)
    test_res = metrics.summary(yt, yp, classes)

    print("\n--- VALIDATION (used for checkpoint selection) ---")
    print(metrics.format_report(val_res, classes))
    print("\n--- TEST (held-out actors, reported once) ---")
    print(metrics.format_report(test_res, classes))

    result = {
        "model_name": model_name,
        "classes": classes,
        "window_s": window_s,
        "seed": seed,
        "device": str(device),
        "split": actor_split,
        "split_info": getattr(prep, "split_info", None),
        "best_epoch": best["epoch"],
        "history": history,
        "val": val_res,
        "test": test_res,
        "augmentation": {"copies": getattr(prep, "aug_copies", None),
                         "preset": cfgmod.get(cfg, "augment.preset", "full"),
                         "train_clips": int(len(y_tr)),
                         "synthetic_fraction": round(prep.synthetic_fraction, 4)},
        "majority_baseline": metrics.majority_baseline(np.concatenate([y_va, y_te])),
    }
    (out_root / "metrics.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_root / 'metrics.json'}")

    try:                                     # plots are nice, not required
        from .plots import save_all

        save_all(out_root, result, yt, yp, classes)
    except Exception as exc:                             # noqa: BLE001
        print(f"(plots skipped: {type(exc).__name__}: {exc})")
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Train speech emotion classifier")
    ap.add_argument("--config", default=None, help="path to a YAML config")
    ap.add_argument("--set", nargs="*", default=[],
                    help="overrides, e.g. --set train.epochs=1 data.limit=12")
    args = ap.parse_args(argv)

    cfg = cfgmod.resolve(args.config)
    for item in args.set:
        key, _, val = item.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = cfgmod._coerce(val)               # noqa: SLF001
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

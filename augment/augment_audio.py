#!/usr/bin/env python3
"""Offline audio augmentation for speech emotion recognition.

Purpose: 3x the training set on disk, with emotion-preserving transforms, so
that a wav2vec2 fine-tune sees more speakers/conditions than the ~110 actors
in CREMA-D + RAVDESS. Everything here is CPU-only, free, and needs
numpy + soundfile (librosa optional).

Two usage modes
---------------
1) Offline (default). Writes copied WAVs next to a manifest, so the very same
   npz preprocessing step you already have can consume them:

     # minority classes only, 6 copies per original
     python augment/augment_audio.py --splits data/splits.json \
            --out data/augmented --copies 6 --preset minority

     # everything, 2 copies, mild preset
     python augment/augment_audio.py --splits data/splits.json \
            --out data/augmented --copies 2 --preset light

2) As a library (on-the-fly, no extra disk):

     from augment.audio_ops import apply_chain
     y2 = apply_chain(y, sr=16000, rng=rng, chance=0.8)

Presets
-------
  light     gain + mild noise                (safe, small gain)
  phone     band-limit 300-3400 Hz + noise   (mimics the demo's laptop mic)
  speed     +-5% speed perturbation          (best single transform)
  reverb    synthetic room impulse           (train/test room mismatch)
  minority  speed + noise + gain + shift     (targets calm/surprised)
  full      all of the above, randomly sampled per copy

Rule of thumb: augment TRAIN only. Never augment val/test, or your metric
stops measuring generalisation and starts measuring your augmentation.

Output layout
-------------
  <out>/wav/<dataset>/<emotion>/<orig_stem>__augNN.wav
  <out>/train.csv      same columns as make_splits.py output + `source`
  <out>/aug_manifest.csv  one row per generated file: what was applied
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio_ops as ops                                       # noqa: E402
import labels as L                                            # noqa: E402


# --------------------------------------------------------------------------- #
# Transform catalogue
# --------------------------------------------------------------------------- #
def _tx_gain(y, rng, sr):
    return ops.rms_normalize(y, target_dbfs=float(rng.uniform(-28, -16)))


def _tx_noise(y, rng, sr):
    snr = float(rng.uniform(12, 30))          # keep it audible, not destroyed
    return ops.add_noise(y, snr, rng, kind="pink")


def _tx_white_noise(y, rng, sr):
    return ops.add_noise(y, float(rng.uniform(15, 35)), rng, kind="white")


def _tx_speed(y, rng, sr):
    return ops.speed(y, float(rng.uniform(0.92, 1.08)), sr)


def _tx_pitch(y, rng, sr):
    return ops.pitch_shift(y, float(rng.uniform(-1.5, 1.5)), sr)


def _tx_reverb(y, rng, sr):
    return ops.reverb(y, float(rng.uniform(0.15, 0.45)), rng,
                      wet=float(rng.uniform(0.15, 0.35)), sr=sr)


def _tx_bandlimit(y, rng, sr):
    lo = float(rng.uniform(80, 200))
    hi = float(rng.uniform(3400, 7000))
    return ops.highpass(ops.lowpass(y, hi, sr), lo, sr)


def _tx_phone(y, rng, sr):
    """Telephone band 300-3400 Hz: closest analogue to a cheap laptop mic."""
    return ops.highpass(ops.lowpass(y, 3400, sr), 300, sr)


def _tx_shift(y, rng, sr):
    return ops.time_shift(y, float(rng.uniform(20, 150)), rng, sr)


def _tx_mask(y, rng, sr):
    return ops.time_mask_wave(y, n_masks=int(rng.integers(1, 3)),
                              max_ms=float(rng.uniform(40, 120)), rng=rng, sr=sr)


TRANSFORMS = {
    "gain": (_tx_gain, 0.6),
    "noise": (_tx_noise, 0.5),
    "white_noise": (_tx_white_noise, 0.15),
    "speed": (_tx_speed, 0.45),
    "pitch": (_tx_pitch, 0.15),
    "reverb": (_tx_reverb, 0.3),
    "bandlimit": (_tx_bandlimit, 0.3),
    "phone": (_tx_phone, 0.25),
    "shift": (_tx_shift, 0.35),
    "mask": (_tx_mask, 0.2),
}

PRESETS: dict[str, list[str]] = {
    "light": ["gain", "noise"],
    "phone": ["phone", "gain", "noise"],
    "speed": ["speed", "gain"],
    "reverb": ["reverb", "gain", "noise"],
    "minority": ["speed", "noise", "gain", "shift", "pitch"],
    "full": ["gain", "noise", "speed", "pitch", "reverb", "bandlimit",
             "shift", "mask"],
}

# Classes that only one corpus provides, and that the notebook's model failed
# to learn (calm/surprised had 152 of 7,109 clips). They get most of the budget.
RARE_BOOST = {"calm": 2.5, "surprised": 2.5, "neutral": 1.3}


# --------------------------------------------------------------------------- #
# Chain
# --------------------------------------------------------------------------- #
def apply_chain(y: np.ndarray, sr: int = ops.TARGET_SR,
                rng: np.random.Generator | None = None,
                preset: str = "full", chance: float = 0.8,
                seed: int | None = None) -> tuple[np.ndarray, list[str]]:
    """Apply a random subset of a preset. Returns (audio, applied_names)."""
    rng = rng or np.random.default_rng(seed)
    names = PRESETS.get(preset, PRESETS["full"])
    applied: list[str] = []
    out = np.asarray(y, dtype=np.float32)
    for name in names:
        fn, p = TRANSFORMS[name]
        if rng.random() > (p * chance):
            continue
        try:
            cand = fn(out, rng, sr)
            if cand is not None and len(cand) > 0 and np.isfinite(cand).all():
                out = cand
                applied.append(name)
        except Exception:                                  # noqa: BLE001
            continue                       # a transform must never kill the run
    # A transform may have changed level/length; restore both.
    out = ops.rms_normalize(out, target_dbfs=-23.0)
    out = ops.peak_normalize(out, peak=0.95)
    return out.astype(np.float32), applied


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def load_split(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data["splits"]["train"] if "splits" in data else data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", required=True,
                    help="splits.json from make_splits.py (only 'train' is used)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--copies", type=int, default=3,
                    help="copies per original clip (default 3)")
    ap.add_argument("--preset", default="full", choices=sorted(PRESETS))
    ap.add_argument("--chance", type=float, default=0.8,
                    help="probability multiplier for each transform")
    ap.add_argument("--balance", action="store_true",
                    help="size the copy count per class (rare classes get more)")
    ap.add_argument("--target-max", type=int, default=0,
                    help="with --balance: cap every class at this many clips")
    ap.add_argument("--max-copies", type=int, default=20,
                    help="never make more than N copies of one clip")
    ap.add_argument("--window", type=float, default=None,
                    help="trim/pad output to N seconds (e.g. 3 for live demo)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N originals")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = load_split(Path(args.splits))
    if not rows:
        print("no training rows in splits file", file=sys.stderr)
        return 1
    if args.limit:
        rows = rows[: args.limit]

    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["emotion"]].append(r)

    # How many copies per class?
    plan: dict[str, int] = {}
    if args.balance:
        sizes = {c: len(v) for c, v in by_class.items()}
        target = args.target_max or max(sizes.values())
        for c, n in sizes.items():
            # ceil, so a class just under target still gets a top-up copy
            copies = max(0, int(np.ceil(target / n)) - 1)
            plan[c] = min(copies, args.max_copies)
        print(f"balanced plan (target {target}/class, max {args.max_copies} "
              f"copies): { {c: p for c, p in sorted(plan.items())} }")
    else:
        for c in by_class:
            copies = int(round(args.copies * RARE_BOOST.get(c, 1.0)))
            plan[c] = min(copies, args.max_copies)

    out = Path(args.out)
    wav_dir = out / "wav"
    manifest_path = out / "aug_manifest.csv"
    train_csv = out / "train.csv"

    total = sum(max(0, plan[c]) * len(v) for c, v in by_class.items())
    print(f"originals: {len(rows)}  ->  to generate: {total:,} augmented clips")
    print(f"preset: {args.preset}  window: {args.window or 'source length'}")
    if args.dry_run:
        for c, v in sorted(by_class.items()):
            print(f"  {c:10s} originals={len(v):5d}  copies_each={plan[c]}")
        return 0

    wav_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    applied_counter: Counter = Counter()
    new_rows: list[dict] = []

    with open(manifest_path, "w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        writer.writerow(["aug_path", "source_path", "dataset", "emotion",
                         "speaker", "copy", "transforms"])
        for emotion, items in sorted(by_class.items()):
            for item in items:
                src = Path(item["path"])
                try:
                    y = ops.load_audio(src)
                except Exception as exc:                   # noqa: BLE001
                    print(f"  skip {src.name}: {type(exc).__name__}", file=sys.stderr)
                    continue
                if args.window:
                    y = ops.fit_window(y, args.window, rng=None)
                for k in range(plan[emotion]):
                    aug, applied = apply_chain(
                        y, ops.TARGET_SR, rng=rng, preset=args.preset,
                        chance=args.chance)
                    if args.window:
                        aug = ops.fit_window(aug, args.window, rng=None)
                    dst = (wav_dir / item["dataset"] / emotion /
                           f"{src.stem}__aug{k:02d}.wav")
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    ops.save_audio(dst, aug)
                    writer.writerow([dst, src, item["dataset"], emotion,
                                     item["speaker"], k, "+".join(applied)])
                    for name in applied:
                        applied_counter[name] += 1
                    new_rows.append({**{kk: item[kk] for kk in
                                        ("dataset", "emotion", "speaker")},
                                     "path": str(dst), "source": str(src)})
                    made += 1
                    if made % 500 == 0:
                        print(f"  generated {made:,}/{total:,}", flush=True)

    # train.csv = originals + augmentations, so training can use one manifest.
    with open(train_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "dataset", "emotion", "speaker", "source", "is_augmented"])
        for r in rows:
            w.writerow([r["path"], r["dataset"], r["emotion"], r["speaker"], "", 0])
        for r in new_rows:
            w.writerow([r["path"], r["dataset"], r["emotion"], r["speaker"],
                        r["source"], 1])

    print(f"\ngenerated {made:,} clips")
    print(f"  wavs      : {wav_dir}")
    print(f"  manifest  : {manifest_path}")
    print(f"  train.csv : {train_csv}  ({len(rows) + len(new_rows):,} rows)")
    print(f"  transforms used: {dict(applied_counter.most_common())}")
    print("\nNEXT: point your npz preprocessing at train.csv, and keep using "
          "the ORIGINAL val/test splits.\n"
          "Never augment validation or test data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

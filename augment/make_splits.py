#!/usr/bin/env python3
"""Build speaker-disjoint train/val/test splits from the three corpora.

Why this file exists
-------------------
The training notebook used CREMA-D + RAVDESS for train/val and TESS as test.
TESS is then *also* the set the best checkpoint was chosen on, which makes the
reported score optimistic. It also guarantees that `calm` (RAVDESS only) and
`surprised` (ps files) can never be evaluated properly.

This module gives you two honest options:

  --profile 3way   each corpus is split by speaker into train/val/test.
                   TESS speakers (OAF/YAF) go to train -> TESS clips appear in
                   all three splits, so all 8 classes are evaluated. Split is
                   speaker-disjoint *within* each corpus.
  --profile holdout  CREMA-D + RAVDESS -> train/val (by speaker).
                   TESS -> test (untouched). Use when you need an external
                   number, and accept that `calm` is not in the test set.

Splitting by speaker, never by file, is the whole point: otherwise the same
actor's voice is in train and test and accuracy is inflated by voice identity.

Usage
-----
  python augment/make_splits.py --scan ~/Downloads --profile 3way --out data
  python augment/make_splits.py --scan ~/Downloads --profile holdout --out data
Writes splits.json + train.csv/val.csv/test.csv into --out.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_datasets as audit                                # noqa: E402
import labels as L                                            # noqa: E402

TESS_AS_TEST = "holdout"        # profile name that keeps TESS pure test


# --------------------------------------------------------------------------- #
# Core split logic (importable: audit_datasets.py calls build_split)
# --------------------------------------------------------------------------- #
def _split_speakers(speakers: list[str], val_frac: float, test_frac: float,
                    seed: int) -> tuple[set[str], set[str], set[str]]:
    spk = sorted(set(speakers))
    rng = random.Random(seed)
    rng.shuffle(spk)
    n = len(spk)
    if n == 1:
        # Cannot hold anything out without losing the corpus entirely.
        return set(spk), set(), set()
    if n == 2:
        # One speaker stays for training; the other is held out for evaluation.
        return set(spk[:1]), set(), set(spk[1:])
    n_test = max(1, int(round(n * test_frac))) if test_frac > 0 else 0
    n_val = max(1, int(round(n * val_frac)))
    # never leave the training set empty
    while n - n_test - n_val < 1:
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break
    test = set(spk[:n_test])
    val = set(spk[n_test:n_test + n_val])
    train = set(spk[n_test + n_val:])
    return train, val, test


def build_split(rows: list[dict], val_frac: float = 0.20, test_frac: float = 0.10,
                seed: int = 42, profile: str = "3way") -> dict[str, list[dict]]:
    """rows: [{path, dataset, emotion, speaker}] -> {'train'|'val'|'test': rows}."""
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)

    split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for ds, items in by_ds.items():
        if profile == TESS_AS_TEST and ds == "TESS":
            split["test"].extend(items)
            continue
        train_s, val_s, test_s = _split_speakers(
            [r["speaker"] for r in items], val_frac, test_frac, seed)
        for r in items:
            if r["speaker"] in train_s:
                split["train"].append(r)
            elif r["speaker"] in val_s:
                split["val"].append(r)
            elif r["speaker"] in test_s:
                split["test"].append(r)
            else:                                   # safety net
                split["train"].append(r)
    return split


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def describe(split: dict[str, list[dict]], classes: list[str]) -> None:
    print(f"  {'split':6s} {'clips':>7s} {'speakers':>9s}  per-class")
    for name in ("train", "val", "test"):
        rows = split[name]
        spk = {r["speaker"] for r in rows}
        counter = Counter(r["emotion"] for r in rows)
        per = " ".join(f"{c[:4]}:{counter.get(c, 0)}" for c in classes)
        print(f"  {name:6s} {len(rows):>7,} {len(spk):>9,}  {per}")

    sets = {n: {r["speaker"] for r in split[n]} for n in split}
    overlaps = {
        "train^val": sets["train"] & sets["val"],
        "train^test": sets["train"] & sets["test"],
        "val^test": sets["val"] & sets["test"],
    }
    bad = {k: sorted(v)[:5] for k, v in overlaps.items() if v}
    print(f"\n  speaker overlap between splits: "
          f"{'NONE (good)' if not bad else bad}")
    if bad:
        print("  -> same voice in two splits inflates accuracy; "
              "use --profile holdout or fewer test speakers")

    for name in ("train", "val", "test"):
        counter = Counter(r["emotion"] for r in split[name])
        gaps = [c for c in classes if counter.get(c, 0) == 0]
        if gaps:
            print(f"  ! {name}: no examples of {gaps}"
                  + ("  (cannot be evaluated there)" if name == "test" else ""))
    # A class absent from TRAIN is a training defect; absent from VAL only means
    # model selection is blind for it. Say which is which.
    tr_counter = Counter(r["emotion"] for r in split["train"])
    dead = [c for c in classes if tr_counter.get(c, 0) == 0]
    if dead:
        print(f"  !! training set cannot learn {dead}: no clips at all")
    ratio = ""
    tr = tr_counter
    if tr:
        vals = [v for v in tr.values() if v]
        if vals:
            ratio = f"  train imbalance {max(vals)/min(vals):.1f}x"
    print(ratio)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--profile", choices=["3way", "holdout"], default="3way")
    ap.add_argument("--val-frac", type=float, default=0.20)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--merge-calm", action="store_true",
                    help="fold calm into neutral (7 classes)")
    ap.add_argument("--no-decode", action="store_true")
    args = ap.parse_args()

    root = Path(args.scan).expanduser().resolve()
    found = audit.scan(root)
    rows: list[dict] = []
    for ds in ("CREMA-D", "RAVDESS", "TESS"):
        for path in found.get(ds, []):
            try:
                emotion, speaker = L.parse(ds, path)
            except L.ParseError:
                continue
            rows.append({"path": str(path), "dataset": ds,
                         "emotion": emotion, "speaker": speaker})

    if not rows:
        print("No files matched the dataset grammars. Scan the folder that "
              "contains the extracted corpora.", file=sys.stderr)
        return 1

    if not args.no_decode:
        alive = []
        for r in rows:
            try:
                import soundfile as sf

                info = sf.info(r["path"])
                if info.frames > 0:
                    alive.append(r)
            except Exception:                              # noqa: BLE001
                pass
        dropped = len(rows) - len(alive)
        print(f"dropped {dropped}/{len(rows)} undecodable files")
        if dropped > 0.01 * len(rows):
            print("WARNING: >1% of the corpus is unreadable. Check for Git-LFS "
                  "pointers before training.", file=sys.stderr)
        rows = alive

    if args.merge_calm:
        for r in rows:
            r["emotion"] = L.to_7class(r["emotion"])

    classes = L.CLASSES_7 if args.merge_calm else L.CLASSES
    split = build_split(rows, args.val_frac, args.test_frac, args.seed, args.profile)

    print(f"\nprofile={args.profile}  seed={args.seed}  "
          f"classes={len(classes)}  merge_calm={args.merge_calm}")
    describe(split, classes)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "splits.json").write_text(json.dumps({
        "profile": args.profile, "seed": args.seed,
        "classes": classes, "merge_calm": args.merge_calm,
        "splits": split,
    }, indent=2))
    for name in ("train", "val", "test"):
        with open(out / f"{name}.csv", "w", encoding="utf-8") as fh:
            fh.write("path,dataset,emotion,speaker,label\n")
            for r in split[name]:
                fh.write(f"{r['path']},{r['dataset']},{r['emotion']},"
                         f"{r['speaker']},{classes.index(r['emotion'])}\n")
    print(f"\nwrote {out/'splits.json'} and {out}/{{train,val,test}}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

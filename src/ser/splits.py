"""Speaker-disjoint splitting of the 1,440 clips into train/val/test.

The dataset ships ONE split ("train"). Splitting it yourself is not optional:
if you shuffle rows instead of actors, the same voice appears in train and
test and accuracy measures voice identity rather than emotion.

Default: 24 actors -> 16 train / 4 val / 4 test.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np


def speaker_disjoint_split(actors: list[int], val_frac: float = 0.17,
                           test_frac: float = 0.17,
                           seed: int = 42) -> dict[str, list[int]]:
    """Return {'train': [...], 'val': [...], 'test': [...]} of actor ids."""
    uniq = sorted(set(int(a) for a in actors))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    if n < 3:
        return {"train": uniq, "val": [], "test": []}
    n_val = max(1, int(round(n * val_frac)))
    n_test = max(1, int(round(n * test_frac)))
    while n - n_val - n_test < 1:
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
    test = uniq[:n_test]
    val = uniq[n_test:n_test + n_val]
    train = uniq[n_test + n_val:]
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def indices_for(actors: np.ndarray, chosen: list[int]) -> np.ndarray:
    mask = np.isin(actors, np.asarray(chosen))
    return np.flatnonzero(mask)


def describe(actor_split: dict[str, list[int]], actors: np.ndarray,
             labels: np.ndarray, classes: list[str]) -> dict:
    """Per-split clip counts, speakers, per-class counts, and sanity checks."""
    info: dict = {"splits": {}, "problems": []}
    sets = {name: set(v) for name, v in actor_split.items()}
    for name, chosen in actor_split.items():
        idx = indices_for(actors, chosen)
        counts = Counter(classes[i] for i in labels[idx])
        info["splits"][name] = {
            "clips": int(len(idx)),
            "actors": sorted(chosen),
            "per_class": {c: int(counts.get(c, 0)) for c in classes},
        }
        missing = [c for c in classes if counts.get(c, 0) == 0]
        if missing and name == "test":
            info["problems"].append(
                f"test split has no examples of {missing}: those classes "
                f"cannot be measured")
        if missing and name == "train":
            info["problems"].append(
                f"train split has no examples of {missing}: they cannot be learned")
        if missing and name == "val":
            info["problems"].append(
                f"val split has no examples of {missing}: model selection is "
                f"blind to them")
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b and sets[a] & sets[b]:
                info["problems"].append(
                    f"actor(s) {sorted(sets[a] & sets[b])} appear in both "
                    f"{a} and {b} -> speaker leakage")
    train_counts = info["splits"]["train"]["per_class"]
    vals = [v for v in train_counts.values() if v]
    if vals:
        info["train_imbalance"] = round(max(vals) / min(vals), 2)
    return info


def save(actor_split: dict, path: str | Path, extra: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"kind": "speaker_disjoint", "splits": actor_split}
    payload.update(extra or {})
    path.write_text(json.dumps(payload, indent=2))
    return path


def load(path: str | Path) -> dict[str, list[int]]:
    return json.loads(Path(path).read_text())["splits"]

#!/usr/bin/env python3
"""Verify xbgoose/ravdess still matches the assumptions this repo relies on.

The whole pipeline assumes: 1,440 rows, one split called "train", 24 actors,
8 emotion classes with clean string labels, audio-only speech. If the dataset
is updated (or a mirror is used), those assumptions can silently break.

Runs in CI without downloading audio: it uses the HF datasets-server metadata
endpoint, so it takes ~2 seconds.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ser import labels                                     # noqa: E402

REPO = "xbgoose/ravdess"
STATS_URL = ("https://datasets-server.huggingface.co/statistics"
             f"?dataset={REPO.replace('/', '%2F')}&config=default&split=train")
SPLITS_URL = (f"https://datasets-server.huggingface.co/splits"
              f"?dataset={REPO.replace('/', '%2F')}")


def get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "ser-contract-check"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def main() -> int:
    problems: list[str] = []
    try:
        splits = get(SPLITS_URL)
        names = [s["split"] for s in splits.get("splits", [])]
        print(f"splits exposed by {REPO}: {names}")
        if names != ["train"]:
            problems.append(
                f"expected a single 'train' split, found {names}. "
                f"If the dataset was re-uploaded with real splits, update "
                f"ser.data.load_ravdess_hf and this contract.")
        stats = get(STATS_URL)
    except Exception as exc:                                # noqa: BLE001
        print(f"WARNING: could not reach the datasets server ({exc}).")
        print("Skipping the live contract check.")
        return 0

    cols = {c["column_name"]: c for c in stats["statistics"]}
    n = stats["num_examples"]
    print(f"rows: {n}")

    for required in ("audio", "emotion", "actor", "gender"):
        if required not in cols:
            problems.append(f"missing required column {required!r}")

    if n != labels.EXPECTED_ROWS:
        problems.append(f"row count {n} != expected {labels.EXPECTED_ROWS}")

    if "emotion" in cols:
        freqs = cols["emotion"]["column_statistics"]["frequencies"]
        print(f"emotion counts: {freqs}")
        if set(freqs) != set(labels.CLASSES):
            problems.append(f"emotion labels changed: {sorted(freqs)} "
                            f"!= {sorted(labels.CLASSES)}")
        for cls, expected in labels.EXPECTED_COUNTS.items():
            if freqs.get(cls, 0) != expected:
                problems.append(f"{cls}: {freqs.get(cls, 0)} clips, "
                                f"expected {expected}")

    if "actor" in cols:
        actor_stats = cols["actor"]["column_statistics"]
        n_actors = int(actor_stats["max"] - actor_stats["min"] + 1)
        print(f"actors: {n_actors}")
        if n_actors != labels.EXPECTED_ACTORS:
            problems.append(f"{n_actors} actors != expected "
                            f"{labels.EXPECTED_ACTORS} (split sizes change!)")

    for col, expected in (("modality", "audio-only"), ("vocal_channel", "speech")):
        if col in cols:
            freqs = cols[col]["column_statistics"]["frequencies"]
            if set(freqs) != {expected}:
                problems.append(f"{col} is {sorted(freqs)}, expected only "
                                f"{expected!r}")

    if "audio" in cols:
        a = cols["audio"]["column_statistics"]
        print(f"clip duration: {a['min']:.2f}-{a['max']:.2f}s")
        lo, hi = labels.DURATION_RANGE_S
        if a["min"] < lo - 0.5 or a["max"] > hi + 1.0:
            problems.append("clip durations moved outside the documented range; "
                            "re-check data.window_s")

    if problems:
        print("\nDATASET CONTRACT VIOLATIONS:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nDATASET CONTRACT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

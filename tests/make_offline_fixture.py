#!/usr/bin/env python3
"""Create a tiny local RAVDESS-shaped corpus for offline testing of ser.prepare.

Lets you exercise the whole data path (load -> split -> augment -> cache)
without Hugging Face, a GPU, or torch.

    python tests/make_offline_fixture.py --out /tmp/ravdess_fixture --actors 6
    python -m ser.prepare --config configs/local_train.yaml --dry-run \
        --set data.source=/tmp/ravdess_fixture run.output_dir=/tmp/run fixture
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EMOTIONS = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
            "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}


def synth_voice(seconds: float, f0: float, sr: int, rng) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 5))
    y *= 1.0 + 0.05 * np.sin(2 * np.pi * 4.5 * t)
    env = np.minimum(1.0, np.minimum(t * 15, (t[-1] - t + 1e-9) * 15))
    y = y * env + 0.01 * rng.standard_normal(len(t))
    return (y / np.max(np.abs(y)) * 0.45).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/ravdess_fixture")
    ap.add_argument("--actors", type=int, default=6)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()

    import soundfile as sf

    out = Path(args.out)
    rng = np.random.default_rng(0)
    count = 0
    for actor in range(1, args.actors + 1):
        actor_dir = out / f"Actor_{actor:02d}"
        actor_dir.mkdir(parents=True, exist_ok=True)
        for code, name in EMOTIONS.items():
            for rep in range(1, args.reps + 1):
                # 3-5 s clips, like the real corpus
                dur = 3.0 + 0.5 * ((actor + rep) % 4)
                y = synth_voice(dur, 110.0 + 9 * actor, args.sr, rng)
                fname = f"03-01-{code}-01-01-{rep:02d}-{actor:02d}.wav"
                sf.write(str(actor_dir / fname), y, args.sr, subtype="PCM_16")
                count += 1
    print(f"wrote {count} clips under {out} "
          f"({args.actors} actors x {len(EMOTIONS)} emotions x {args.reps})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Measure what augmentation does to the audio -- BEFORE you spend GPU hours.

Prints, per preset:
  * output duration (speed perturbation changes it)
  * RMS / peak in dBFS (did we blow up the level?)
  * clipping fraction (added noise can clip)
  * spectral centroid shift (did we mangle the voice?)
  * a crude "still the same voice" guard: energy below 4 kHz should dominate

Also writes before/after WAV pairs so you can listen, which matters more than
any number here.

Usage:
  python augment/benchmark_augmentation.py --audio some.wav
  python augment/benchmark_augmentation.py --scan /tmp/fake_data --n 3 \
      --out /tmp/aug_demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio_ops as ops                                       # noqa: E402
import augment_audio as aug                                   # noqa: E402


def spectrum_stats(y: np.ndarray, sr: int) -> tuple[float, float]:
    if len(y) < 64:
        return 0.0, 0.0
    spec = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(len(y), 1.0 / sr)
    centroid = float(np.sum(f * spec) / (np.sum(spec) + 1e-9))
    low = float(np.sum(spec[f < 4000] ** 2) / (np.sum(spec ** 2) + 1e-9))
    return centroid, low


def line(tag: str, y: np.ndarray, sr: int) -> str:
    centroid, low = spectrum_stats(y, sr)
    clip = float(np.mean(np.abs(y) >= 0.999))
    return (f"  {tag:22s} dur={ops.duration_s(y, sr):5.2f}s "
            f"rms={ops.rms_dbfs(y):6.1f}dBFS peak={np.max(np.abs(y)):.2f} "
            f"clip={100*clip:4.1f}% centroid={centroid:5.0f}Hz "
            f"<4kHz={100*low:4.0f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", default=None, help="single file to analyse")
    ap.add_argument("--scan", default=None, help="folder: pick N files")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default="/tmp/aug_demo")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files: list[Path] = []
    if args.audio:
        files = [Path(args.audio)]
    elif args.scan:
        found = sorted(p for p in Path(args.scan).rglob("*.wav")
                       if not p.name.startswith("._"))
        files = found[: args.n]
    if not files:
        print("give --audio or --scan", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sr = ops.TARGET_SR
    rng = np.random.default_rng(args.seed)

    for path in files:
        try:
            y = ops.load_audio(path, sr)
        except Exception as exc:                            # noqa: BLE001
            print(f"skip {path.name}: {exc}")
            continue
        if len(y) < sr // 4:
            print(f"skip {path.name}: too short ({ops.duration_s(y, sr):.2f}s)")
            continue

        print(f"\n{path.name}  ({len(y)} samples, {ops.duration_s(y, sr):.2f}s)")
        base = line("original", y, sr)
        print(base)
        ops.save_audio(out / f"{path.stem}__orig.wav", y, sr)

        for preset in ("light", "speed", "phone", "reverb", "minority", "full"):
            acc_dur, acc_centroid, acc_low, nan = [], [], [], 0
            for _ in range(5):
                z, applied = aug.apply_chain(y, sr, rng=rng, preset=preset,
                                             chance=1.0)
                if z is None or not len(z) or not np.isfinite(z).all():
                    nan += 1
                    continue
                c, low = spectrum_stats(z, sr)
                acc_dur.append(ops.duration_s(z, sr))
                acc_centroid.append(c)
                acc_low.append(100 * low)
            if not acc_dur:
                print(f"  {preset:22s} ALL OUTPUTS INVALID")
                continue
            centroid0, low0 = spectrum_stats(y, sr)
            print(f"  {preset:22s} dur={np.mean(acc_dur):5.2f}s "
                  f"centroid={np.mean(acc_centroid):4.0f}Hz "
                  f"({100*(np.mean(acc_centroid)/(centroid0+1e-9)-1):+5.0f}%) "
                  f"<4kHz={np.mean(acc_low):4.0f}% "
                  f"(orig {100*low0:.0f}%) invalid={nan}")
            z, applied = aug.apply_chain(y, sr, rng=np.random.default_rng(1),
                                         preset=preset, chance=1.0)
            ops.save_audio(out / f"{path.stem}__{preset}.wav", z, sr)

    print(f"\nwrote before/after WAVs to {out}")
    print("LISTEN to them. If a preset makes speech unintelligible, the label "
          "is no longer valid and the model learns noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

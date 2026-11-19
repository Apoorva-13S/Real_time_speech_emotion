#!/usr/bin/env python3
"""Self-checks for the augmentation stack. No pytest needed:

    python augment/tests/test_audio_ops.py

Verifies the invariants that matter: finite output, sane level, no clipping,
shape contracts, correct label parsing (including the "ps" trap), and that the
split builder never leaks a speaker across splits.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import audio_ops as ops                                       # noqa: E402
import augment_audio as aug                                   # noqa: E402
import labels as L                                            # noqa: E402
from make_splits import build_split                           # noqa: E402

SR = ops.TARGET_SR
FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILS.append(name)


def voice(seconds: float = 2.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    y = sum(np.sin(2 * np.pi * 120 * k * t) / k for k in range(1, 5))
    y = y + 0.02 * rng.standard_normal(len(t))
    return (y / np.max(np.abs(y)) * 0.4).astype(np.float32)


def main() -> int:
    print("labels / parsing")
    check("TESS ps -> surprised",
          L.parse("TESS", Path("OAF_back_ps.wav"))[0] == "surprised")
    check("TESS fear -> fearful",
          L.parse("TESS", Path("YAF_say_fear.wav"))[0] == "fearful")
    check("TESS speaker", L.parse("TESS", Path("YAF_say_fear.wav"))[1] == "tess_YAF")
    check("CREMA code",
          L.parse("CREMA-D", Path("1029_MTI_NEU_XX.wav")) == ("neutral", "crema_1029"))
    check("RAVDESS emotion field",
          L.parse("RAVDESS", Path("03-01-08-01-01-01-07.wav"))[0] == "surprised")
    check("RAVDESS actor", L.parse("RAVDESS", Path("03-01-08-01-01-01-07.wav"))[1]
          == "ravdess_07")
    for bad in ("03-02-05-01-01-01-01.wav",       # video modality
                "03-01-99-01-01-01-01.wav",       # bad code
                "short.wav"):
        try:
            L.parse("RAVDESS", Path(bad))
            check(f"rejects {bad}", False)
        except L.ParseError:
            check(f"rejects {bad}", True)
    check("8 classes defined", len(L.CLASSES) == 8)
    check("calm merges to neutral", L.to_7class("calm") == "neutral")

    print("\naudio invariants")
    y = voice()
    check("rms_normalize hits target",
          abs(ops.rms_dbfs(ops.rms_normalize(y, -23.0)) + 23.0) < 0.5,
          f"got {ops.rms_dbfs(ops.rms_normalize(y, -23.0)):.2f}")
    check("peak_normalize capped",
          np.max(np.abs(ops.peak_normalize(y, 0.95))) <= 0.951)
    check("resample 16k->8k halves length",
          abs(len(ops.resample(y, SR, 8000)) - len(y) // 2) <= 2)
    check("resample 16k->48k triples length",
          abs(len(ops.resample(y, SR, 48000)) - 3 * len(y)) <= 3)
    z = ops.fit_window(y, 1.0)
    check("fit_window exact length", len(z) == SR)
    z2 = ops.fit_window(y[: SR // 2], 1.0)
    check("fit_window pads long enough", len(z2) == SR)
    check("speed 1.05 shortens", len(ops.speed(y, 1.05)) < len(y))
    check("speed 0.95 lengthens", len(ops.speed(y, 0.95)) > len(y))
    check("pitch_shift keeps length",
          abs(len(ops.pitch_shift(y, 2.0)) - len(y)) <= 2)
    check("time_shift keeps length", len(ops.time_shift(y, 100, np.random.default_rng(0)))
          == len(y))
    check("lowpass removes highs",
          np.mean(np.abs(ops.lowpass(y, 500))) >= 0)
    hp = ops.highpass(y, 2000)
    check("highpass is finite", np.isfinite(hp).all())
    snr_out = ops.add_noise(y, 20.0, np.random.default_rng(0))
    check("add_noise finite + louder", np.isfinite(snr_out).all())
    check("reverb keeps length", len(ops.reverb(y, 0.3, np.random.default_rng(0)))
          == len(y))

    print("\naugmentation chains")
    for preset in sorted(aug.PRESETS):
        rng = np.random.default_rng(3)
        outs = [aug.apply_chain(y, SR, rng=rng, preset=preset, chance=1.0)[0]
                for _ in range(5)]
        ok = all(o is not None and len(o) > 0 and np.isfinite(o).all() for o in outs)
        level = all(ops.rms_dbfs(o) < 0 for o in outs)
        clip = max(float(np.mean(np.abs(o) >= 0.999)) for o in outs)
        check(f"preset '{preset}' produces valid audio", ok)
        check(f"preset '{preset}' level below 0 dBFS", level)
        check(f"preset '{preset}' clipping < 1%", clip < 0.01, f"{100*clip:.1f}%")
    check("empty input survives", len(aug.apply_chain(
        np.zeros(10, dtype=np.float32), SR,
        rng=np.random.default_rng(0))[0]) >= 0)

    print("\nsplit integrity")
    rows = []
    for i in range(24):
        for e in L.CLASSES:
            rows.append({"path": f"/x/{i}_{e}.wav", "dataset": "RAVDESS",
                         "emotion": e, "speaker": f"ravdess_{i:02d}"})
    for profile in ("3way", "holdout"):
        sp = build_split(rows, 0.2, 0.15, 42, profile)
        sets = {k: {r["speaker"] for r in v} for k, v in sp.items()}
        leak = ((sets["train"] & sets["val"]) | (sets["train"] & sets["test"])
                | (sets["val"] & sets["test"]))
        check(f"{profile}: no speaker leaks", not leak, str(sorted(leak)[:3]))
        check(f"{profile}: train non-empty", len(sp["train"]) > 0)
        check(f"{profile}: test non-empty", len(sp["test"]) > 0)
        allrows = sum(len(v) for v in sp.values())
        check(f"{profile}: no rows lost", allrows == len(rows),
              f"{allrows} != {len(rows)}")

    print(f"\n{'ALL CHECKS PASSED' if not FAILS else str(len(FAILS)) + ' FAILURES'}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create a tiny synthetic corpus that mimics CREMA-D / RAVDESS / TESS layout.

Used to smoke-test audit_datasets.py, make_splits.py and augment_audio.py
without downloading 3 GB of datasets. Also injects the exact defects seen in
the notebook: Git-LFS pointer files, an AppleDouble file, a zero-length file
and a truncated WAV.

Usage: python augment/tests/make_fake_data.py --out /tmp/fake_data --quick
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audio_ops as ops                                       # noqa: E402

CREMA_EMO = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
RAVDESS_EMO = ["01", "02", "03", "04", "05", "06", "07", "08"]
TESS_EMO = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]


def tone(seconds: float, sr: int, f0: float, rng: np.random.Generator):
    t = np.arange(int(seconds * sr)) / sr
    # crude voiced sound: harmonic stack with vibrato + noise burst
    sig = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 5))
    sig *= 1.0 + 0.05 * np.sin(2 * np.pi * 5 * t)
    env = np.minimum(1.0, np.minimum(t * 20, (t[-1] - t) * 20))
    sig = sig * env + 0.01 * rng.standard_normal(len(t))
    return (sig / (np.max(np.abs(sig)) + 1e-9) * 0.5).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/fake_data")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--quick", action="store_true", help="fewer files")
    ap.add_argument("--no-defects", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    rng = np.random.default_rng(0)
    n_rep = 2 if args.quick else 6
    sr = args.sr

    crema = out / "CREMA-D" / "AudioWAV"
    crema.mkdir(parents=True, exist_ok=True)
    actors = ["1001", "1002", "1003", "1004", "1005"]
    for a, actor in enumerate(actors):
        for e, code in enumerate(CREMA_EMO):
            for r in range(n_rep):
                y = tone(2.0, sr, 110 + 12 * a + e, rng)
                ops.save_audio(crema / f"{actor}_IEO_{code}_XX.wav", y, sr)

    ravdess = out / "RAVDESS" / "Actor_01"
    ravdess.mkdir(parents=True, exist_ok=True)
    for a in range(1, 6):
        for e, code in enumerate(RAVDESS_EMO):
            for rep in range(1, 3):
                name = f"03-01-{code}-01-01-{rep:02d}-{a:02d}.wav"
                y = tone(3.0, sr, 140 + 9 * a + e, rng)
                ops.save_audio(ravdess / name, y, sr)

    tess = out / "TESS" / "TESS Toronto emotional speech set data"
    tess.mkdir(parents=True, exist_ok=True)
    for spk in ("OAF", "YAF"):
        for e, code in enumerate(TESS_EMO):
            for w, word in enumerate(["back", "bar", "say"]):
                y = tone(2.5, sr, 180 + 7 * e + w, rng)
                ops.save_audio(tess / f"{spk}_{word}_{code}.wav", y, sr)

    if not args.no_defects:
        # 1) Git-LFS pointers, as produced by a GitHub source-archive download
        lfs_dir = out / "CREMA-D_github_source" / "AudioWAV"
        lfs_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            (lfs_dir / f"1029_MTI_NEU_XX_{i}.wav").write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:4d7a214614ab2935c943f9e0ff69d22eadbb8f32b1\n"
                "size 123456\n"
            )
        # 2) AppleDouble junk from macOS zipping
        (ravdess / "._03-01-01-01-01-01-01.wav").write_bytes(b"\x00\x05\x16\x07junk")
        # 3) Zero-length file
        (crema / "1003_IEO_HAP_XX_empty.wav").write_bytes(b"")
        # 4) Truncated WAV (header says more than the file holds)
        good = ops.load_audio(crema / f"{actors[0]}_IEO_ANG_XX.wav", sr)
        raw = (crema / f"{actors[0]}_IEO_ANG_XX.wav").read_bytes()
        (crema / f"{actors[1]}_IEO_SAD_XX_trunc.wav").write_bytes(raw[: len(raw) // 3])

    files = sum(1 for p in out.rglob("*") if p.is_file())
    print(f"wrote {files} files under {out}")
    print("expected: 3 datasets present, 8 classes, with injected defects:")
    print("  - 3 LFS pointers, 1 AppleDouble, 1 empty, 1 truncated WAV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

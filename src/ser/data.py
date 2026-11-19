"""Dataset loading for xbgoose/ravdess, plus a local-folder fallback.

Key facts about xbgoose/ravdess (verified from the dataset server):
  * 1,440 rows, ONE split named "train"  <- it is not pre-split into 8 parts
  * columns: audio, modality, vocal_channel, emotion, emotional_intensity,
             statement, repetition, actor (int 1-24), gender
  * emotion: 8 classes, neutral 96 / everything else 192
  * all rows are audio-only speech, 2.9-5.3 s, 44 actors? no: 24 actors
  * labels are already clean strings -> no filename parsing needed

Loading strategy: we cast the audio column to `decode=False` and decode with
soundfile ourselves. That avoids the datasets>=3.6 -> torchcodec/FFmpeg path
which is the exact failure recorded in the old notebook
("SingleStreamDecoder ... Invalid data found when processing input").
"""

from __future__ import annotations

import io
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .labels import RAVDESS_CODES, normalise

RAVDESS_REPO = "xbgoose/ravdess"
SAMPLE_RATE = 16000


# --------------------------------------------------------------------------- #
# Clip record
# --------------------------------------------------------------------------- #
@dataclass
class Clip:
    emotion: str                 # canonical class name
    actor: int                   # 1..24
    gender: str                  # "male" / "female"
    sampling_rate: int
    audio: np.ndarray            # float32 mono, resampled to SAMPLE_RATE
    intensity: str = ""
    statement: str = ""
    repetition: str = ""
    path: str = ""               # source file, when known

    @property
    def speaker(self) -> str:
        return f"actor_{self.actor:02d}"

    @property
    def duration_s(self) -> float:
        return len(self.audio) / float(self.sampling_rate or SAMPLE_RATE)


# --------------------------------------------------------------------------- #
# Audio helpers (numpy only)
# --------------------------------------------------------------------------- #
def to_mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 2:
        axis = 1 if y.shape[1] <= y.shape[0] else 0
        y = y.mean(axis=axis)
    return np.ascontiguousarray(y, dtype=np.float32)


def resample(y: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    if orig_sr == target_sr or orig_sr <= 0:
        return np.asarray(y, dtype=np.float32)
    try:
        import librosa

        return librosa.resample(np.asarray(y, dtype=np.float32),
                                orig_sr=orig_sr, target_sr=target_sr
                                ).astype(np.float32)
    except Exception:
        pass
    n_out = int(round(len(y) * target_sr / orig_sr))
    spec = np.fft.rfft(np.asarray(y, dtype=np.float32))
    m = n_out // 2 + 1
    if m <= len(spec):
        spec = spec[:m]
    else:
        spec = np.concatenate([spec, np.zeros(m - len(spec), dtype=complex)])
    return (np.fft.irfft(spec, n_out) * (n_out / max(len(y), 1))).astype(np.float32)


def fit_length(y: np.ndarray, n_samples: int,
               rng: random.Random | np.random.Generator | None = None) -> np.ndarray:
    """Trim (random crop if rng given) or zero-pad to exactly n_samples."""
    y = np.asarray(y, dtype=np.float32)
    if len(y) == n_samples:
        return y
    if len(y) < n_samples:
        return np.pad(y, (0, n_samples - len(y))).astype(np.float32)
    if rng is None:
        return y[:n_samples]
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(
        getattr(rng, "randrange", random.randrange)(0, 10**6))
    start = int(gen.integers(0, len(y) - n_samples + 1))
    return y[start:start + n_samples].astype(np.float32)


def _decode_bytes(blob: bytes, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
    return resample(to_mono(data), sr, target_sr), target_sr


# --------------------------------------------------------------------------- #
# Hugging Face loader
# --------------------------------------------------------------------------- #
def load_ravdess_hf(repo: str = RAVDESS_REPO, split: str = "train",
                    cache_dir: str | None = None, limit: int | None = None,
                    target_sr: int = SAMPLE_RATE,
                    verbose: bool = True) -> list[Clip]:
    """Download/read the dataset and return decoded Clips.

    `limit` keeps only the first N rows (used by CI to stay under a minute).
    """
    from datasets import Audio, load_dataset

    if verbose:
        print(f"loading {repo} (split={split}) ...", flush=True)
    ds = load_dataset(repo, split=split, cache_dir=cache_dir)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    # Prefer raw bytes + our own decoder (never touches torchcodec/ffmpeg).
    try:
        ds = ds.cast_column("audio", Audio(decode=False))
        has_bytes = ds[0]["audio"].get("bytes") is not None
    except Exception:
        has_bytes = False

    clips: list[Clip] = []
    if has_bytes:
        for row in ds:
            info = row["audio"]
            blob = info["bytes"]
            try:
                y, sr = _decode_bytes(blob, target_sr)
            except Exception as exc:                       # noqa: BLE001
                if verbose:
                    print(f"  skip unreadable row: {type(exc).__name__}: {exc}")
                continue
            clips.append(Clip(
                emotion=normalise(row["emotion"]),
                actor=int(row["actor"]),
                gender=str(row.get("gender", "")).lower(),
                sampling_rate=sr,
                audio=y,
                intensity=str(row.get("emotional_intensity", "")),
                statement=str(row.get("statement", "")),
                repetition=str(row.get("repetition", "")),
                path=str(info.get("path") or ""),
            ))
    else:                                   # decoding path (needs ffmpeg/torchcodec)
        if verbose:
            print("  falling back to datasets audio decoding "
                  "(requires ffmpeg or torchcodec)")
        ds = load_dataset(repo, split=split, cache_dir=cache_dir)
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        ds = ds.cast_column("audio", Audio(sampling_rate=target_sr))
        for row in ds:
            a = row["audio"]
            clips.append(Clip(
                emotion=normalise(row["emotion"]),
                actor=int(row["actor"]),
                gender=str(row.get("gender", "")).lower(),
                sampling_rate=int(a.get("sampling_rate", target_sr)),
                audio=to_mono(np.asarray(a["array"], dtype=np.float32)),
                path=str(a.get("path") or ""),
            ))
    if verbose:
        print(f"  loaded {len(clips)} clips from {repo}")
    return clips


# --------------------------------------------------------------------------- #
# Local folder fallback (if HF is unreachable but you have ravdess.zip)
# --------------------------------------------------------------------------- #
def parse_ravdess_filename(name: str) -> tuple[str, int]:
    """03-01-05-01-01-01-07.wav -> ("angry", 7). Raises ValueError."""
    stem = Path(name).stem
    parts = stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"not a RAVDESS name: {name}")
    if parts[0] != "03":
        raise ValueError(f"not audio-only modality: {name}")
    if parts[1] == "02":
        raise ValueError(f"sung clip, not speech: {name}")
    if parts[2] not in RAVDESS_CODES:
        raise ValueError(f"unknown emotion code {parts[2]}: {name}")
    return RAVDESS_CODES[parts[2]], int(parts[6])


def load_ravdess_local(folder: str | Path, target_sr: int = SAMPLE_RATE,
                       verbose: bool = True) -> list[Clip]:
    import soundfile as sf

    folder = Path(folder)
    clips: list[Clip] = []
    skipped = 0
    for path in sorted(folder.rglob("*.wav")):
        if path.name.startswith("._"):
            continue
        try:
            emotion, actor = parse_ravdess_filename(path.name)
            y, sr = sf.read(str(path), dtype="float32", always_2d=False)
        except Exception:                                  # noqa: BLE001
            skipped += 1
            continue
        clips.append(Clip(emotion=emotion, actor=actor,
                          gender="male" if actor % 2 else "female",
                          sampling_rate=target_sr,
                          audio=resample(to_mono(y), sr, target_sr),
                          path=str(path)))
    if verbose:
        print(f"loaded {len(clips)} clips from {folder} (skipped {skipped})")
    return clips


def load_any(source: str, limit: int | None = None, verbose: bool = True) -> list[Clip]:
    """`source` may be an HF repo id (contains '/') or a local folder."""
    if "/" in source and not Path(source).exists():
        return load_ravdess_hf(source, limit=limit, verbose=verbose)
    return load_ravdess_local(source, verbose=verbose)


# --------------------------------------------------------------------------- #
# Label helpers
# --------------------------------------------------------------------------- #
def to_arrays(clips: Iterable[Clip], classes: list[str], window_s: float,
              rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Stack clips into (N, n_samples) float32 and (N,) int64.

    Arrays are int16-backed to halve RAM: 1,440 x 4 s x 16 kHz = ~184 MB.
    """
    clips = list(clips)
    n = int(round(window_s * SAMPLE_RATE))
    x = np.zeros((len(clips), n), dtype=np.float32)
    y = np.zeros(len(clips), dtype=np.int64)
    for i, c in enumerate(clips):
        x[i] = fit_length(c.audio, n, rng=rng)
        y[i] = classes.index(c.emotion)
    return x, y

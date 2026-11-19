"""Feature preparation + a cached npz format that needs no pickle.

Why not store log-mels: wav2vec2 consumes raw waveforms. Extracting log-mels
and feeding them to a wav2vec2 checkpoint means re-deriving the 16 kHz signal,
which is lossy. Keep the (normalised) waveform.

Why a flat blob instead of an object array: object arrays need
allow_pickle=True, which is a code-execution vector and breaks across numpy
versions. Flat data + lengths is portable, compresses well, and loads without
pickle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import SAMPLE_RATE

# wav2vec2 feature extractor normalises to zero mean / unit variance per clip;
# we do the same here so the cached arrays match the model's expectation and
# so training does not depend on each clip's recording level.
def normalize(x: np.ndarray) -> np.ndarray:
    mean = float(np.mean(x))
    std = float(np.std(x))
    return ((x - mean) / (std + 1e-7)).astype(np.float32)


def to_flat(x: np.ndarray, dtype=np.float16) -> dict:
    """(N, T) padded float array -> flat blob + lengths (no pickles)."""
    lengths = np.asarray([len(row) for row in x], dtype=np.int32)
    return {
        "data": np.concatenate([np.asarray(r, dtype=dtype) for r in x])
        if len(x) else np.zeros(0, dtype=dtype),
        "lengths": lengths,
    }


def from_flat(blob: dict, row: int) -> np.ndarray:
    end = int(np.cumsum(blob["lengths"])[row])
    start = end - int(blob["lengths"][row])
    return blob["data"][start:end].astype(np.float32)


def save_npz(path: str | Path, x: np.ndarray, y: np.ndarray,
             actors: np.ndarray, flag: np.ndarray | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = to_flat(x)
    arrays = {"data": flat["data"], "lengths": flat["lengths"],
              "labels": np.asarray(y, dtype=np.int64),
              "actors": np.asarray(actors, dtype=np.int32)}
    if flag is not None:
        arrays["is_augmented"] = np.asarray(flag, dtype=np.int8)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)                    # atomic: never a half-written cache


def load_npz(path: str | Path, limit: int | None = None) -> dict:
    with np.load(Path(path)) as z:        # note: no allow_pickle needed
        out = {k: z[k] for k in z.files}
    if limit:
        for k in out:
            out[k] = out[k][:limit]
    return out


def n_samples(window_s: float, sr: int = SAMPLE_RATE) -> int:
    return int(round(window_s * sr))

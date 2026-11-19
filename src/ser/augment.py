"""Emotion-preserving waveform augmentation (numpy + soundfile only).

Why these transforms and not others, for this dataset:
  * RAVDESS is studio-recorded with 24 actors. The live demo uses a laptop mic.
    The biggest measured gap is channel + room, not "not enough data".
  * `calm` and `neutral` are 2.7 % of the corpus each after merging and the
    previous run could not recall them at all. Extra copies go there.
  * Anything that can change the *emotion* (background music, second speaker,
    heavy stretch) is deliberately absent: with 192 clips per class you cannot
    afford label noise.

Every function is pure and seeded, so runs are reproducible.
"""

from __future__ import annotations

import numpy as np

from .data import SAMPLE_RATE, fit_length

_EPS = 1e-9


# --------------------------------------------------------------------------- #
# Level
# --------------------------------------------------------------------------- #
def rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y, dtype=np.float64)) + _EPS))


def rms_dbfs(y: np.ndarray) -> float:
    return float(20.0 * np.log10(max(rms(y), 1e-8)))


def rms_normalize(y: np.ndarray, target_dbfs: float = -23.0) -> np.ndarray:
    gain = 10.0 ** ((target_dbfs - rms_dbfs(y)) / 20.0)
    return (y * gain).astype(np.float32)


def peak_normalize(y: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(y))) if len(y) else 0.0
    return y if m < _EPS else (y * (peak / m)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Spectral
# --------------------------------------------------------------------------- #
def _gain_curve(y: np.ndarray, curve, sr: int) -> np.ndarray:
    spec = np.fft.rfft(y)
    f = np.fft.rfftfreq(len(y), 1.0 / sr)
    return np.fft.irfft(spec * curve(f), len(y)).astype(np.float32)


def lowpass(y: np.ndarray, cutoff: float, sr: int = SAMPLE_RATE,
            roll: float = 0.15) -> np.ndarray:
    lo, hi = cutoff * (1 - roll), cutoff * (1 + roll)

    def curve(f):
        g = np.ones_like(f)
        g[f >= hi] = 0.0
        band = (f > lo) & (f < hi)
        g[band] = 0.5 * (1 + np.cos(np.pi * (f[band] - lo) / (hi - lo)))
        return g

    return _gain_curve(y, curve, sr)


def highpass(y: np.ndarray, cutoff: float, sr: int = SAMPLE_RATE,
             roll: float = 0.15) -> np.ndarray:
    lo, hi = cutoff * (1 - roll), cutoff * (1 + roll)

    def curve(f):
        g = np.ones_like(f)
        g[f <= lo] = 0.0
        band = (f > lo) & (f < hi)
        g[band] = 0.5 * (1 - np.cos(np.pi * (f[band] - lo) / (hi - lo)))
        return g

    return _gain_curve(y, curve, sr)


def telephone(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """300-3400 Hz: the cheap-mic / phone channel the demo actually sees."""
    return highpass(lowpass(y, 3400, sr), 300, sr)


# --------------------------------------------------------------------------- #
# Noise
# --------------------------------------------------------------------------- #
def pink_noise(n: int, rng: np.random.Generator, sr: int = SAMPLE_RATE) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    f[0] = f[1] if len(f) > 1 else 1.0
    out = np.fft.irfft(spec / np.sqrt(f), n)
    return (out / (np.std(out) + _EPS)).astype(np.float32)


def add_noise(y: np.ndarray, snr_db: float, rng: np.random.Generator,
              kind: str = "pink") -> np.ndarray:
    noise = (rng.standard_normal(len(y)).astype(np.float32) if kind == "white"
             else pink_noise(len(y), rng))
    p_sig = float(np.mean(np.square(y, dtype=np.float64))) + _EPS
    p_noise = float(np.mean(np.square(noise, dtype=np.float64))) + _EPS
    scale = np.sqrt((p_sig / (10 ** (snr_db / 10.0))) / p_noise)
    return (y + noise * scale).astype(np.float32)


# --------------------------------------------------------------------------- #
# Room
# --------------------------------------------------------------------------- #
def synthetic_rir(rt60: float, rng: np.random.Generator,
                  sr: int = SAMPLE_RATE) -> np.ndarray:
    n = max(int(rt60 * sr * 1.2), 32)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / max(rt60, 1e-3))
    ir[0] = 1.0
    ir[: max(1, int(0.005 * sr))] *= 1.5
    return (ir / (np.sqrt(np.sum(ir ** 2)) + _EPS)).astype(np.float32)


def reverb(y: np.ndarray, rt60: float, rng: np.random.Generator,
           wet: float = 0.3, sr: int = SAMPLE_RATE) -> np.ndarray:
    ir = synthetic_rir(rt60, rng, sr)
    wet_sig = np.convolve(y, ir)[: len(y)]
    return ((1 - wet) * y + wet * wet_sig).astype(np.float32)


# --------------------------------------------------------------------------- #
# Time / pitch
# --------------------------------------------------------------------------- #
def _resample_n(y: np.ndarray, n_out: int) -> np.ndarray:
    if len(y) == n_out or len(y) == 0:
        return np.asarray(y, dtype=np.float32)
    spec = np.fft.rfft(y)
    m = n_out // 2 + 1
    spec = spec[:m] if m <= len(spec) else np.concatenate(
        [spec, np.zeros(m - len(spec), dtype=complex)])
    return (np.fft.irfft(spec, n_out) * (n_out / len(y))).astype(np.float32)


def speed(y: np.ndarray, factor: float) -> np.ndarray:
    """Kaldi-style speed perturbation: tempo AND pitch move together."""
    return _resample_n(y, int(round(len(y) / max(factor, 1e-3))))


def pitch_shift(y: np.ndarray, semitones: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    if abs(semitones) < 1e-3:
        return y
    try:
        import librosa

        return librosa.effects.pitch_shift(y, sr=sr, n_steps=float(semitones)
                                           ).astype(np.float32)
    except Exception:
        return _resample_n(_resample_n(y, int(len(y) / 2 ** (semitones / 12))),
                           len(y))


def time_shift(y: np.ndarray, max_ms: float, rng: np.random.Generator,
               sr: int = SAMPLE_RATE) -> np.ndarray:
    k = int(rng.integers(-int(max_ms * sr / 1000), int(max_ms * sr / 1000) + 1))
    out = np.zeros_like(y)
    if k > 0:
        out[k:] = y[:-k]
    elif k < 0:
        out[:k] = y[-k:]
    else:
        out = y.copy()
    return out


def time_mask(y: np.ndarray, n_masks: int, max_ms: float,
              rng: np.random.Generator, sr: int = SAMPLE_RATE) -> np.ndarray:
    out = y.copy()
    for _ in range(n_masks):
        width = int(rng.integers(1, max(2, int(max_ms * sr / 1000))))
        start = int(rng.integers(0, max(len(out) - width, 1)))
        out[start:start + width] = 0.0
    return out


# --------------------------------------------------------------------------- #
# Chains
# --------------------------------------------------------------------------- #
PRESETS: dict[str, dict[str, float]] = {
    #          transform: probability
    "light":     {"gain": 0.6, "noise": 0.5},
    "speed":     {"speed": 0.5, "gain": 0.5},
    "channel":   {"telephone": 0.5, "gain": 0.5, "noise": 0.4},
    "room":      {"reverb": 0.4, "gain": 0.5, "noise": 0.4},
    "minority":  {"speed": 0.5, "noise": 0.5, "gain": 0.6,
                  "shift": 0.4, "pitch": 0.2},
    "full":      {"gain": 0.6, "noise": 0.5, "speed": 0.45, "pitch": 0.15,
                  "reverb": 0.3, "telephone": 0.25, "shift": 0.35,
                  "mask": 0.2},
}

# Rare classes get more copies; they are the ones the old model never recalled.
CLASS_COPY_BOOST = {"calm": 2.0, "neutral": 1.3, "surprised": 1.3}


def _apply_one(name: str, y: np.ndarray, rng: np.random.Generator,
               sr: int) -> np.ndarray:
    if name == "gain":
        return rms_normalize(y, float(rng.uniform(-28, -16)))
    if name == "noise":
        return add_noise(y, float(rng.uniform(12, 30)), rng, "pink")
    if name == "white_noise":
        return add_noise(y, float(rng.uniform(15, 35)), rng, "white")
    if name == "speed":
        return speed(y, float(rng.uniform(0.92, 1.08)))
    if name == "pitch":
        return pitch_shift(y, float(rng.uniform(-1.5, 1.5)), sr)
    if name == "reverb":
        return reverb(y, float(rng.uniform(0.15, 0.45)), rng,
                      float(rng.uniform(0.15, 0.35)), sr)
    if name == "telephone":
        return telephone(y, sr)
    if name == "shift":
        return time_shift(y, float(rng.uniform(20, 150)), rng, sr)
    if name == "mask":
        return time_mask(y, int(rng.integers(1, 3)), float(rng.uniform(40, 120)),
                         rng, sr)
    raise KeyError(f"unknown transform {name}")


def augment(y: np.ndarray, rng: np.random.Generator, preset: str = "full",
            sr: int = SAMPLE_RATE, scale: float = 1.0,
            length: int | None = None) -> tuple[np.ndarray, list[str]]:
    """Apply a random subset of `preset`. Returns (audio, applied names).

    `scale` multiplies every probability (use 0.5 for a gentler mix).
    `length` forces the output to exactly that many samples -- required, since
    `speed` deliberately changes clip length and callers stack into arrays.
    """
    spec = PRESETS.get(preset, PRESETS["full"])
    out = np.asarray(y, dtype=np.float32)
    applied: list[str] = []
    for name, prob in spec.items():
        if rng.random() > min(prob * scale, 1.0):
            continue
        try:
            cand = _apply_one(name, out, rng, sr)
            if cand is not None and len(cand) and np.isfinite(cand).all():
                out, _ = cand, applied.append(name)
        except Exception:                                  # noqa: BLE001
            continue                    # an augmentation must never kill a run
    if length is not None:
        out = fit_length(out, length, rng)
    out = peak_normalize(rms_normalize(out, -23.0), 0.95)
    return out.astype(np.float32), applied


def copies_for(label: str, base_copies: int) -> int:
    """Per-class copy count (rare classes get more)."""
    return max(0, int(round(base_copies * CLASS_COPY_BOOST.get(label, 1.0))))


def expand(x: np.ndarray, y: np.ndarray, classes: list[str],
           copies: int, preset: str = "full", seed: int = 0,
           scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Augment every clip `copies` times (boosted per class).

    Returns (x_all, y_all, is_augmented) with the originals first. Output rows
    always have the same length as the input rows, so the result can be stacked.
    """
    rng = np.random.default_rng(seed)
    n = x.shape[1]
    xs, ys, flags = [x], [y], [np.zeros(len(y), dtype=np.int8)]
    names = np.array(classes)
    for i in range(len(x)):
        label = names[y[i]]
        k = copies_for(label, copies)
        if k <= 0:
            continue
        for _ in range(k):
            aug, _ = augment(x[i], rng, preset=preset, scale=scale, length=n)
            xs.append(aug[None, :])
            ys.append(np.array([y[i]]))
            flags.append(np.ones(1, dtype=np.int8))
    return (np.concatenate(xs, axis=0).astype(np.float32),
            np.concatenate(ys, axis=0).astype(np.int64),
            np.concatenate(flags, axis=0))

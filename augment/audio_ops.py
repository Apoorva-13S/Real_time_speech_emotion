"""Pure numpy/soundfile audio operations for emotion-preserving augmentation.

Deliberately dependency-light: numpy + soundfile only. librosa, if installed,
is used opportunistically for high-quality resampling and pitch shifting, but
every function has a numpy fallback so this runs in a bare environment.

All functions operate on mono float32 numpy arrays in [-1, 1].

Design rule: an augmentation must not change the *emotion* of the clip.
Gain, mild noise, mild speed perturbation, light reverb and band-limiting all
preserve vocal effort and prosody. Heavy time-stretch, extreme pitch shift,
background music or added speech do not -- those are deliberately absent.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

TARGET_SR = 16000
_EPS = 1e-9


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_audio(path, sr: int = TARGET_SR) -> np.ndarray:
    """Read any soundfile-supported audio as mono float32 at `sr`."""
    y, orig_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if orig_sr != sr:
        y = resample(y, orig_sr, sr)
    return np.ascontiguousarray(y, dtype=np.float32)


def save_audio(path, y: np.ndarray, sr: int = TARGET_SR) -> None:
    """Write 16-bit PCM WAV (compact, universally readable)."""
    y = np.clip(np.asarray(y, dtype=np.float32), -1.0, 1.0)
    sf.write(str(path), y, sr, subtype="PCM_16")


def duration_s(y: np.ndarray, sr: int = TARGET_SR) -> float:
    return len(y) / float(sr)


# --------------------------------------------------------------------------- #
# Resampling (librosa if present, otherwise FFT)
# --------------------------------------------------------------------------- #
def _fft_resample(y: np.ndarray, n_out: int) -> np.ndarray:
    n_in = len(y)
    if n_in == 0 or n_out == 0:
        return np.zeros(max(n_out, 0), dtype=np.float32)
    spec = np.fft.rfft(y)
    m_out = n_out // 2 + 1
    if m_out <= len(spec):
        out = spec[:m_out].copy()
    else:
        out = np.concatenate([spec, np.zeros(m_out - len(spec), dtype=complex)])
    y_out = np.fft.irfft(out, n_out)
    return (y_out * (n_out / n_in)).astype(np.float32)


def resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return np.asarray(y, dtype=np.float32)
    try:  # better anti-aliasing if available
        import librosa

        return librosa.resample(
            np.asarray(y, dtype=np.float32),
            orig_sr=orig_sr,
            target_sr=target_sr,
        ).astype(np.float32)
    except Exception:
        pass
    return _fft_resample(
        np.asarray(y, dtype=np.float32),
        int(round(len(y) * target_sr / orig_sr)),
    )


def _stretch_to_length(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) == n:
        return y
    return _fft_resample(y, n)


# --------------------------------------------------------------------------- #
# Level helpers
# --------------------------------------------------------------------------- #
def rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y, dtype=np.float64)) + _EPS))


def rms_dbfs(y: np.ndarray) -> float:
    return float(20.0 * np.log10(max(rms(y), 1e-8)))


def peak_normalize(y: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(y))) if len(y) else 0.0
    return y if m < _EPS else (y * (peak / m)).astype(np.float32)


def rms_normalize(y: np.ndarray, target_dbfs: float = -23.0) -> np.ndarray:
    cur = rms_dbfs(y)
    gain = 10.0 ** ((target_dbfs - cur) / 20.0)
    return (y * gain).astype(np.float32)


# --------------------------------------------------------------------------- #
# Spectral shaping (cheap, FFT based; no scipy needed)
# --------------------------------------------------------------------------- #
def _apply_gain_curve(y: np.ndarray, freqs: "callable", sr: int) -> np.ndarray:
    n = len(y)
    spec = np.fft.rfft(y)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    return np.fft.irfft(spec * freqs(f), n).astype(np.float32)


def lowpass(y: np.ndarray, cutoff: float, sr: int = TARGET_SR,
            roll: float = 0.15) -> np.ndarray:
    """Smooth raised-cosine lowpass: 1 below cutoff*(1-roll), 0 above cutoff."""
    lo, hi = cutoff * (1.0 - roll), cutoff * (1.0 + roll)

    def curve(f):
        g = np.ones_like(f)
        g[f >= hi] = 0.0
        band = (f > lo) & (f < hi)
        g[band] = 0.5 * (1.0 + np.cos(np.pi * (f[band] - lo) / (hi - lo)))
        return g

    return _apply_gain_curve(y, curve, sr)


def highpass(y: np.ndarray, cutoff: float, sr: int = TARGET_SR,
             roll: float = 0.15) -> np.ndarray:
    lo, hi = cutoff * (1.0 - roll), cutoff * (1.0 + roll)

    def curve(f):
        g = np.ones_like(f)
        g[f <= lo] = 0.0
        band = (f > lo) & (f < hi)
        g[band] = 0.5 * (1.0 - np.cos(np.pi * (f[band] - lo) / (hi - lo)))
        return g

    return _apply_gain_curve(y, curve, sr)


# --------------------------------------------------------------------------- #
# Noise
# --------------------------------------------------------------------------- #
def white_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n).astype(np.float32)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise: closer to real room/mic noise than white noise."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / TARGET_SR)
    f[0] = f[1] if len(f) > 1 else 1.0
    spec = spec / np.sqrt(f)
    out = np.fft.irfft(spec, n)
    return (out / (np.std(out) + _EPS)).astype(np.float32)


def add_noise(y: np.ndarray, snr_db: float, rng: np.random.Generator,
              kind: str = "pink") -> np.ndarray:
    noise = pink_noise(len(y), rng) if kind == "pink" else white_noise(len(y), rng)
    p_sig = float(np.mean(np.square(y, dtype=np.float64))) + _EPS
    p_noise = float(np.mean(np.square(noise, dtype=np.float64))) + _EPS
    target = p_sig / (10.0 ** (snr_db / 10.0))
    return (y + noise * np.sqrt(target / p_noise)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Reverb (synthetic RIR: models laptop-mic / room mismatch seen at inference)
# --------------------------------------------------------------------------- #
def synthetic_rir(rt60: float, rng: np.random.Generator,
                  sr: int = TARGET_SR) -> np.ndarray:
    n = max(int(rt60 * sr * 1.2), 32)
    t = np.arange(n) / sr
    decay = np.exp(-6.9 * t / max(rt60, 1e-3))
    ir = rng.standard_normal(n) * decay
    ir[0] = 1.0                     # direct path
    ir[: max(1, int(0.005 * sr))] *= 1.5
    ir /= np.sqrt(np.sum(ir ** 2)) + _EPS
    return ir.astype(np.float32)


def reverb(y: np.ndarray, rt60: float, rng: np.random.Generator,
           wet: float = 0.35, sr: int = TARGET_SR) -> np.ndarray:
    ir = synthetic_rir(rt60, rng, sr)
    wet_sig = np.convolve(y, ir)[: len(y)]
    return ((1.0 - wet) * y + wet * wet_sig).astype(np.float32)


# --------------------------------------------------------------------------- #
# Time / pitch
# --------------------------------------------------------------------------- #
def time_shift(y: np.ndarray, max_ms: float, rng: np.random.Generator,
               sr: int = TARGET_SR) -> np.ndarray:
    """Shift content inside a fixed-length window (zero fill, no wrap-around)."""
    k = int(rng.integers(-int(max_ms * sr / 1000), int(max_ms * sr / 1000) + 1))
    out = np.zeros_like(y)
    if k > 0:
        out[k:] = y[:-k]
    elif k < 0:
        out[:k] = y[-k:]
    else:
        out = y.copy()
    return out


def speed(y: np.ndarray, factor: float, sr: int = TARGET_SR) -> np.ndarray:
    """Kaldi-style speed perturbation (changes tempo AND pitch together).

    This is the single most reliable cheap augmentation in ASR/SER recipes.
    Output length changes by 1/factor; callers trim/pad as needed.
    """
    n_out = int(round(len(y) / max(factor, 1e-3)))
    return _fft_resample(y, n_out)


def pitch_shift(y: np.ndarray, semitones: float, sr: int = TARGET_SR) -> np.ndarray:
    """Pitch shift with unchanged duration. Uses librosa when available."""
    if abs(semitones) < 1e-3:
        return y
    try:
        import librosa

        return librosa.effects.pitch_shift(
            y, sr=sr, n_steps=float(semitones)
        ).astype(np.float32)
    except Exception:
        # Fallback: compromise between pitch and tempo change, restored length.
        shifted = _fft_resample(y, int(round(len(y) / 2.0 ** (semitones / 12.0))))
        return _stretch_to_length(shifted, len(y))


def time_mask_wave(y: np.ndarray, n_masks: int, max_ms: float,
                   rng: np.random.Generator, sr: int = TARGET_SR) -> np.ndarray:
    """Zero a few short spans (a mild, waveform-domain SpecAugment)."""
    out = y.copy()
    n = len(out)
    if n == 0:
        return out
    max_len = max(int(max_ms * sr / 1000), 1)
    for _ in range(n_masks):
        width = int(rng.integers(1, max_len + 1))
        start = int(rng.integers(0, max(n - width, 1)))
        out[start:start + width] = 0.0
    return out


# --------------------------------------------------------------------------- #
# Within-class mixup (safe: both clips share the label)
# --------------------------------------------------------------------------- #
def mix_with(y: np.ndarray, other: np.ndarray, rng: np.random.Generator,
             alpha: float = 0.3) -> np.ndarray:
    n = max(len(y), len(other))
    a = np.pad(y, (0, n - len(y)))
    b = np.pad(other, (0, n - len(other)))
    lam = float(rng.beta(alpha, alpha)) if alpha > 0 else 0.5
    return (lam * a + (1.0 - lam) * b).astype(np.float32)


# --------------------------------------------------------------------------- #
# Fit to a fixed window (what the model expects)
# --------------------------------------------------------------------------- #
def fit_window(y: np.ndarray, seconds: float, sr: int = TARGET_SR,
               rng: np.random.Generator | None = None,
               pad_value: float = 0.0) -> np.ndarray:
    """Trim to `seconds`, or zero-pad. Random crop when a rng is supplied."""
    n = int(seconds * sr)
    if len(y) <= n:
        return np.pad(y, (0, n - len(y)), constant_values=pad_value).astype(np.float32)
    if rng is None:
        start = 0
    else:
        start = int(rng.integers(0, len(y) - n + 1))
    return y[start:start + n].astype(np.float32)

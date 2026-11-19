"""Shared, strict dataset parsing for CREMA-D / RAVDESS / TESS.

Every other module imports the label maps and filename parsers from here, so
there is exactly one definition of "what is the label of this file". The old
notebook had nine slightly different copies; that is how the 400 TESS
"pleasant surprise" files got silently dropped.

Filename grammars
-----------------
CREMA-D : 1001_IEO_ANG_XX.wav          actor_sentence_EMOTION_level
RAVDESS : 03-01-05-01-01-01-07.wav     modality-vocal-emotion-int-stmt-rep-actor
TESS    : OAF_back_angry.wav           speaker_word_emotion

Emotion vocabularies
--------------------
CREMA-D : ANG DIS FEA HAP NEU SAD                         (6)
RAVDESS : 01..08 = neutral calm happy sad angry fearful
          disgust surprised                            (8)
TESS    : angry disgust fear happy neutral ps sad         (7, "ps" = pleasant
          surprise -> surprised)
"""

from __future__ import annotations

import re
from pathlib import Path

# The project's class order. Label ids are positions in this list.
CLASSES = [
    "angry", "calm", "disgust", "fearful",
    "happy", "neutral", "sad", "surprised",
]
LABEL2ID = {name: i for i, name in enumerate(CLASSES)}
ID2LABEL = {i: name for i, name in enumerate(CLASSES)}

# 7-class variant (calm merged into neutral) -- the recommended target set.
CLASSES_7 = ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
MERGE_7 = {"calm": "neutral"}

CREMA_MAP = {
    "ANG": "angry", "DIS": "disgust", "FEA": "fearful",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad",
}
RAVDESS_MAP = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
# "ps" is the one the notebook missed. Keep every alias in one place.
TESS_MAP = {
    "angry": "angry", "disgust": "disgust", "fear": "fearful",
    "fearful": "fearful", "happy": "happy", "neutral": "neutral",
    "sad": "sad", "ps": "surprised", "surprise": "surprised",
    "surprised": "surprised", "pleasant_surprise": "surprised",
}

DATASET_EMOTIONS = {
    "CREMA-D": sorted(set(CREMA_MAP.values())),
    "RAVDESS": sorted(set(RAVDESS_MAP.values())),
    "TESS": sorted(set(TESS_MAP.values())),
}

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif")


class ParseError(ValueError):
    """Raised when a filename does not match its dataset's grammar."""


def is_audio(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTS


def is_appledouble(path: Path) -> bool:
    return path.name.startswith("._")


def parse_crema(path: Path) -> tuple[str, str]:
    """Return (emotion, speaker)."""
    parts = path.stem.split("_")
    if len(parts) < 3:
        raise ParseError(f"CREMA-D needs >=3 '_' fields: {path.name}")
    code = parts[2].upper()
    if code not in CREMA_MAP:
        raise ParseError(f"unknown CREMA-D emotion code {code!r}: {path.name}")
    return CREMA_MAP[code], f"crema_{parts[0]}"


# RAVDESS vocal channel: 01 = speech, 02 = song. Song is rarely useful for
# speech-emotion work and its labels describe sung affect, so it is excluded by
# default. Set this False if you deliberately want the sung material.
RAVDESS_SPEECH_ONLY = True


def parse_ravdess(path: Path) -> tuple[str, str]:
    """Return (emotion, speaker). Raises on non-speech or malformed names."""
    parts = path.stem.split("-")
    if len(parts) != 7:
        raise ParseError(f"RAVDESS needs 7 '-' fields: {path.name}")
    modality, channel, emotion_code, actor = parts[0], parts[1], parts[2], parts[6]
    if modality != "03":
        raise ParseError(f"not audio-only (modality={modality}): {path.name}")
    if RAVDESS_SPEECH_ONLY and channel == "02":
        raise ParseError(f"sung clip, not speech (vocal channel=02): {path.name}")
    if emotion_code not in RAVDESS_MAP:
        raise ParseError(f"unknown RAVDESS emotion code {emotion_code!r}: {path.name}")
    return RAVDESS_MAP[emotion_code], f"ravdess_{actor}"


def parse_tess(path: Path) -> tuple[str, str]:
    """Return (emotion, speaker). TESS names: speaker_word_emotion.wav"""
    tokens = path.stem.split("_")
    if len(tokens) < 3:
        raise ParseError(f"TESS needs >=3 '_' fields: {path.name}")
    code = tokens[-1].lower()
    if code not in TESS_MAP:
        raise ParseError(f"unknown TESS emotion code {code!r}: {path.name}")
    speaker = tokens[0].upper()
    return TESS_MAP[code], f"tess_{speaker}"


PARSERS = {"CREMA-D": parse_crema, "RAVDESS": parse_ravdess, "TESS": parse_tess}


def parse(dataset: str, path: Path) -> tuple[str, str]:
    return PARSERS[dataset](path)


def to_7class(emotion: str) -> str:
    return MERGE_7.get(emotion, emotion)


def fingerprint_speaker(dataset: str, speaker: str) -> str:
    """Prefix speakers with the dataset so ids never collide across corpora."""
    return speaker if "_" in speaker else f"{dataset.lower()}_{speaker}"


_LFS_RE = re.compile(rb"^version https://git-lfs")


def looks_like_lfs_pointer(path: Path) -> bool:
    """CREMA-D fetched from a GitHub source archive yields 130-byte pointers."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(64)
    except OSError:
        return False
    return bool(_LFS_RE.match(head))


def has_wav_header(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            head = fh.read(12)
    except OSError:
        return False
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return True
    if head[:4] == b"fLaC":                     # flac
        return True
    return False

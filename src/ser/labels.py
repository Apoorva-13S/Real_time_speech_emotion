"""Single source of truth for emotion labels.

The dataset (xbgoose/ravdess) already carries a clean `emotion` string column,
so no filename parsing is needed for the main pipeline. These maps exist so
that legacy filename-based data and the HF column agree, and so that CI can
assert they never drift.
"""

from __future__ import annotations

CLASSES = [
    "angry", "calm", "disgust", "fearful",
    "happy", "neutral", "sad", "surprised",
]
LABEL2ID = {name: i for i, name in enumerate(CLASSES)}
ID2LABEL = {i: name for i, name in enumerate(CLASSES)}

# 7-class variant: calm folded into neutral (see process.txt step "why 7 vs 8")
CLASSES_7 = ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
MERGE_7 = {"calm": "neutral"}

# Label variants seen across corpora / model cards.
ALIASES = {
    "anger": "angry",
    "angry": "angry",
    "calm": "calm",
    "disgust": "disgust",
    "fear": "fearful",
    "fearful": "fearful",
    "happy": "happy",
    "happiness": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprised",
    "surprised": "surprised",
    "ps": "surprised",                 # TESS "pleasant surprise"
    "pleasant surprise": "surprised",
}

# RAVDESS filename code -> label (fallback path only)
RAVDESS_CODES = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
CREMA_CODES = {
    "ANG": "angry", "DIS": "disgust", "FEA": "fearful",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad",
}

# Known, documented properties of xbgoose/ravdess (verified via the dataset
# viewer / statistics endpoint). Used by tests and by the audit report.
EXPECTED_ACTORS = 24
EXPECTED_ROWS = 1440
EXPECTED_COUNTS = {
    "neutral": 96, "calm": 192, "angry": 192, "disgust": 192,
    "fearful": 192, "happy": 192, "sad": 192, "surprised": 192,
}
DURATION_RANGE_S = (2.9, 5.3)


def normalise(label: str) -> str:
    """Map any known spelling to a canonical class; raise on unknown."""
    key = str(label).strip().lower()
    if key in ALIASES:
        return ALIASES[key]
    raise ValueError(f"unknown emotion label: {label!r}")


def to_7(label: str) -> str:
    return MERGE_7.get(normalise(label), normalise(label))


def names_for(n_classes: int) -> list[str]:
    if n_classes == 8:
        return list(CLASSES)
    if n_classes == 7:
        return list(CLASSES_7)
    raise ValueError("only 7 or 8 classes are supported")


def id2label_for(names: list[str]) -> dict[int, str]:
    return {i: n for i, n in enumerate(names)}


def label2id_for(names: list[str]) -> dict[str, int]:
    return {n: i for i, n in enumerate(names)}

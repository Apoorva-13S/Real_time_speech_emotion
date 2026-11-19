"""Fast tests that need no network, no GPU, no dataset download.

    python -m pytest tests -q          (or)  python tests/test_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ser import augment, config, features, labels, metrics, splits  # noqa: E402
from ser.data import fit_length, parse_ravdess_filename, resample, to_mono  # noqa: E402

SR = 16000
FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILS.append(name)


def tone(seconds: float = 2.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    y = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 4))
    y += 0.02 * rng.standard_normal(len(t))
    return (y / np.max(np.abs(y)) * 0.4).astype(np.float32)


def test_labels() -> None:
    check("8 class names", len(labels.CLASSES) == 8)
    check("7 class names", len(labels.CLASSES_7) == 7)
    check("calm merges to neutral", labels.to_7("calm") == "neutral")
    check("alias fear -> fearful", labels.normalise("fear") == "fearful")
    check("alias ps -> surprised", labels.normalise("ps") == "surprised")
    try:
        labels.normalise("bored")
        check("unknown label rejected", False)
    except ValueError:
        check("unknown label rejected", True)
    check("expected dataset counts sum to 1440",
          sum(labels.EXPECTED_COUNTS.values()) == labels.EXPECTED_ROWS)


def test_filenames() -> None:
    check("ravdess name", parse_ravdess_filename("03-01-05-01-01-01-07.wav")
          == ("angry", 7))
    for bad in ("03-02-05-01-01-01-01.wav", "03-01-99-01-01-01-01.wav",
                "junk.wav"):
        try:
            parse_ravdess_filename(bad)
            check(f"rejects {bad}", False)
        except ValueError:
            check(f"rejects {bad}", True)


def test_audio() -> None:
    y = tone()
    check("to_mono collapses stereo",
          to_mono(np.stack([y, y], axis=1)).ndim == 1)
    check("resample halves length",
          abs(len(resample(y, SR, 8000)) - len(y) // 2) <= 2)
    check("fit_length pads", len(fit_length(y[:1000], 4000)) == 4000)
    check("fit_length trims", len(fit_length(y, 1000)) == 1000)


def test_augment() -> None:
    y = tone()
    for preset in augment.PRESETS:
        out, applied = augment.augment(y, np.random.default_rng(1), preset)
        check(f"augment '{preset}' finite", np.isfinite(out).all())
        check(f"augment '{preset}' non-empty", len(out) > 0)
        check(f"augment '{preset}' no clipping",
              float(np.mean(np.abs(out) >= 0.999)) < 0.02)
    x = np.stack([y, y, y])
    yy = np.array([0, 1, 2])
    classes = labels.CLASSES
    xa, ya, flag = augment.expand(x, yy, classes, copies=1, seed=0)
    check("expand keeps originals first", len(xa) >= len(x))
    check("expand flags originals as 0", int(flag[: len(x)].sum()) == 0)
    check("expand produced synthetic rows", int(flag.sum()) > 0)
    check("expand keeps labels aligned", len(xa) == len(ya) == len(flag))


def test_splits() -> None:
    actors = [a for a in range(1, 25) for _ in range(60)]
    sp = splits.speaker_disjoint_split(actors, 0.17, 0.17, 42)
    sets = {k: set(v) for k, v in sp.items()}
    check("24 actors distributed", sum(len(v) for v in sets.values()) == 24)
    check("no actor overlap",
          not (sets["train"] & sets["val"]) and
          not (sets["train"] & sets["test"]) and
          not (sets["val"] & sets["test"]))
    check("train has the most actors", len(sets["train"]) >= len(sets["val"]))
    y = np.array([i % 8 for i in range(len(actors))])
    info = splits.describe(sp, np.array(actors), y, labels.CLASSES)
    check("describe reports no password-style problems",
          isinstance(info["problems"], list))


def test_features(tmp: Path = Path("/tmp")) -> None:
    x = np.stack([tone(1.0), tone(1.0, seed=2)])
    y = np.array([0, 1])
    p = tmp / "feat.npz"
    features.save_npz(p, x, y, np.array([1, 2]), np.array([0, 1]))
    blob = features.load_npz(p)
    check("npz roundtrip keys",
          set(blob) >= {"data", "lengths", "labels", "actors", "is_augmented"})
    check("npz keeps row count", len(blob["labels"]) == 2)
    check("npz lengths match", blob["lengths"].tolist() == [16000, 16000])
    row0 = features.from_flat(blob, 0)
    check("npz row reconstruction", len(row0) == 16000)


def test_metrics() -> None:
    classes = labels.CLASSES
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2])
    res = metrics.summary(y_true, y_pred, classes)
    check("accuracy computed", abs(res["accuracy"] - 5 / 6) < 1e-3,
          f"got {res['accuracy']}")
    # labels 0,1,2 are present in y_true; the other five have zero support
    check("absent classes flagged",
          sorted(res["absent_from_test"]) == sorted(
              ["fearful", "happy", "neutral", "sad", "surprised"]),
          str(res["absent_from_test"]))
    check("macro F1 lower than weighted when classes are missing",
          res["macro_f1"] <= res["weighted_f1"] + 1e-9)
    ok, reasons = metrics.gate(res, min_macro_f1=0.99)
    check("gate fails high thresholds", not ok and reasons)
    ok2, _ = metrics.gate(res, min_macro_f1=0.0)
    check("gate passes zero threshold", ok2)


def test_config(tmp: Path = Path("/tmp")) -> None:
    cfg = config.resolve(ROOT / "configs" / "ci_smoke.yaml")
    check("config loads", cfg["data"]["limit"] == 72)
    check("config has train block", "epochs" in cfg["train"])
    check("dotted get", config.get(cfg, "train.epochs") == 1)
    check("missing key default", config.get(cfg, "nope.nope", "x") == "x")
    check("coerce bool", config._coerce("true") is True)          # noqa: SLF001


def test_smoke_config_sane() -> None:
    cfg = config.resolve(ROOT / "configs" / "ci_smoke.yaml")
    limit = cfg["data"]["limit"]
    # the CI config must feed the split guard enough actors to work with
    check("ci limit covers several actors", limit and limit >= 40)


def main() -> int:
    import tempfile

    print("labels"); test_labels()
    print("filenames"); test_filenames()
    print("audio"); test_audio()
    print("augment"); test_augment()
    print("splits"); test_splits()
    print("metrics"); test_metrics()
    with tempfile.TemporaryDirectory() as td:
        print("features"); test_features(Path(td))
        print("config"); test_config(Path(td))
    print("smoke config"); test_smoke_config_sane()
    print(f"\n{'ALL TESTS PASSED' if not FAILS else str(len(FAILS)) + ' FAILED'}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())

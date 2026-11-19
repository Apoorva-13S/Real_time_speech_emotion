"""Evaluation that cannot hide a broken class.

The previous run reported weighted F1 0.5267 while two of its eight classes had
zero support. `weighted` gives those classes weight 0, so the damage was
invisible. This module always reports macro AND weighted, per-class recall for
every class (0.0 when absent, not "undefined"), and a coverage count.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def _to_index(value, classes: list[str]) -> int:
    """Accept either a class name or an integer label id."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    return classes.index(value)


def confusion_matrix(y_true, y_pred, classes: list[str]) -> np.ndarray:
    n = len(classes)
    m = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(np.asarray(y_true).tolist(), np.asarray(y_pred).tolist()):
        m[_to_index(t, classes), _to_index(p, classes)] += 1
    return m


def per_class(y_true: np.ndarray, y_pred: np.ndarray,
              classes: list[str]) -> dict[str, dict[str, float]]:
    m = confusion_matrix(y_true, y_pred, classes)
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(classes):
        tp = int(m[i, i])
        support = int(m[i].sum())
        predicted = int(m[:, i].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        out[name] = {"precision": round(precision, 4),
                     "recall": round(recall, 4),
                     "f1": round(f1, 4),
                     "support": support}
    return out


def summary(y_true: np.ndarray, y_pred: np.ndarray,
            classes: list[str]) -> dict:
    """All the numbers we commit to reporting, computed defensively."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pc = per_class(y_true, y_pred, classes)
    supports = np.array([pc[c]["support"] for c in classes], dtype=float)
    rec = np.array([pc[c]["recall"] for c in classes], dtype=float)
    f1s = np.array([pc[c]["f1"] for c in classes], dtype=float)
    n = int(len(y_true))
    acc = float(np.mean(y_true == y_pred)) if n else 0.0
    present = supports > 0
    out = {
        "n": n,
        "accuracy": round(acc, 4),
        "macro_f1": round(float(f1s[present].mean()) if present.any() else 0.0, 4),
        "weighted_f1": round(
            float((f1s * supports).sum() / supports.sum())
            if supports.sum() else 0.0, 4),
        "macro_recall": round(
            float(rec[present].mean()) if present.any() else 0.0, 4),
        "classes_in_test": int(present.sum()),
        "per_class": pc,
    }
    absent = [c for c in classes if pc[c]["support"] == 0]
    never_predicted = [c for c in classes if pc[c]["precision"] == 0.0]
    out["absent_from_test"] = absent
    out["never_predicted"] = never_predicted
    return out


def majority_baseline(y_true: np.ndarray) -> float:
    if not len(y_true):
        return 0.0
    top = Counter(y_true).most_common(1)[0][1]
    return round(top / len(y_true), 4)


def format_report(res: dict, classes: list[str]) -> str:
    lines = [
        f"samples           : {res['n']}",
        f"accuracy          : {res['accuracy']:.4f}",
        f"macro F1          : {res['macro_f1']:.4f}   <- unweighted; catches dead classes",
        f"weighted F1       : {res['weighted_f1']:.4f}",
        f"macro recall      : {res['macro_recall']:.4f}",
        f"classes in test   : {res['classes_in_test']}/{len(classes)}",
        "",
        f"{'class':12s}{'prec':>7s}{'rec':>7s}{'f1':>7s}{'support':>9s}",
    ]
    for c in classes:
        d = res["per_class"][c]
        flag = "  <-- NO TEST DATA" if d["support"] == 0 else (
            "  <-- never predicted" if d["precision"] == 0.0 else "")
        lines.append(f"{c:12s}{d['precision']:7.2f}{d['recall']:7.2f}"
                     f"{d['f1']:7.2f}{d['support']:9d}{flag}")
    if res["absent_from_test"]:
        lines += ["", f"WARNING: no test samples for {res['absent_from_test']}; "
                      f"their F1 is reported as 0.0, not hidden."]
    if res["never_predicted"]:
        lines.append(f"WARNING: model never predicted {res['never_predicted']}.")
    return "\n".join(lines)


def gate(res: dict, min_macro_f1: float = 0.0,
         min_accuracy: float = 0.0) -> tuple[bool, list[str]]:
    """CI gate. Returns (ok, reasons-it-failed)."""
    reasons = []
    if res["macro_f1"] < min_macro_f1:
        reasons.append(f"macro F1 {res['macro_f1']:.4f} < required "
                       f"{min_macro_f1:.4f}")
    if res["accuracy"] < min_accuracy:
        reasons.append(f"accuracy {res['accuracy']:.4f} < required "
                       f"{min_accuracy:.4f}")
    return (not reasons, reasons)

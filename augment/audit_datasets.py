#!/usr/bin/env python3
"""Audit the local data folder: is it complete and *in sync* with 8-class training?

Answers, with evidence, the question "is my data synced for 8-class
classification?":

  1. Inventory      -- how many files per dataset and per emotion, anywhere on disk
  2. Liveness       -- can each file actually be decoded? (LFS pointers, truncation)
  3. Label coverage -- which of the 8 classes exist at all
  4. Balance        -- per-class counts, imbalance ratio, "is any class dead?"
  5. Splits         -- builds the speaker-disjoint train/val/test split and reports
                       whether every split covers every class
  6. Sync verdict   -- PASS / WARN / FAIL per check, then one overall verdict

Usage
-----
  python augment/audit_datasets.py --scan ~/Downloads --limit 40
  python augment/audit_datasets.py --scan /path/to/datasets --json out.json
  python augment/audit_datasets.py --scan D:/data --no-decode      # fast listing
  python augment/audit_datasets.py --scan D:/data --seconds 3      # decode <=3 s/file

Exit code is 0 when the verdict is PASS or WARN, 1 when it is FAIL (CI friendly).

Only numpy + soundfile are required; librosa is optional.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio_ops as ops                                     # noqa: E402
import labels as L                                          # noqa: E402

BAR = "=" * 78
SUB = "-" * 78


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def classify_path(path: Path) -> str | None:
    """Guess the dataset from the filename grammar only (no folder assumptions).

    Order matters: TESS names such as `OAF_back_sad.wav` also look like
    CREMA-D names, because SAD is a valid CREMA emotion code. CREMA-D actor ids
    are always 4 digits, so that is required for the CREMA-D rule.
    """
    stem = path.stem
    if stem.count("-") == 6:
        return "RAVDESS"
    if "_" not in stem:
        return None
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    first, code = parts[0], parts[2].upper()
    if first.isdigit() and len(first) == 4 and code in L.CREMA_MAP:
        return "CREMA-D"
    if parts[-1].lower() in L.TESS_MAP:
        return "TESS"
    if code in L.CREMA_MAP:            # non-standard actor id, still CREMA-like
        return "CREMA-D"
    return None


def scan(root: Path) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = defaultdict(list)
    unknown = 0
    skipped_ad = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if L.is_appledouble(path):
            skipped_ad += 1
            continue
        if not L.is_audio(path):
            continue
        ds = classify_path(path)
        if ds is None:
            unknown += 1
            continue
        found[ds].append(path)
    found["_unknown_audio"] = [] if not unknown else [Path(f"<{unknown} unnamed files>")]
    found["_appledouble"] = [] if not skipped_ad else [Path(f"<{skipped_ad} files>")]
    for k in found:
        found[k] = sorted(found[k])
    return found


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def analyse(files: list[Path], dataset: str, decode: bool, seconds: float | None):
    rows, parse_errors = [], []
    for path in files:
        item = {"path": str(path), "dataset": dataset}
        try:
            emotion, speaker = L.parse(dataset, Path(path))
        except L.ParseError as exc:
            parse_errors.append(str(exc))
            continue
        item.update(emotion=emotion, speaker=speaker, readable=None,
                    dur=None, lfs=None, header=None)
        if decode:
            item["lfs"] = L.looks_like_lfs_pointer(Path(path))
            item["header"] = L.has_wav_header(Path(path))
            try:
                import soundfile as sf

                info = sf.info(str(path))
                item["readable"] = info.frames > 0 and info.samplerate > 0
                item["sr"] = info.samplerate
                item["channels"] = info.channels
                if item["readable"] and seconds:
                    y = ops.load_audio(path)[: int(seconds * ops.TARGET_SR)]
                    item["dur"] = ops.duration_s(y)
                else:
                    item["dur"] = info.frames / max(info.samplerate, 1)
            except Exception as exc:                     # noqa: BLE001
                item["readable"] = False
                item["error"] = f"{type(exc).__name__}: {exc}"[:160]
        rows.append(item)
    return rows, parse_errors


def status(pass_: bool, warn: bool = False) -> str:
    return "PASS" if pass_ else ("WARN" if warn else "FAIL")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", required=True, help="folder to search recursively")
    ap.add_argument("--no-decode", action="store_true",
                    help="skip opening files (fast; cannot detect LFS/truncation)")
    ap.add_argument("--seconds", type=float, default=None,
                    help="decode at most N seconds per file for duration stats")
    ap.add_argument("--limit", type=int, default=0,
                    help="decode only the first N files per dataset (0 = all)")
    ap.add_argument("--json", default=None, help="write the full report as JSON")
    ap.add_argument("--seconds-window", type=float, default=3.0,
                    help="training window used when judging clip length")
    args = ap.parse_args()

    root = Path(args.scan).expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 1

    print(BAR)
    print(f"DATA AUDIT  --  {root}")
    print(f"decode={not args.no_decode}  limit={args.limit or 'all'}  "
          f"window={args.seconds_window}s")
    print(BAR)

    found = scan(root)
    verdicts: list[tuple[str, str, str]] = []          # (check, status, detail)

    audio_counts = {k: len(v) for k, v in found.items() if not k.startswith("_")}
    print("\n1) INVENTORY")
    print(SUB)
    for ds in ("CREMA-D", "RAVDESS", "TESS"):
        print(f"  {ds:9s} {audio_counts.get(ds, 0):>6,} files")
    n_unknown = len(found.get("_unknown_audio", []))
    if n_unknown:
        print(f"  {'UNKNOWN':9s} {n_unknown} file group(s) named like nothing known")
    if found.get("_appledouble"):
        print(f"  note: {found['_appledouble'][0].name}")
    present = [ds for ds in ("CREMA-D", "RAVDESS", "TESS") if audio_counts.get(ds)]
    print(f"\n  datasets detected on disk: {present or 'NONE'}")
    verdicts.append((
        "all three datasets present",
        status(len(present) == 3, len(present) >= 1),
        f"found {len(present)}/3",
    ))

    # ------------------------------------------------------------------ #
    rows: list[dict] = []
    parse_errors: list[str] = []
    print("\n2) DECODING / INTEGRITY")
    print(SUB)
    for ds in present:
        files = found[ds]
        if args.limit:
            files = files[: args.limit]
        r, pe = analyse(files, ds, decode=not args.no_decode, seconds=args.seconds)
        rows.extend(r)
        parse_errors.extend(pe)
        unreadable = [x for x in r if x["readable"] is False]
        lfs = [x for x in r if x.get("lfs")]
        bad_header = [x for x in r if x.get("header") is False]
        durs = [x["dur"] for x in r if x.get("dur")]
        line = f"  {ds:9s} checked {len(r):>5,}/{len(files):<5,}"
        if not args.no_decode:
            line += (f" unreadable={len(unreadable):<5,} lfs_pointers={len(lfs):<5,}"
                     f" bad_header={len(bad_header):<4,}")
            if durs:
                line += (f" dur min/med/max="
                         f"{min(durs):.2f}/{np.median(durs):.2f}/{max(durs):.2f}s")
        print(line)
        if unreadable[:3]:
            for x in unreadable[:3]:
                print(f"      ! {Path(x['path']).name}: {x.get('error', 'unreadable')}")
        if not args.no_decode:
            rate = len(unreadable) / max(len(r), 1)
            verdicts.append((
                f"{ds}: readable audio",
                status(rate == 0, rate <= 0.01),
                f"{len(unreadable)}/{len(r)} failed ({100*rate:.1f}%)",
            ))
            if lfs:
                verdicts.append((f"{ds}: no Git-LFS pointers", "FAIL",
                                 f"{len(lfs)} pointers (not real audio)"))
        if pe[:2]:
            print(f"      parse errors: {len(pe)} e.g. {pe[0]}")

    # ------------------------------------------------------------------ #
    print("\n3) LABEL COVERAGE (8-class target)")
    print(SUB)
    per_ds: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        per_ds[r["dataset"]][r["emotion"]] += 1
    total = Counter()
    for ds, counter in per_ds.items():
        total.update(counter)

    header = "  " + "dataset".ljust(10) + "".join(c[:8].rjust(10) for c in L.CLASSES)
    print(header)
    for ds in present:
        counter = per_ds[ds]
        print("  " + ds.ljust(10) +
              "".join(str(counter.get(c, 0)).rjust(10) for c in L.CLASSES))
    print("  " + "TOTAL".ljust(10) +
          "".join(str(total.get(c, 0)).rjust(10) for c in L.CLASSES))

    missing = [c for c in L.CLASSES if total.get(c, 0) == 0]
    print(f"\n  classes with data      : "
          f"{[c for c in L.CLASSES if total.get(c, 0)]}")
    print(f"  classes MISSING        : {missing or 'none'}")
    verdicts.append((
        "all 8 classes have data",
        status(not missing),
        f"missing {missing}" if missing else "8/8 present",
    ))

    # Which datasets can teach which class (source of generalisation)
    print("\n  per-class provenance (which corpus supplies this class):")
    for c in L.CLASSES:
        srcs = [ds for ds in present if per_ds[ds].get(c, 0)]
        flag = "  <-- single source" if len(srcs) == 1 else ""
        print(f"    {c:10s} {str(srcs) if srcs else 'NONE':50s}{flag}")
    single = [c for c in L.CLASSES
              if len([ds for ds in present if per_ds[ds].get(c, 0)]) == 1]
    verdicts.append((
        "classes supported by >1 corpus",
        status(not single, True),
        f"single-corpus classes: {single}" if single else "all classes multi-source",
    ))

    # ------------------------------------------------------------------ #
    print("\n4) BALANCE")
    print(SUB)
    counts = np.array([total.get(c, 0) for c in L.CLASSES if total.get(c, 0)])
    if len(counts):
        ratio = counts.max() / max(counts.min(), 1)
        med = float(np.median(counts))
        thin = [c for c in L.CLASSES
                if 0 < total.get(c, 0) < 0.25 * med]
        print(f"  largest/smallest class ratio : {ratio:.1f}x")
        print(f"  minority share               : "
              f"{100*counts.min()/counts.sum():.2f}% of all data "
              f"({counts.min()} clips)")
        print(f"  under-represented (<25% of median): {thin or 'none'}")
        print(f"  majority-class baseline acc.  : "
              f"{100*counts.max()/counts.sum():.2f}%  (beat this, then beat 12.5%)")
        verdicts.append((
            "class balance",
            status(ratio <= 3, ratio <= 6),
            f"{ratio:.1f}x imbalance",
        ))
        if thin:
            verdicts.append((
                "no under-represented classes",
                "WARN",
                f"{thin} below 25% of the median class size",
            ))
    else:
        print("  no labelled files found -- nothing to balance")
        verdicts.append(("class balance", "FAIL", "no data"))

    # ------------------------------------------------------------------ #
    print("\n5) SPLIT SYNC (speaker-disjoint, TESS held out)")
    print(SUB)
    try:
        from make_splits import build_split, describe

        table = [{"path": r["path"], "dataset": r["dataset"],
                  "emotion": r["emotion"], "speaker": r["speaker"]} for r in rows]
        split = build_split(table, val_frac=0.20, seed=42)
        describe(split, L.CLASSES)
        for name in ("train", "val", "test"):
            present_cls = {c for c in L.CLASSES
                           if any(r["emotion"] == c for r in split[name])}
            gaps = [c for c in L.CLASSES if c not in present_cls]
            verdicts.append((
                f"split '{name}' covers all 8 classes",
                status(not gaps, len(gaps) <= 2),
                f"missing {gaps}" if gaps else "8/8",
            ))
    except Exception as exc:                              # noqa: BLE001
        print(f"  could not build the split: {type(exc).__name__}: {exc}")
        verdicts.append(("split sync", "WARN", "split not evaluated"))

    # ------------------------------------------------------------------ #
    print("\n6) SYNCHRONISATION VERDICT")
    print(SUB)
    width = max(len(c) for c, _, _ in verdicts) + 2
    for check, st, detail in verdicts:
        mark = {"PASS": "[ ok ]", "WARN": "[warn]", "FAIL": "[FAIL]"}[st]
        print(f"  {mark} {check.ljust(width)} {detail}")
    overall = ("FAIL" if any(s == "FAIL" for _, s, _ in verdicts)
               else "WARN" if any(s == "WARN" for _, s, _ in verdicts) else "PASS")
    print(SUB)
    print(f"  OVERALL: {overall}")
    print(explain(overall, present, missing, total, args))
    print(BAR)

    if args.json:
        Path(args.json).write_text(json.dumps({
            "root": str(root),
            "counts": {ds: dict(per_ds[ds]) for ds in per_ds},
            "totals": dict(total),
            "missing_classes": missing,
            "verdicts": [{"check": c, "status": s, "detail": d}
                         for c, s, d in verdicts],
            "overall": overall,
            "files": rows,
        }, indent=2))
        print(f"report written to {args.json}")

    return 0 if overall in ("PASS", "WARN") else 1


def explain(overall: str, present: list[str], missing: list[str],
            total: Counter, args) -> str:
    lines = []
    if overall == "PASS":
        lines.append("  Your local data can train an 8-class model as-is.")
    if missing:
        lines.append(f"  Cannot train these classes: {missing}.")
        lines.append("  Either obtain the corpus that contains them, merge them "
                     "away (calm -> neutral),")
        lines.append("  or use an N-class head that matches reality.")
    if "CREMA-D" in present and total.get("calm", 0) == 0:
        lines.append("  REMINDER: CREMA-D has no calm/surprised; RAVDESS supplies "
                     "both, TESS has no calm.")
    if args.no_decode:
        lines.append("  Decoding was skipped: re-run without --no-decode to detect "
                     "LFS pointers/truncation.")
    if overall == "FAIL":
        lines.append("  Fix the FAIL rows above before training; each one silently "
                     "distorts accuracy.")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

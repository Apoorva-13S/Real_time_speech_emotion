"""One command surface for everything.

    python -m ser.cli prepare   --config configs/local_train.yaml
    python -m ser.cli train     --config configs/colab_train.yaml
    python -m ser.cli evaluate  --model runs/local/best
    python -m ser.cli predict   --audio clip.wav
    python -m ser.cli audit     --config configs/local_train.yaml
    python -m ser.cli export    --model runs/local/best --out ser_ravdess.zip
    python -m ser.cli serve
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def _apply_sets(cfg: dict, sets: list[str]) -> dict:
    from . import config as cfgmod

    for item in sets:
        key, _, val = item.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = cfgmod._coerce(val)
    return cfg


def _cfg(args) -> dict:
    from . import config as cfgmod

    return _apply_sets(cfgmod.resolve(getattr(args, "config", None)),
                       getattr(args, "set", []) or [])


# --------------------------------------------------------------------------- #
def cmd_prepare(args) -> int:
    from .prepare import prepare

    p = prepare(_cfg(args))
    print("\n" + p.summary())
    return 0


def cmd_train(args) -> int:
    from .train import train

    result = train(_cfg(args))
    print(f"\nbest epoch {result['best_epoch']}  "
          f"test macro F1 {result['test']['macro_f1']:.4f}  "
          f"test accuracy {result['test']['accuracy']:.4f}")
    return 0


def cmd_evaluate(args) -> int:
    from . import metrics
    from .evaluate import evaluate

    res = evaluate(args.model, _cfg(args))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))
        print(f"wrote {args.json}")
    if args.min_macro_f1 is not None:
        ok, reasons = metrics.gate(res, min_macro_f1=args.min_macro_f1)
        for r in reasons:
            print(f"GATE FAILED: {r}")
        return 0 if ok else 1
    return 0


def cmd_predict(args) -> int:
    from .predict import main as predict_main

    return predict_main(["--audio", args.audio,
                         *(["--model", args.model] if args.model else []),
                         *(["--json", args.json] if args.json else [])])


def cmd_audit(args) -> int:
    """Report the dataset facts this pipeline depends on, from a live load."""
    import numpy as np

    from .data import load_any
    from .labels import CLASSES, EXPECTED_ACTORS, EXPECTED_ROWS
    from .splits import describe, speaker_disjoint_split

    cfg = _cfg(args)
    limit = cfg.get("data", {}).get("limit")
    clips = load_any(cfg.get("data", {}).get("source", "xbgoose/ravdess"),
                     limit=limit)
    actors = sorted({c.actor for c in clips})
    counts = {c: sum(1 for k in clips if k.emotion == c) for c in CLASSES}
    durations = [c.duration_s for c in clips]

    print(f"clips loaded     : {len(clips)}")
    print(f"actors           : {len(actors)} {actors[:6]}{'...' if len(actors) > 6 else ''}")
    print(f"duration min/med/max: {min(durations):.2f} / "
          f"{np.median(durations):.2f} / {max(durations):.2f} s")
    print("per class        :")
    for c in CLASSES:
        print(f"  {c:10s} {counts[c]:6d}")
    problems = []
    if limit is None:
        if len(clips) != EXPECTED_ROWS:
            problems.append(f"{len(clips)} rows != expected {EXPECTED_ROWS}")
        if len(actors) != EXPECTED_ACTORS:
            problems.append(f"{len(actors)} actors != expected {EXPECTED_ACTORS}")
        missing = [c for c, n in counts.items() if n == 0]
        if missing:
            problems.append(f"no clips for {missing}")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        return 1
    sp = speaker_disjoint_split([c.actor for c in clips])
    print(f"\nsplit preview: train={len(sp['train'])} actors, "
          f"val={len(sp['val'])}, test={len(sp['test'])}")
    print(json.dumps(describe(sp, np.asarray([c.actor for c in clips]),
                              np.asarray([CLASSES.index(c.emotion) for c in clips]),
                              CLASSES), indent=2)[:1200])
    print("\nAUDIT OK")
    return 0


def cmd_export(args) -> int:
    """Zip a checkpoint folder plus labels.json/metrics.json for release."""
    model = Path(args.model)
    if not model.is_dir():
        print(f"ERROR: {model} is not a folder")
        return 1
    missing = [f for f in ("config.json",) if not (model / f).is_file()]
    weights = list(model.glob("*.safetensors")) + list(model.glob("pytorch_model.bin"))
    if missing or not weights:
        print(f"ERROR: {model} is not a model folder "
              f"(missing {missing + ([] if weights else ['weights'])})")
        return 1
    stage = Path(args.stage or "/tmp/ser_bundle") / "model"
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(model, stage)
    for extra in ("labels.json", "metrics.json"):
        src = model.parent / extra
        if src.is_file():
            shutil.copy2(src, stage.parent / extra)
    out = Path(args.out or f"{model.name}.zip")
    shutil.make_archive(str(out.with_suffix("")), "zip", root_dir=stage.parent)
    size = out.stat().st_size / 1e6
    print(f"wrote {out} ({size:.0f} MB)")
    print("Next: publish it (see process.txt, 'Publishing the model') or "
          "unzip next to the app and run:\n"
          f"  python app/server.py --model {stage}")
    return 0


def cmd_serve(args) -> int:
    """Delegate to app/server.py (it is not an installed package)."""
    import runpy

    script = Path(__file__).resolve().parents[2] / "app" / "server.py"
    if not script.is_file():
        print(f"ERROR: web app not found at {script}")
        return 1
    sys.argv = [str(script)] + list(getattr(args, "serve_args", []) or [])
    runpy.run_path(str(script), run_name="__main__")
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ser", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def add(name, func, help_text, model=False, json_out=False):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", default=None)
        p.add_argument("--set", nargs="*", default=[],
                       help="overrides, e.g. train.epochs=1 data.limit=72")
        if model:
            p.add_argument("--model", default="runs/local/best")
        if json_out:
            p.add_argument("--json", default=None)
        p.set_defaults(func=func)
        return p

    add("prepare", cmd_prepare, "load, split, augment and cache the data")
    add("train", cmd_train, "fine-tune wav2vec2")
    ev = add("evaluate", cmd_evaluate, "evaluate a checkpoint on held-out actors",
             model=True, json_out=True)
    ev.add_argument("--min-macro-f1", type=float, default=None)
    pr = add("predict", cmd_predict, "predict one audio file", json_out=True)
    pr.add_argument("--audio", required=True)
    pr.add_argument("--model", default=None)
    add("audit", cmd_audit, "check the dataset against expectations")
    ex = add("export", cmd_export, "zip a checkpoint for release", model=True)
    ex.add_argument("--out", default=None)
    ex.add_argument("--stage", default=None)
    sv = add("serve", cmd_serve, "run the web demo")
    sv.add_argument("serve_args", nargs="*", default=[])
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

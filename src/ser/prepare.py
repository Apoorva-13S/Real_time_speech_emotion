"""Data preparation: load -> split -> augment -> cache. No torch required.

Kept separate from training so that:
  * it can be run and verified on any machine (numpy + soundfile only)
  * a Colab session can re-prepare data after a disconnect without re-importing
    torch, and
  * CI can exercise the whole data path in seconds.

    python -m ser.prepare --config configs/local_train.yaml
    python -m ser.prepare --config configs/local_train.yaml --dry-run
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config as cfgmod
from . import splits
from .data import load_any, to_arrays
from .features import save_npz
from .labels import names_for


@dataclass
class Prepared:
    classes: list[str]
    window_s: float
    actor_split: dict[str, list[int]]
    split_info: dict
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    synthetic_fraction: float
    n_clips_loaded: int
    out_dir: Path

    def summary(self) -> str:
        lines = [f"loaded clips     : {self.n_clips_loaded}",
                 f"window           : {self.window_s}s",
                 f"train / val / test: {len(self.y_train)} / {len(self.y_val)} "
                 f"/ {len(self.y_test)}",
                 f"synthetic train  : {100 * self.synthetic_fraction:.1f}%"]
        for name in ("train", "val", "test"):
            per = self.split_info["splits"][name]["per_class"]
            lines.append(f"  {name:5s} actors={self.split_info['splits'][name]['actors']}"
                         f" per_class={per}")
        if self.split_info.get("problems"):
            lines.append("problems:")
            lines += [f"  ! {p}" for p in self.split_info["problems"]]
        return "\n".join(lines)


def prepare(cfg: dict, verbose: bool = True) -> Prepared:
    seed = int(cfgmod.get(cfg, "run.seed", 42))
    out_root = Path(cfgmod.get(cfg, "run.output_dir", "runs/local"))
    (out_root / "data").mkdir(parents=True, exist_ok=True)

    classes = names_for(int(cfgmod.get(cfg, "model.n_classes", 8)))
    window_s = float(cfgmod.get(cfg, "data.window_s", 4.0))
    source = cfgmod.get(cfg, "data.source", "xbgoose/ravdess")

    clips = load_any(source, limit=cfgmod.get(cfg, "data.limit"), verbose=verbose)
    if not clips:
        raise RuntimeError(
            f"no clips loaded from {source!r}. If this is a folder, check it "
            f"contains RAVDESS wav files (03-01-05-01-01-01-01.wav). If it is a "
            f"Hub id, check your internet connection.")
    x, y = to_arrays(clips, classes, window_s)
    actors = np.asarray([c.actor for c in clips], dtype=np.int32)
    if verbose:
        print(f"loaded {len(clips)} clips  shape={x.shape}  "
              f"per_class={np.bincount(y, minlength=len(classes)).tolist()}")

    # ---- split (speaker-disjoint, reuse if a file already exists) -------- #
    sp_path = Path(cfgmod.get(cfg, "data.splits_file",
                              str(out_root / "data/splits.json")))
    if sp_path.is_file():
        actor_split = splits.load(sp_path)
        present = {a for a in actors.tolist()}
        wanted = {a for v in actor_split.values() for a in v}
        if not wanted.issubset(present):
            if verbose:
                print(f"existing split {sp_path} does not match this data "
                      f"(missing actors {sorted(wanted - present)[:5]}); "
                      f"rebuilding")
            actor_split = splits.speaker_disjoint_split(
                actors.tolist(),
                float(cfgmod.get(cfg, "data.val_frac", 0.17)),
                float(cfgmod.get(cfg, "data.test_frac", 0.17)), seed)
            splits.save(actor_split, sp_path)
        elif verbose:
            print(f"using existing split: {sp_path}")
    else:
        actor_split = splits.speaker_disjoint_split(
            actors.tolist(),
            float(cfgmod.get(cfg, "data.val_frac", 0.17)),
            float(cfgmod.get(cfg, "data.test_frac", 0.17)), seed)
        splits.save(actor_split, sp_path)
        if verbose:
            print(f"wrote split: {sp_path}")

    info = splits.describe(actor_split, actors, y, classes)
    (out_root / "data" / "split_report.json").write_text(
        json.dumps({"classes": classes, "split": actor_split, "info": info},
                   indent=2))
    if verbose:
        for p in info["problems"]:
            print(f"  WARNING: {p}")

    idx = {name: splits.indices_for(actors, actor_split[name])
           for name in ("train", "val", "test")}
    x_tr, y_tr = x[idx["train"]], y[idx["train"]]
    x_va, y_va = x[idx["val"]], y[idx["val"]]
    x_te, y_te = x[idx["test"]], y[idx["test"]]
    if not len(y_tr):
        raise RuntimeError(
            "the training split is empty. This happens with a very small "
            "data.limit: all loaded clips landed on the held-out actors. "
            "Raise data.limit or lower data.val_frac/data.test_frac.")

    # ---- augmentation: TRAIN ONLY --------------------------------------- #
    copies = int(cfgmod.get(cfg, "augment.copies", 0))
    synth = 0.0
    if copies > 0 and len(y_tr):
        from .augment import expand

        t0 = time.time()
        boot = int(cfgmod.get(cfg, "augment.bootstrap", 6))
        use_x, use_y = x_tr, y_tr
        if len(y_tr) < boot:                # tiny data (CI): duplicate repeats
            use_x, use_y = np.repeat(x_tr, 2, axis=0), np.repeat(y_tr, 2)
        x_tr, y_tr, flag = expand(
            use_x, use_y, classes, copies=copies,
            preset=cfgmod.get(cfg, "augment.preset", "full"), seed=seed,
            scale=float(cfgmod.get(cfg, "augment.scale", 1.0)))
        synth = float(flag.mean())
        if verbose:
            print(f"augmented train: {len(y_tr)} clips "
                  f"({100 * synth:.0f}% synthetic) in {time.time() - t0:.1f}s")

    save_npz(out_root / "data" / "train.npz", x_tr, y_tr,
             np.zeros(len(y_tr), dtype=np.int32))
    save_npz(out_root / "data" / "val.npz", x_va, y_va,
             np.asarray(actor_split["val"], dtype=np.int32))
    save_npz(out_root / "data" / "test.npz", x_te, y_te,
             np.asarray(actor_split["test"], dtype=np.int32))
    if verbose:
        print(f"cached npz files under {out_root / 'data'}")

    return Prepared(classes=classes, window_s=window_s, actor_split=actor_split,
                    split_info=info, x_train=x_tr, y_train=y_tr,
                    x_val=x_va, y_val=y_va, x_test=x_te, y_test=y_te,
                    synthetic_fraction=synth, n_clips_loaded=len(clips),
                    out_dir=out_root)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Prepare data (no training)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true",
                    help="prepare and validate, then stop before training")
    args = ap.parse_args(argv)

    cfg = cfgmod.resolve(args.config)
    for item in args.set:
        key, _, val = item.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = cfgmod._coerce(val)               # noqa: SLF001

    p = prepare(cfg)
    print("\n" + p.summary())
    if args.dry_run:
        print("\ndry run: data preparation only, nothing was trained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

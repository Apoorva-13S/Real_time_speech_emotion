"""Load a checkpoint once, predict on any audio file or waveform."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .data import SAMPLE_RATE, resample, to_mono

DEFAULT_LOCAL = "runs/local/best"
FALLBACK_HF = "Dpngtm/wav2vec2-emotion-recognition"


class EmotionPredictor:
    """Wraps a waveform classifier with the project's label convention.

    Falls back to a public HF model only if no local checkpoint exists, and says
    so loudly -- the two models have different label sets, so silently swapping
    one for the other is how a demo ends up mislabelled.
    """

    def __init__(self, model_dir: str | Path | None = None,
                 device: str = "auto", max_seconds: float = 4.0,
                 verbose: bool = True):
        from transformers import (AutoFeatureExtractor,
                                  AutoModelForAudioClassification)

        self.device = torch.device(
            "cuda" if (device in ("auto", "cuda") and torch.cuda.is_available())
            else "cpu")
        self.max_seconds = max_seconds
        self.fallback = False
        self.meta: dict = {}

        path = Path(model_dir) if model_dir else None
        if path is None or not path.is_dir():
            looks_local = path is not None and (
                "/" not in str(path) and "\\" not in str(path))
            if looks_local:
                raise FileNotFoundError(
                    f"no checkpoint at {path}. Train first "
                    f"(python -m ser.train --config configs/local_train.yaml) "
                    f"or pass --model with a valid folder.")
            if verbose:
                print(f"WARNING: no local checkpoint "
                      f"{'(default: ' + DEFAULT_LOCAL + ')' if path is None else ''}"
                      f" -> using public model {FALLBACK_HF}.\n"
                      f"         Its labels may differ from this project's 8 "
                      f"classes. Do not present this as the trained model.")
            self.fallback = True
            self.model_dir = FALLBACK_HF
        else:
            self.model_dir = str(path)
            meta_file = Path(self.model_dir).parent / "labels.json"
            if meta_file.is_file():
                self.meta = json.loads(meta_file.read_text())

        self.model = AutoModelForAudioClassification.from_pretrained(
            self.model_dir).to(self.device).eval()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.model_dir)
        self.sr = int(getattr(self.feature_extractor, "sampling_rate", SAMPLE_RATE))
        labels = getattr(self.model.config, "id2label", None) or {}
        self.classes = [labels[i] for i in sorted(labels)] if labels else []
        if self.meta.get("window_s"):
            self.max_seconds = float(self.meta["window_s"])
        if verbose:
            print(f"model={self.model_dir}  device={self.device}  "
                  f"labels={self.classes}  window={self.max_seconds}s")

    # ------------------------------------------------------------------ #
    def _prepare(self, audio: np.ndarray, sr: int) -> np.ndarray:
        y = resample(to_mono(audio), sr, self.sr)
        n = int(self.max_seconds * self.sr)
        if len(y) > n:
            # keep the loudest window rather than the first: speech often
            # starts with silence, and the first 4 s may be entirely room tone
            hop = max(n // 2, 1)
            best, best_e = None, -1.0
            for start in range(0, len(y) - n + 1, hop):
                seg = y[start:start + n]
                e = float(np.sum(np.square(seg, dtype=np.float64)))
                if e > best_e:
                    best, best_e = seg, e
            y = best if best is not None else y[:n]
        elif len(y) < n:
            y = np.pad(y, (0, n - len(y)))
        y = y - float(np.mean(y))
        std = float(np.std(y))
        return ((y / (std + 1e-7)).astype(np.float32),)

    def predict(self, audio: np.ndarray, sr: int = SAMPLE_RATE,
                top_k: int | None = None) -> list[dict]:
        (y,) = self._prepare(audio, sr)
        with torch.no_grad():
            x = torch.from_numpy(y).unsqueeze(0).to(self.device)
            logits = self.model(input_values=x).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        order = np.argsort(-probs)[: (top_k or len(probs))]
        return [{"label": self.classes[i] if self.classes else str(i),
                 "score": float(probs[i])} for i in order]

    def predict_file(self, path: str | Path, top_k: int | None = None) -> list[dict]:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        return self.predict(data, sr, top_k=top_k)


def format_prediction(preds: list[dict], top_k: int = 3) -> str:
    return " | ".join(f"{p['label']} {p['score'] * 100:.1f}%"
                      for p in preds[:top_k])


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Predict emotion for an audio file")
    ap.add_argument("--audio", required=True, help="path to a WAV/FLAC file")
    ap.add_argument("--model", default=None,
                    help=f"checkpoint folder (default {DEFAULT_LOCAL})")
    ap.add_argument("--top-k", type=int, default=None, help="show N labels (default all)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    path = Path(args.audio).expanduser()
    if not path.is_file():
        print(f"ERROR: file not found: {path}")
        print("Tip: drag-and-drop paths from Windows/macOS often carry quotes "
              "or a leading space; remove them, or quote the path.")
        return 2

    pred = EmotionPredictor(args.model)
    preds = pred.predict_file(path, top_k=args.top_k)
    print(f"\n{path.name}: {format_prediction(preds, 3)}")
    print("\nall classes:")
    for p in preds:
        bar = "#" * int(round(p["score"] * 40))
        print(f"  {p['label']:10s} {p['score'] * 100:5.1f}%  {bar}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"file": str(path), "model": pred.model_dir, "predictions": preds},
            indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

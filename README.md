# Real-Time Speech Emotion Recognition

8-class speech emotion recognition with a fine-tuned Wav2Vec2 model, trained on
[`xbgoose/ravdess`](https://huggingface.co/datasets/xbgoose/ravdess) (1,440 clips,
24 actors, 8 emotions, one `train` split).

**New here? Read [`process.txt`](process.txt).** It walks from installing Python to a
running web demo, with a symptom → fix table for every error you are likely to hit.

## What is in the box

| Path | Purpose |
|---|---|
| `src/ser/` | data loading, speaker-disjoint splits, augmentation, training, evaluation, prediction |
| `app/` | web demo (standard library only) — upload a clip, get 8 probabilities |
| `configs/` | `local_train.yaml`, `colab_train.yaml`, `ci_smoke.yaml` |
| `notebooks/colab_train.ipynb` | step-by-step Colab flow (free T4) |
| `.github/workflows/` | CI (tests, data path, app, dataset contract) + model release |
| `tests/` | unit tests, offline fixture, dataset contract check |
| `instructions.txt` | technical audit of the original code: every bug with evidence |
| `augment/FINDINGS.md` | data-sync verdict and augmentation rationale |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[train]"

ser audit    --config configs/local_train.yaml    # verify the dataset
ser prepare  --config configs/local_train.yaml    # split + augment + cache
ser train    --config configs/local_train.yaml    # fine-tune
ser evaluate --model runs/local/best              # honest score on unseen actors

python app/server.py --model runs/local/best      # demo at http://localhost:8000
```

Training on a laptop CPU works but is slow; `notebooks/colab_train.ipynb` does the
same in ~15 minutes on a free T4.

## How the data is split (and why it matters)

The dataset ships **one** split named `train` — the "8" is emotions, not splits.
Splitting by file would put the same voice in training and testing, inflating every
score. This repo splits the **24 actors**: 16 train / 4 validation / 4 test, and
refuses to continue if an actor appears in two splits.

## Evaluation you can trust

`ser evaluate` reports accuracy, **macro F1**, weighted F1, per-class recall and a
confusion matrix, plus the majority-class baseline. Classes with no test data are
marked `NO TEST DATA` instead of being averaged away — the previous version of this
repo reported weighted F1 0.5267 while two of its eight classes had zero test clips.

## Augmentation

Applied to the training split only: speed perturbation, pink noise, synthetic room
reverb, telephone band-limiting, gain, time shift and masking. Rare classes get more
copies. All numpy/soundfile — no extra services, no accounts. Validation and test data
are never augmented.

## Status

Replaces the original scripts (`live_emotion.py`, `tes_audio.py`, the 107-cell
notebook), which loaded a third-party model and documented a run that had silently
dropped 84% of its training data. Those files are kept for reference; see
`instructions.txt` for the evidence.

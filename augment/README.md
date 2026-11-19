# Data audit + augmentation

This folder answers two questions with evidence instead of guesswork:

1. **Is my data complete and in sync for 8-class training?**
2. **How do I augment it, for free, without breaking the labels?**

Everything here runs on CPU, needs only `numpy` + `soundfile`
(`librosa` is optional and used when present), and is safe to run before you
spend a single GPU minute.

```
augment/
  labels.py               one shared definition of every dataset's filename
                          grammar + label map (the "ps" bug lived here)
  audio_ops.py            numpy-only DSP: gain, noise, reverb, speed, pitch,
                          band-limit, time-shift, masking, within-class mixup
  audit_datasets.py       #1  scan a folder, decode-check, coverage, balance,
                          split coverage, PASS/WARN/FAIL verdict
  make_splits.py          #2  speaker-disjoint train/val/test (2 profiles)
  augment_audio.py        #3  offline augmentation + manifests
  benchmark_augmentation.py   #4  measure what augmentation does; listen
  tests/make_fake_data.py     synthetic corpus for smoke tests
  tests/test_audio_ops.py     43 self-checks, no pytest needed
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install numpy soundfile            # required
pip install librosa                    # optional, better resample/pitch
```

## 1. Audit your data (do this first)

```bash
python augment/audit_datasets.py --scan "/path/to/your/data"
```

Point it at the folder that **contains** the extracted corpora. It works by
filename grammar, not folder names, so `ravdess.zip` extracted anywhere, or a
`CREMA-D/AudioWAV` folder, or a TESS folder, are all found. It also detects a
`ravdess.zip` that is still zipped only as "no audio found" — unzip it first.

What it prints:

| section | what it tells you |
|---|---|
| 1 Inventory | files per dataset; flags files that match no known grammar |
| 2 Integrity | per-dataset count of **undecodable** files, **Git-LFS pointers**, bad headers, duration min/median/max |
| 3 Label coverage | a dataset x 8-class table, which classes are **missing**, and which classes come from only **one corpus** |
| 4 Balance | imbalance ratio, minority share, under-represented classes, majority-class baseline |
| 5 Split sync | builds a speaker-disjoint split and shows whether every split truly contains all 8 classes |
| 6 Verdict | PASS / WARN / FAIL per check + overall, with a one-line explanation |

Exit code is `1` on FAIL, so it can gate CI.

Useful flags: `--no-decode` (fast listing), `--limit 40` (sample), `--json report.json`.

## 2. Build speaker-disjoint splits

```bash
python augment/make_splits.py --scan "/path/to/data" --profile 3way --out data
python augment/make_splits.py --scan "/path/to/data" --profile holdout --out data
```

| profile | train/val | test | when to use |
|---|---|---|---|
| `3way` | all three corpora split **by speaker** | a speaker-disjoint slice of each corpus **including TESS** | when you need all 8 classes evaluated. TESS speakers (OAF/YAF) land in train, so TESS clips appear in every split |
| `holdout` | CREMA-D + RAVDESS, by speaker | **TESS only, untouched** | when you need an external number — but note `calm` does not exist in TESS, so it is never tested |

Splitting is by speaker, never by file. Otherwise the same actor's voice sits
in train and test and accuracy measures voice identity, not emotion.

`--merge-calm` folds `calm` into `neutral` and produces 7 classes, which is
what the third-party model in `live_emotion.py` already does.

## 3. Augment the training set

```bash
# minority classes first (calm is 25x rarer than angry)
python augment/augment_audio.py --splits data/splits.json --out data/aug \
       --balance --max-copies 6 --preset minority --window 3

# or a general mild pass over everything
python augment/augment_audio.py --splits data/splits.json --out data/aug \
       --copies 3 --preset full
```

Presets (each transform fires with its own probability, `--chance` scales them):

| preset | transforms | rationale |
|---|---|---|
| `light` | gain, pink noise | safe baseline |
| `speed` | ±8 % speed perturbation | the single most effective cheap transform in speech recipes |
| `phone` | 300–3400 Hz band-limit + gain + noise | matches the cheap mic in the live demo |
| `reverb` | synthetic synthetic-RIR reverb | trains robustness to room change |
| `minority` | speed, noise, gain, shift, pitch | aimed at `calm` / `surprised` |
| `full` | all eight | random subset per copy |

Output: `data/aug/wav/<dataset>/<emotion>/<name>__augNN.wav`,
`data/aug/aug_manifest.csv` (what was applied to what) and
`data/aug/train.csv` (originals + augmentations, with an `is_augmented` column).

**Rules that matter more than the transform list:**
- augment **train only** — never val/test, or your metric measures the augmentation
- never change class balance across splits; augment inside the split
- listen to the output; if speech is unintelligible the label is a lie
- cap copies per clip (`--max-copies`); 20 copies of one clip teaches memorisation

## 4. Check the augmentation before trusting it

```bash
python augment/benchmark_augmentation.py --audio sample.wav
python augment/benchmark_augmentation.py --scan /path/to/data --n 3 --out /tmp/aug_demo
```

Prints duration, level, clipping and spectral centroid per preset, writes
before/after WAVs, and flags any invalid output.

## 5. Self-checks

```bash
python augment/tests/test_audio_ops.py              # 43 invariants, ~1 s
python augment/tests/make_fake_data.py --out /tmp/fake --quick
python augment/audit_datasets.py --scan /tmp/fake    # should end with FAIL,
                                                    # because the fixtures
                                                    # contain LFS pointers
```

## On-the-fly augmentation (no extra disk)

```python
import numpy as np
from augment.audio_ops import fit_window
from augment.augment_audio import apply_chain

rng = np.random.default_rng(seed)
y_aug, applied = apply_chain(y_train, sr=16000, rng=rng, preset="full")
```

If you switch to a `datasets` + `Trainer` pipeline instead of the npz cache:

```python
pip install audiomentations          # free, BSD/MIT, offline, no account
```
`audiomentations` is the closest "one import does it" library:
`AddGaussianNoise`, `PitchShift`, `TimeStretch`, `RoomSimulator`,
`SevenBandParametricEQ`, `LoudnessNormalization`, `AddBackgroundNoise`
(the last one needs a noise corpus — see the docs file for free options).
The stack here covers the same ground with zero extra dependencies, which is
why it is the default.

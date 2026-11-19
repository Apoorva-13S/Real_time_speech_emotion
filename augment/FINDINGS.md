# Data audit findings, sync verdict, and augmentation plan

Status: written without access to the actual dataset (Google Drive is not
reachable from the analysis environment, and the shared link is a single
429 MB file rather than a folder). Everything below is either **known from the
committed notebook outputs** (marked *verified*) or a **hypothesis the audit
tool will confirm or refute in one run** (marked *to verify*).

Run this first, it takes under a minute and settles the open questions:

```bash
python augment/audit_datasets.py --scan "/path/to/data" --json audit.json
```

---

## 1. What the shared link actually is

The link resolves to:

```
https://drive.usercontent.google.com/download?id=1DZx2dSNv_7GCJgAsrDyhG4qLMfJ5Gv3K
ravdess.zip  (429 MB)  "too large for Google to scan for viruses"
```

So it is **not** a data folder — it is the RAVDESS archive, the same one the
notebook reads from
`/content/drive/MyDrive/SpeechEmotionProject/ravdess.zip`.

Consequences, in order of importance:

1. **CREMA-D and TESS are not in this link.** They are 7,442 and 2,800 files
   respectively (*verified*: notebook cells 13-15 and 20-21). Whatever sync you
   do, you still need those two corpora from other sources.
2. **429 MB is roughly 10x the expected size for 1,440 RAVDESS WAVs.** The
   notebook measured 2,880 WAV entries for 1,440 unique filenames (*verified*,
   cells 26-32): every file is stored twice. Expected size of a clean copy is
   ~45-50 MB. The extra size is most likely macOS metadata (`__MACOSX/`,
   `._*` AppleDouble sidecars) added when zipping from a Mac. *To verify:*
   `unzip -l ravdess.zip | head -30` and count entries starting with `__MACOSX`.
3. **A zip is not a synced dataset.** It is an archive that is re-extracted
   into `/content/ravdess_extracted` on every Colab run, which is the reason
   the notebook carries four different RAVDESS extraction cells (26, 38, 49,
   64) with three different dedupe rules.

## 2. Is the data "in sync" with 8-class classification?

Short answer: **the corpus covers 8 classes, but no single dataset covers all
8, and the last pipeline the repo shipped did not actually train on 8.**

Per-dataset emotion vocabularies (*verified* from notebook cells 14, 21, 27):

| dataset | angry | calm | disgust | fearful | happy | neutral | sad | surprised |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| CREMA-D (7,442) | ✓ | – | ✓ | ✓ | ✓ | ✓ | ✓ | – |
| RAVDESS (1,440) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TESS (2,800) | ✓ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (`ps`) |
| **union** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

What this means in practice:

- `calm` exists **only** in RAVDESS: 192 clips after dedupe (*verified*, cell
  58), 2.7 % of the corpus, from 24 actors.
- `surprised` exists in RAVDESS (192 clips) and TESS (`ps`, 400 clips).
- CREMA-D cannot teach `calm` or `surprised` at all.

Now the three ways the last pipeline was **out of sync** with its own label set:

| # | symptom | evidence |
|---|---|---|
| 1 | Silent data loss: trained on **1,139 of 7,111** clips (84 % dropped) | cell 71 printed `skipping 5972` / `skipping 1470` (*verified*) |
| 2 | Final TESS map has no `ps` entry, so **all 400 pleasant-surprise files vanished** | cell 21 shows `ps: 400`; cell 85 index shows TESS `2,400` (*verified*) |
| 3 | Test set therefore has **6 classes, not 8**; `calm` and `surprised` support were 0 | cell 88 classification report (*verified*) |

So: **data yes, pipeline no.** The corpora support 8 classes; the preprocessing
that was committed does not preserve them.

### The checklist the audit tool applies

| check | pass condition | why |
|---|---|---|
| all three datasets present | 3/3 non-empty | each supplies classes the others lack |
| readable audio | 0 undecodable, <1 % tolerated | "unreadable" is how 84 % of data disappeared |
| no Git-LFS pointers | 0 | a GitHub source-archive download of CREMA-D yields 130-byte pointers |
| no AppleDouble files | 0 `._*` | harmless, but they inflate file counts |
| all 8 classes present | 8/8 | a class with no data cannot be learned |
| multi-source classes | `calm` and `surprised` should come from >1 corpus | single-source classes generalise worst |
| class balance | largest/smallest ≤ 3x (warn ≤ 6x) | 25x imbalance is the current state |
| split coverage | every split has all 8 classes | a class absent from test is never measured |
| speaker disjointness | 0 shared speakers | otherwise accuracy is voice identity |

Expected result on a correct local copy of the three corpora:
`all three datasets present: PASS`, `all 8 classes: PASS`,
`multi-source classes: WARN (calm, surprised)`, `balance: WARN`,
`READABLE audio: PASS`. Overall **WARN** — usable, with known weak classes.
If you see `FAIL` on readability or LFS pointers, that is the same defect that
destroyed the previous run.

## 3. What to do about `calm`

`calm` is the problem class: 192 clips, one corpus, 24 actors, and it is
acoustically close to `neutral` (RAVDESS `calm` is "emotion expressed at low
intensity"). Two defensible options:

- **Merge `calm` into `neutral` → 7 classes.** Removes the 152-clip class from
  training entirely, halves the chance the model produces a rare class by
  accident, and matches what `Dpngtm/wav2vec2-emotion-recognition` (the model in
  `live_emotion.py`) already does. Use
  `python augment/make_splits.py --merge-calm ...` and `labels.CLASSES_7`.
- **Keep 8 and augment `calm` hard.** Only worth it if `calm` is part of your
  claim. You then must guarantee `calm` appears in the test split (profile
  `3way`), and you must report per-class recall for it — the previous run
  reported weighted F1 0.5267 while `calm` recall was undefined.

Recommendation: **7 classes for the shipped model, 8-class only as an
ablation.** The literature agrees — most SER papers use 6-7 classes and merge
calm/neutral.

## 4. Augmentation plan (free, offline, CPU)

Ordered by measured value per unit of effort for this specific dataset:

| priority | transform | setting | why it helps here |
|---|---|---|---|
| 1 | **speed perturbation** | 0.92–1.08 | 110 actors total is a small speaker pool; tempo/pitch jitter makes each actor look like several. Standard in Kaldi/ESPnet recipes |
| 2 | **band-limit / phone channel** | 300–3400 Hz | the demo runs on a laptop mic in a room; training audio is studio-clean. This is the actual train/inference domain gap |
| 3 | **synthetic reverb** | RT60 0.15–0.45 s, 15–35 % wet | same gap, other half (room, not mic) |
| 4 | **pink noise at 12–30 dB SNR** | probability 0.5 | prevents the model keying on absolute noise floor |
| 5 | **per-clip loudness normalisation** | -28…-16 dBFS target | acted corpora are level-consistent; live audio is not |
| 6 | **time shift + short masking** | ≤150 ms, ≤120 ms masks | cheap robustness, acts like SpecAugment |
| 7 | **pitch shift** | ±1.5 semitones | mild only; heavy shifts change perceived emotion |
| 8 | **within-class mixup** | α = 0.3, same label only | helps the 152-clip classes without inventing labels |

Deliberately **not** included: added background music, added second speaker,
heavy time-stretch (>1.15), heavy pitch shift (>3 semitones), and cropping
words. All of these can change the emotion or the transcription, i.e. they
inject label noise — which is exactly what you cannot afford with 152 clips of
`calm`.

Sizing: the previous run had 7,109 train clips. With `--copies 3` plus
`--balance` on the rare classes you land at ~21-25 k clips, which is still
small for wav2vec2 but ~4x cheaper than collecting new data. Disk: 16 kHz mono
PCM_16 is 32 kB/s, so 25 k clips × 3 s ≈ 2.4 GB.

### Vocabulary of tools, all free tier, no account required

| tool | what you get | licence | note |
|---|---|---|---|
| `audiomentations` | 30+ transforms incl. `RoomSimulator`, `SevenBandParametricEQ`, `AddBackgroundNoise` | MIT | `pip install audiomentations`; the closest "batteries included" option |
| `torch-audiomentations` | same idea, GPU, batched | MIT | if you move to a `Trainer` loop |
| `librosa` effects | `time_stretch`, `pitch_shift`, `preemphasis` | ISC | already in `requirements.txt` |
| **MUSAN** | noise + music + speech corpora for `AddBackgroundNoise` | free for research | the standard noise source; ~10 GB full, a subset is enough |
| **DEMAND** | 16 real environments, 48 kHz | free for research | better than synthetic RIR if you want realism |
| **OpenSLR RIRs (`slr28`)** | real room impulse responses | CC BY | more realistic than the synthetic RIR used here |
| SpecAugment | time/frequency masking on the log-mel | free | built into `transformers` for some heads; for wav2vec2 the input is raw audio, so use the waveform masking here |
| Whisper / Wav2Vec2 + TTS | synthetic "actors" via TTS voices | varies | cheap speaker diversity; quality is the limit for emotion |

What will **not** help, despite being commonly suggested:
- **SMOTE / class-weighted sampling on raw waveforms** — interpolating two
  different speakers' waveforms produces a voice that exists nowhere and
  frequently sounds like a third emotion.
- **Mixup across classes** — creates ambiguous labels; with 8 classes on
  ~100 actors the noise dominates.
- **More epochs on the same 7 k clips** — the previous run already went from
  1.6267 to 0.8475 train loss while validation accuracy moved 49.97 → 64.78 →
  it is starting to fit actors, not emotions.

## 5. Expected effect (honest estimate)

Augmentation of this kind typically buys **3-8 accuracy points** on a
speaker-disjoint split for a small acted corpus, mostly by closing the
train/inference domain gap, and it disproportionately helps the rare classes.

Larger wins, in order, for this repo:

1. **Fix the data loss and the `ps` mapping** (Section 2). The last run trained
   on 16 % of its data and evaluated on 6 of 8 classes; no augmentation
   compensates for that.
2. **Stop selecting the checkpoint on TESS** (see `instructions.txt` C-3). The
   reported 52.54 % is the max of three checkpoints scored on the reported set.
3. **Augment + 7 classes.** Fewer, better-supported classes beat 8 with two
   starved ones.
4. **Only then** tune the model (LR warmup, scheduler, layer freezing).

Measured baseline to beat (*verified*, cell 88): 2,400 TESS clips, accuracy
0.4813, weighted F1 0.4797, macro F1 0.36, with `disgust` recall 0.99 and
`fearful` recall 0.16 — a model that collapses most inputs into two classes.
A macro F1 above 0.55 with all classes recalled above 0.4 would be a real
improvement, not a re-labelled one.

## 6. Commands, end to end

```bash
# 0. unzip everywhere first -- the tools never read inside archives
unzip ravdess.zip -d data/RAVDESS
unzip "TESS Toronto emotional speech set data.zip" -d data/TESS

# 1. is the data complete and in sync?
python augment/audit_datasets.py --scan data --json audit.json

# 2. speaker-disjoint splits (3way keeps all 8 classes in every split)
python augment/make_splits.py --scan data --profile 3way --out data/splits

# 3. augment only the training split
python augment/augment_audio.py --splits data/splits/splits.json \
       --out data/aug --balance --max-copies 6 --preset minority --window 3

# 4. sanity-check and listen
python augment/benchmark_augmentation.py --scan data/TESS --n 2 --out /tmp/aug_demo
python augment/tests/test_audio_ops.py
```

Then point the npz preprocessing step at `data/aug/train.csv` and keep using
`data/splits/{val,test}.csv` unchanged.

# Training and Evaluation

The custom speech emotion recognition model was fine-tuned from `facebook/wav2vec2-base` using the CREMA-D and RAVDESS datasets.

## Dataset Summary
- CREMA-D: 7,439 samples
- RAVDESS: 1,439 samples
- TESS: 2,400 samples
- Total: 11,278 samples

## Training
The model was trained for three epochs in Google Colab. Validation performance improved across the training runs.

## Evaluation
- Epoch 1 validation accuracy: 49.97%
- Epoch 2 validation accuracy: 62.92%
- Epoch 3 validation accuracy: 64.78%
- Epoch 2 TESS comparison accuracy: 52.54%

TESS was used for checkpoint comparison, so its score is not an independent final test estimate.

The training notebook and detailed evaluation results should be added here as they are prepared.

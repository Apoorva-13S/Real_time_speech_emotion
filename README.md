# Real-Time On-Device Speech Emotion Recognition

## Overview

This project aims to recognize emotions expressed in speech using a Wav2Vec2-based deep learning model. The system is designed to accept audio input, process speech, and predict an emotion with confidence scores.

## Objectives

- Develop a speech emotion classification pipeline.
- Fine-tune a pretrained Wav2Vec2 model for emotion recognition.
- Evaluate performance on different speech datasets.
- Integrate the model with microphone-based audio input.
- Explore local inference without requiring a cloud prediction API.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Wav2Vec2
- NumPy and audio-processing libraries
- Google Colab for model training
- VS Code for local development

## Datasets

The project uses CREMA-D, RAVDESS and TESS.

The indexed dataset contains 11,278 samples:
- Training: 7,109 samples
- Validation: 1,769 samples
- TESS evaluation: 2,400 samples

## Model Training

The Wav2Vec2-base model was fine-tuned for eight emotion classes over three epochs.

The highest validation accuracy across these epochs was 64.78%, achieved in Epoch 3.

## Evaluation

During checkpoint comparison on TESS, Epoch 2 achieved:
- Accuracy: 52.54%
- Weighted F1-score: 0.5267

TESS was used to compare checkpoints, so this score should not be interpreted as an entirely independent final test estimate.

## Current Status

- Dataset preparation and indexing completed.
- Model fine-tuning completed.
- Validation and cross-dataset checkpoint evaluation completed.
- Microphone recording and prediction tested with a pretrained model.
- Integration and verification of the custom checkpoint in the final real-time pipeline remain in progress.

## Limitations

- Performance varies across datasets and recording conditions.
- Acted emotional speech may differ from natural conversation.
- Predictions are estimates and do not establish a person's actual emotional state.
- Real-time latency and resource usage require further measurement.

## Future Work

- Complete integration of the custom model with microphone input.
- Measure inference latency and memory usage.
- Improve generalization to natural speech.
- Perform additional evaluation on data not used for checkpoint selection.

## Running the Project

Installation instructions and required dependencies will be provided in `requirements.txt`. Model checkpoint setup instructions will be documented separately.

**Note:** This repository should contain the source code and documentation required to understand the project. Large model weights and datasets may need to be obtained separately.

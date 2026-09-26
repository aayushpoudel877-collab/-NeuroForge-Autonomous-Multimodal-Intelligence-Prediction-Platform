# NeuroForge Deep Learning Layer

NeuroForge now includes a trainable PyTorch reference layer alongside the deterministic baseline adapters.

## Architecture

Each modality maps its feature vector into a shared embedding space. A learned softmax gate estimates the contribution of each modality, producing a weighted fused representation before binary prediction.

Supported reference inputs are text (8 features), image (10), audio (8), and time-series (6).

The neural layer is separated from API adapters so training can evolve independently from serving. The included trainer uses synthetic data only and is intended for architecture validation, not scientific benchmarking.

## Demo

Install the optional deep-learning dependency and run: python scripts/train_neural_demo.py

For research use, replace the synthetic dataset with versioned datasets and add modality-specific pretrained encoders such as Transformer, ViT, wav2vec-style, and temporal architectures. Record dataset versions, splits, calibration results, and model artifacts for reproducibility.

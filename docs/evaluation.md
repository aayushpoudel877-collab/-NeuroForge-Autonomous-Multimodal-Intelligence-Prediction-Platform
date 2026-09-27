# Evaluation and Benchmarking

NeuroForge now includes reproducible data preparation, neural evaluation, and modality-ablation benchmarking.

## Demo dataset
```bash
python -m neuroforge.cli prepare-demo --output data/synthetic --samples 24
```
This is a local development fixture, not a real-world benchmark.

## Evaluation
```bash
python -m neuroforge.cli evaluate --manifest data/synthetic/manifest.json --root data/synthetic
```
Supply `--checkpoint` to evaluate a trained checkpoint.

## Modality benchmark
```bash
python -m neuroforge.cli benchmark --manifest data/synthetic/manifest.json --root data/synthetic
```
The benchmark compares full fusion against one-modality-at-a-time ablations. Results are diagnostic evidence, not a universal ranking of modalities.

## External datasets
External data is deliberately not bundled. Build a `DatasetManifest`, validate it, use `ManifestDataset`, and record dataset/version metadata with the experiment.

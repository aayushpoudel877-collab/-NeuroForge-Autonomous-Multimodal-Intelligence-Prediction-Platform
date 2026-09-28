# Training and Experiment Execution

The training layer connects the multimodal dataset contract to PyTorch training and validation.

## Lifecycle
1. Build or load a validated `DatasetManifest`.
2. Materialize `ManifestDataset`.
3. Create deterministic batches with `batch_dataset`.
4. Train with `TrainerEngine`.
5. Validate after every epoch.
6. Save the best checkpoint by validation F1.
7. Stop early when validation F1 stops improving.
8. Track the immutable `RunConfig` and deterministic run ID.
9. Optionally fit temperature scaling on held-out logits.

## Reproducibility
Training seeds Python, NumPy, and PyTorch. Exact bit-for-bit results can still vary across hardware and CUDA kernels.

## CLI
```bash
neuroforge train --manifest data/synthetic/manifest.json --root data/synthetic --epochs 3
```

The CLI writes the best checkpoint under `models/checkpoints` unless another directory is supplied.

## Checkpoints
A checkpoint contains model weights, optimizer state, epoch, and selection metric. Only validated trained artifacts should be promoted to the model registry.

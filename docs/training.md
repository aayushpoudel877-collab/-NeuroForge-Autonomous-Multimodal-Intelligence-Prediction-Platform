# Training Infrastructure

NeuroForge now includes a manifest-backed multimodal dataset abstraction, checkpoint persistence, early stopping, binary evaluation metrics, and optional PyTorch automatic mixed precision helpers.

The checkpoint manager stores model and optimizer state together with epoch and validation metric. Early stopping supports both maximizing and minimizing objectives.

The dataset abstraction intentionally keeps modality decoding injectable. This separates dataset orchestration from domain-specific image, audio, text, and temporal preprocessing.

The current implementation is infrastructure rather than a claim of trained scientific performance. Real experiments should use representative datasets, explicit preprocessing contracts, reproducible splits, and tracked configurations.

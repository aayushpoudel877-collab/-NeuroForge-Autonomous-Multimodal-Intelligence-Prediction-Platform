# Advanced Model Intelligence

NeuroForge provides optional adapters for Hugging Face Transformer text encoders, torchvision vision backbones, and torchaudio Wav2Vec2 audio embeddings. These are optional because weights can be large and may require internet access. The repository does not bundle model weights.

The unified model accepts a [batch, 4] modality mask. Missing inputs use learned modality-specific embeddings instead of treating absence as a zero-valued signal.

Binary evaluation includes accuracy, precision, recall, F1, AUROC, PR-AUC, Brier score and expected calibration error (ECE). AUROC and PR-AUC are reported as 0.0 when only one class is present because they are undefined.

The model-card utility records task, modalities, evaluation metrics and declared limitations, with an explicit separation between measured evaluation results and real-world performance.

Install the optional stack with: pip install -e ".[advanced]"

The synthetic dataset remains a development fixture. No real-world performance claim is made until a suitable licensed dataset and reproducible evaluation protocol are supplied.

# Data Pipeline

NeuroForge now has dataset-manifest, reproducible splitting, quality profiling, validation, seeding, augmentation helpers, and batch collation primitives.

The manifest keeps a stable sample identifier, binary label, and optional paths for text, image, audio, and time-series sources. Splits are deterministic when a seed is fixed.

The current augmentation helpers are lightweight NumPy primitives intended for pipeline composition. They do not claim domain-specific scientific validity; production training should select augmentations appropriate to each dataset and modality.

The next integration layer can connect these primitives to PyTorch Dataset/DataLoader implementations and real modality decoders.

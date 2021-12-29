"""Contains custom errors."""


class AttemptToUseLabelsError(ValueError):
    """Raised when a PokemonGenerationDataProcessor's label handling methods are
    called. Since the task is unsupervised (or self-supervised), no label
    tensors exist; thus, label handling methods are invalid."""


class GANShapeError(ValueError):
    """Raised when a GAN is initialized with generator/discriminator models that
    have incorrect input or output shapes."""

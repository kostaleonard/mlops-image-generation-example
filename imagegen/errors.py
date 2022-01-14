"""Contains custom errors."""


class AttemptToUseLabelsError(ValueError):
    """Raised when a PokemonGenerationDataProcessor's label handling methods are
    called. Since the task is unsupervised (or self-supervised), no label
    tensors exist; thus, label handling methods are invalid."""


class GANShapeError(ValueError):
    """Raised when a GAN is initialized with generator/discriminator models that
    have incorrect input or output shapes."""


class GANHasNoOptimizerError(ValueError):
    """Raised when a GAN is initialized with generator/discriminator models that
    do not have optimizers. This error can be remedied by compiling the model
    (an optimizer will be initialized by default, but one can also be supplied
    as an argument to compile)."""


class WGANDiscriminatorActivationNotLinearError(ValueError):
    """Raised when a WGAN is initialized with discriminator model that uses an
    activation function, rather than leaving the output as-is (sometimes called
    "linear" activation)."""


class WGANOptimizersNotRMSPropError(ValueError):
    """Raised when a WGAN is initialized with generator/discriminator models
    that do not use RMSProp as their optimizers. Optimizers with momentum, such
    as Adam, interfere with training stability and cannot be used."""

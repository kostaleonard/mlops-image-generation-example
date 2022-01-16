"""Tests errors.py."""

from imagegen.errors import AttemptToUseLabelsError, GANShapeError, \
    GANHasNoOptimizerError, WGANDiscriminatorActivationNotLinearError, \
    WGANOptimizersNotRMSPropError, IncompatibleCommandLineArgumentsError


def test_attempt_to_use_labels_error_extends_value_error() -> None:
    """Tests that AttemptToUseLabelsError extends ValueError."""
    err = AttemptToUseLabelsError()
    assert isinstance(err, ValueError)


def test_gan_shape_error_extends_value_error() -> None:
    """Tests that GANShapeError extends ValueError."""
    err = GANShapeError()
    assert isinstance(err, ValueError)


def test_gan_has_no_optimizer_error_extends_value_error() -> None:
    """Tests that GANHasNoOptimizerError extends ValueError."""
    err = GANHasNoOptimizerError()
    assert isinstance(err, ValueError)


def test_wgan_discriminator_activation_error_extends_value_error() -> None:
    """Tests that WGANDiscriminatorActivationNotLinearError extends
    ValueError."""
    err = WGANDiscriminatorActivationNotLinearError()
    assert isinstance(err, ValueError)


def test_wgan_optimizer_error_extends_value_error() -> None:
    """Tests that WGANOptimizersNotRMSPropError extends ValueError."""
    err = WGANOptimizersNotRMSPropError()
    assert isinstance(err, ValueError)


def test_wgan_incompatible_arguments_error_extends_value_error() -> None:
    """Tests that IncompatibleCommandLineArgumentsError extends ValueError."""
    err = IncompatibleCommandLineArgumentsError()
    assert isinstance(err, ValueError)

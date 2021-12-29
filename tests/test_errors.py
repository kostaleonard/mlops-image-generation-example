"""Tests errors.py."""

from imagegen.errors import AttemptToUseLabelsError, GANShapeError, \
    GANHasNoOptimizerError


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

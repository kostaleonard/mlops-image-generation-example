"""Tests errors.py."""

from imagegen.errors import AttemptToUseLabelsError, GANShapeError


def test_attempt_to_use_labels_error_extends_value_error() -> None:
    """Tests that AttemptToUseLabelsError extends ValueError."""
    err = AttemptToUseLabelsError()
    assert isinstance(err, ValueError)


def test_gan_shape_error_extends_value_error() -> None:
    """Tests that GANShapeError extends ValueError."""
    err = GANShapeError()
    assert isinstance(err, ValueError)

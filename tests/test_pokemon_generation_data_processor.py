"""Tests pokemon_generation_data_processor.py."""


def test_get_raw_features_and_labels_returns_expected_keys() -> None:
    """Tests that get_raw_features_and_labels returns the expected keys for the
    train/val/test dataset."""
    # TODO
    assert False


def test_get_raw_features_and_labels_trainvaltest_correct_split() -> None:
    """Tests that the train/val/test datasets are split into non-empty
    subsets."""
    # TODO
    assert False


def test_get_raw_features_trainvaltest_returns_expected_keys() -> None:
    """Tests that get_raw_features returns the expected keys {'X_train',
    'X_val', 'X_test} when called on the train/val/test directory.
    """
    # TODO
    assert False


def test_get_raw_features_match() -> None:
    """Tests that the features produced by get_raw_features_and_labels and
    get_raw_features are the same features."""
    # TODO
    assert False


def test_get_raw_features_correct_shape() -> None:
    """Tests that get_raw_features returns tensors with the expected shapes."""
    # TODO
    assert False


def test_get_raw_features_correct_dtype() -> None:
    """Tests that get_raw_features returns tensors with dtype float32."""
    # TODO
    assert False


def test_get_raw_features_correct_value_range() -> None:
    """Tests that get_raw_features returns tensors in the range [0, 1]."""
    # TODO
    assert False


def test_get_raw_features_no_na() -> None:
    """Tests that get_raw_features returns tensors with no missing values."""
    # TODO
    assert False


def test_get_raw_features_have_multiple_pixel_values() -> None:
    """Tests that the images were loaded correctly by ensuring that more than
    one pixel value exists in the tensors."""
    # TODO
    assert False


def test_get_raw_labels_empty() -> None:
    """Tests that the returned labels are the empty dictionary."""
    # TODO
    assert False


def test_preprocessed_features_same_as_raw() -> None:
    """Tests that the preprocessed features have the same shape as the raw
    features."""
    # TODO
    assert False


def test_preprocess_labels_raises_error() -> None:
    """Tests that preprocess_labels raises an error, since there are no labels
    for this task."""
    # TODO
    assert False


def test_unpreprocess_features_inverts_transformation() -> None:
    """Tests that unpreprocessing the preprocessed features results in the raw
    features."""
    # TODO
    assert False


def test_unpreprocess_labels_raises_error() -> None:
    """Tests that unpreprocess_labels raises an error, since there are no labels
    for this task."""
    # TODO
    assert False

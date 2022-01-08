"""Contains the PokemonGenerationDataProcessor class."""

import os
from typing import Dict
import numpy as np
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import random_rotation
from mlops.dataset.invertible_data_processor import InvertibleDataProcessor
from imagegen.errors import AttemptToUseLabelsError

DEFAULT_DATASET_PATH = 'data'
IMAGES_DIRNAME = 'images'
TRAIN_SPLIT = 0.9
VAL_SPLIT = 0.05
HEIGHT = 120
WIDTH = 120
CHANNELS = 3
AUGMENTATIONS_PER_IMAGE = 5
ROTATION_RANGE = 20
HORIZONTAL_FLIP_CHANCE = 0.5
CHANNEL_AXIS = 2


class PokemonGenerationDataProcessor(InvertibleDataProcessor):
    """Transforms the pokemon dataset at data/pokemon into features for image
    generation."""

    def __init__(
            self,
            augmentations_per_image: int = AUGMENTATIONS_PER_IMAGE,
            rotation_range_degrees: int = ROTATION_RANGE) -> None:
        """Instantiates the object.

        :param augmentations_per_image: The number of augmentations applied to
            each image. Set to 0 for no augmentation.
        :param rotation_range_degrees: The rotation range applied in image
            augmentation, in degrees.
        """
        self.augmentations_per_image = augmentations_per_image
        self.rotation_range_degrees = rotation_range_degrees

    def get_raw_features_and_labels(self, dataset_path: str) -> \
            (Dict[str, np.ndarray], Dict[str, np.ndarray]):
        """Returns the raw feature and label tensors from the dataset path. This
        method is specifically used for the train/val/test sets and not input
        data for prediction, because in some cases the features and labels need
        to be read simultaneously to ensure proper ordering of features and
        labels.

        Raw features are tensors of shape m x h x w x c, where m is the number
        of images, h is the image height, w is the image width, and c is the
        number of channels (3 for RGB), with all values in the interval
        [0, 1]. This is an image generation task, so there are no labels.

        :param dataset_path: The path to the file or directory on the local or
            remote filesystem containing the dataset, specifically
            train/val/test and not prediction data.
        :return: A 2-tuple of the features dictionary and labels dictionary,
            with matching keys and ordered tensors.
        """
        return self.get_raw_features(dataset_path), {}

    def get_raw_features(self, dataset_path: str) -> Dict[str, np.ndarray]:
        """Returns the raw feature tensors from the prediction dataset path. Raw
        features are tensors of shape m x h x w x c, where m is the number of
        images, h is the image height, w is the image width, and c is the number
        of channels (3 for RGB), with all values in the interval [0, 1]. The
        features are already scaled because PNG images load into float32 instead
        of uint8.

        :param dataset_path: The path to the file or directory on the local or
            remote filesystem containing the dataset.
        :return: A dictionary whose values are feature tensors and whose
            corresponding keys are the names by which those tensors should be
            referenced. The returned keys will be {'X_train', 'X_val', 'X_test'}
            if the directory indicated by dataset_path ends with 'trainvaltest',
            and {'X_pred'} otherwise.
        """
        X = []
        image_filenames = [
            filename for filename in
            os.listdir(os.path.join(dataset_path, IMAGES_DIRNAME))
            if filename.endswith('.jpg') or filename.endswith('.png')]
        for filename in image_filenames:
            full_path = os.path.join(dataset_path, IMAGES_DIRNAME, filename)
            # Discard alpha channel.
            tensor = imread(full_path)[:, :, :3]
            if filename.endswith('.jpg'):
                tensor = tensor.astype(np.float32) / 255
            X.append(tensor)
            for _ in range(self.augmentations_per_image):
                augmentation = self._get_augmented_image(tensor)
                X.append(augmentation)
        X = np.array(X)
        np.random.shuffle(X)
        features = {}
        num_train = int(len(X) * TRAIN_SPLIT)
        num_val = int(len(X) * VAL_SPLIT)
        features['X_train'] = X[:num_train]
        features['X_val'] = X[num_train:num_train + num_val]
        features['X_test'] = X[num_train + num_val:]
        return features

    def preprocess_features(self, raw_feature_tensor: np.ndarray) -> np.ndarray:
        """Returns the preprocessed feature tensor from the raw tensor. The
        preprocessed features are how training/validation/test as well as
        prediction data are fed into downstream models. The preprocessed tensors
        are of shape m x h x w x c, where m is the number of images, h is the
        image height, w is the image width, and c is the number of channels
        (3 for RGB), with all values in the interval [0, 1].

        :param raw_feature_tensor: The raw features to be preprocessed.
        :return: The preprocessed feature tensor. This tensor is ready for
            downstream model consumption.
        """
        # PNG images, when loaded by imread, are already scaled into [0, 1].
        return raw_feature_tensor.copy()

    def preprocess_labels(self, raw_label_tensor: np.ndarray) -> np.ndarray:
        """Returns the preprocessed label tensor from the raw tensor. The
        preprocessed labels are how training/validation/test as well as
        prediction data are fed into downstream models. This is a generation
        task, so there are no labels.

        :param raw_label_tensor: The raw labels to be preprocessed.
        :return: The preprocessed label tensor. This tensor is ready for
            downstream model consumption.
        """
        raise AttemptToUseLabelsError

    def unpreprocess_features(self, feature_tensor: np.ndarray) -> np.ndarray:
        """Returns the raw feature tensor from the preprocessed tensor; inverts
        preprocessing. Improves model interpretability by enabling users to
        transform model inputs into real-world values.

        :param feature_tensor: The preprocessed features to be inverted.
        :return: The raw feature tensor.
        """
        # No preprocessing was necessary.
        return feature_tensor.copy()

    def unpreprocess_labels(self, label_tensor: np.ndarray) -> np.ndarray:
        """Returns the raw label tensor from the preprocessed tensor; inverts
        preprocessing. Improves model interpretability by enabling users to
        transform model outputs into real-world values.

        :param label_tensor: The preprocessed labels to be inverted.
        :return: The raw label tensor.
        """
        raise AttemptToUseLabelsError

    def _get_augmented_image(self, image: np.ndarray) -> np.ndarray:
        """Returns an augmented image based on various affine transformations.

        :param image: The image tensor to be transformed.
        :return: An augmented image based on various affine transformations.
        """
        # Random rotation.
        augmented = random_rotation(image,
                                    self.rotation_range_degrees,
                                    channel_axis=CHANNEL_AXIS)
        # Horizontal flip.
        if np.random.random() < HORIZONTAL_FLIP_CHANCE:
            augmented = np.fliplr(augmented)
        return augmented

"""Demonstrates a generative adversarial network architecture on the pokemon
dataset. The model will generate images of never-before-seen pokemon. The
positive class is used to denote real images.

This model syncs with WandB and allows greater control over hyperparameters.
"""

from abc import ABC, abstractmethod
from typing import Optional
import time
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model as tf_load_model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Optimizer, Adam
from tensorflow.keras.callbacks import History
import wandb
from mlops.dataset.versioned_dataset import VersionedDataset

DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 32
DEFAULT_GEN_INPUT_DIM = 128
DEFAULT_CKPT_PREFIX = 'models/gan/checkpoints/gan'
CROSS_ENTROPY_LOSS = BinaryCrossentropy(from_logits=False)
WANDB_PROJECT_TITLE = 'gan_pokemon'
MAX_NUM_WANDB_IMAGES = 50
WANDB_IMAGE_ROWS = 4
WANDB_IMAGE_COLS = 4


class GAN(ABC):
    """Represents a generative adversarial network model."""

    def __init__(self,
                 gen_output_shape: tuple[int],
                 gen_input_dim: int = DEFAULT_GEN_INPUT_DIM) -> None:
        """Instantiates the GAN.

        :param gen_output_shape: The shape of generated outputs (less the first
            dimension that indicates the number of examples). For images, this
            would likely be h x w x c, where h is the image height, w is the
            image width, and c is the number of channels.
        :param gen_input_dim: The size of the noise vector that the generator
            uses as input.
        """
        self.model_hyperparams = locals()
        self.model_hyperparams.pop('self')
        self.gen_output_shape = gen_output_shape
        self.gen_input_dim = gen_input_dim
        self._generator = self._get_new_generator()
        self._discriminator = self._get_new_discriminator()
        # TODO raise error if shapes are incompatible

    @abstractmethod
    def _get_new_generator(self) -> Model:
        """Returns a new instance of the GAN's generator model. Subclasses
        determine this architecture. The output must be of shape
        self.gen_output_shape.

        :return: A new instance of the GAN's generator model.
        """

    @abstractmethod
    def _get_new_discriminator(self) -> Model:
        """Returns a new instance of the GAN's discriminator model. Subclasses
        determine this architecture.

        :return: A new instance of the GAN's discriminator model.
        """

    def save_model(self,
                   generator_filename: str,
                   discriminator_filename: str) -> None:
        """Saves the generator and discriminator networks to the given paths.

        :param generator_filename: The path to which to save the generator.
        :param discriminator_filename: The path to which to save the
            discriminator.
        """
        # TODO how does this interact with versionedmodel?
        self._generator.save(generator_filename)
        self._discriminator.save(discriminator_filename)

    @staticmethod
    def load_model(generator_filename: str,
                   discriminator_filename: str) -> tuple[Model, Model]:
        """Returns the generator and discriminator networks saved at the
        specified locations.

        :param generator_filename: The path to the saved generator.
        :param discriminator_filename: The path to the saved discriminator.
        :return: A 2-tuple of the loaded generator and discriminator,
            respectively.
        """
        # TODO how does this interact with versionedmodel?
        generator = tf_load_model(generator_filename)
        discriminator = tf_load_model(discriminator_filename)
        return generator, discriminator

    @staticmethod
    def _generator_loss(fake_output: np.ndarray) -> float:
        """Returns the generator's loss based on the discriminator's predictions
        on generated images.

        :param fake_output: The discriminator's predictions on fake images
            output by the generator. If the discriminator can spot the fakes,
            these predictions will be close to 0; if the generator can fool the
            discriminator, these predictions will be close to 1. The predictions
            are a tensor of shape m x 1, where m is the batch size.
        :return: The cross-entropy loss of fake_output against an array of ones;
            the generator loss is minimized when it completely fools the
            discriminator.
        """
        return CROSS_ENTROPY_LOSS(tf.ones_like(fake_output), fake_output)

    @staticmethod
    def _discriminator_loss(real_output: np.ndarray,
                            fake_output: np.ndarray) -> float:
        """Returns the discriminator's loss on batches of real and fake images.

        :param real_output: The discriminator's predictions on real images. If
            the discriminator can spot real images, these predictions will be
            close to 1. The predictions are a tensor of shape m x 1, where m is
            the batch size.
        :param fake_output: The discriminator's predictions on fake images
            output by the generator. If the discriminator can spot the fakes,
            these predictions will be close to 0; if the generator can fool the
            discriminator, these predictions will be close to 1. The predictions
            are a tensor of shape m x 1, where m is the batch size.
        :return: The cross-entropy loss of real_output against an array of ones
            and fake_output against an array of zeros; the discriminator loss is
            minimized when it correctly differentiates between real and fake
            images.
        """
        real_loss = CROSS_ENTROPY_LOSS(tf.ones_like(real_output), real_output)
        fake_loss = CROSS_ENTROPY_LOSS(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    @tf.function
    def _train_step(self,
                    X_batch: np.ndarray,
                    gen_optimizer: Optimizer,
                    dis_optimizer: Optimizer) -> tuple[float, float]:
        """Runs one batch of images through the model, computes the loss, and
        applies the gradients to the model; returns the generator and
        discriminator losses on the batch.

        :param X_batch: A batch of input images; a tensor of shape m x h x w x
            c, where m is the batch size, h is the image height, w is the image
            width, and c is the number of channels.
        :param gen_optimizer: The optimizer that will minimize the
            generator loss using the gradient information.
        :param dis_optimizer: The optimizer that will minimize the
            discriminator loss using the gradient information.
        :return: A 2-tuple of the generator and discriminator losses on the
            batch.
        """
        # pylint: disable=invalid-name
        noise = tf.random.normal((len(X_batch), self.gen_input_dim))
        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            generated_images = self._generator(noise, training=True)
            real_output = self._discriminator(X_batch, training=True)
            fake_output = self._discriminator(generated_images, training=True)
            gen_loss = GAN._generator_loss(fake_output)
            dis_loss = GAN._discriminator_loss(real_output, fake_output)
        gen_gradients = gen_tape.gradient(
            gen_loss, self._generator.trainable_variables)
        dis_gradients = dis_tape.gradient(
            dis_loss, self._discriminator.trainable_variables)
        gen_optimizer.apply_gradients(
            zip(gen_gradients, self._generator.trainable_variables))
        dis_optimizer.apply_gradients(
            zip(dis_gradients, self._discriminator.trainable_variables))
        return gen_loss, dis_loss

    def train(self,
              dataset: VersionedDataset,
              epochs: int = DEFAULT_EPOCHS,
              batch_size: int = DEFAULT_BATCH_SIZE,
              model_checkpoint_prefix: Optional[str] = DEFAULT_CKPT_PREFIX,
              use_wandb: bool = True) -> History:
        """Trains the model on the training data.

        :param dataset: The dataset on which to train the model.
        :param epochs: The number of complete passes over the dataset to run
            training.
        :param batch_size: The size of the batches used in mini-batch gradient
            descent.
        :param model_checkpoint_prefix: If specified, the prefix of the path to
            which to save the generator and discriminator. The generator
            file will have the suffix '_generator.h5' and the discriminator
            will have the suffix '_discriminator.h5'.
        :param use_wandb: If True, sync the run with WandB.
        :return: The training History object.
        """
        # pylint: disable=too-many-locals, too-many-statements
        train_hyperparams = locals()
        train_hyperparams.pop('self')
        train_hyperparams.pop('dataset')
        # These files will only be created if a prefix was supplied.
        generator_checkpoint_filename = \
            f'{model_checkpoint_prefix}_generator.h5'
        discriminator_checkpoint_filename = \
            f'{model_checkpoint_prefix}_discriminator.h5'
        if use_wandb:
            all_hyperparams = {**self.model_hyperparams, **train_hyperparams}
            wandb.init(project=WANDB_PROJECT_TITLE,
                       dir='.',
                       config=all_hyperparams)
            wandb.run.summary['generator_graph'] = wandb.Graph.from_keras(
                self._generator)
            wandb.run.summary['discriminator_graph'] = wandb.Graph.from_keras(
                self._discriminator)
        gen_optimizer = Adam(1e-4)
        dis_optimizer = Adam(1e-4)
        train_dataset = tf.data.Dataset \
            .from_tensor_slices(dataset.X_train) \
            .shuffle(len(dataset.X_train)) \
            .batch(batch_size)
        generate_images_epochs = {
            int(num * (epochs - 1) / MAX_NUM_WANDB_IMAGES)
            for num in range(1, MAX_NUM_WANDB_IMAGES + 1)
        }
        history = History()
        history.history['epoch'] = []
        history.history['loss'] = []
        history.history['gen_loss'] = []
        history.history['dis_loss'] = []
        for epoch in range(epochs):
            gen_loss = 0
            dis_loss = 0
            num_batches = 0
            start_time = time.time()
            for train_batch in tqdm(train_dataset):
                gen_loss_batch, dis_loss_batch = self._train_step(
                    train_batch, gen_optimizer, dis_optimizer)
                gen_loss += gen_loss_batch
                dis_loss += dis_loss_batch
                num_batches += 1
            end_time = time.time()
            gen_loss /= num_batches
            dis_loss /= num_batches
            loss = gen_loss + dis_loss
            print(f'Epoch {epoch} ({end_time - start_time:.1f}s): loss='
                  f'{loss:.3f}, gen_loss={gen_loss:.3f}')
            if model_checkpoint_prefix:
                self.save_model(generator_checkpoint_filename,
                                discriminator_checkpoint_filename)
                print(f'Generator loss={gen_loss:.3f}; saving model.')
            if use_wandb:
                logged_items = {
                    'epoch': epoch,
                    'loss': loss,
                    'generator_loss': gen_loss,
                    'discriminator_loss': dis_loss
                }
                if epoch in generate_images_epochs:
                    generated_batch = self.generate(
                        WANDB_IMAGE_ROWS * WANDB_IMAGE_COLS)
                    concatenated_images = GAN._concatenate_images(
                        generated_batch, WANDB_IMAGE_ROWS, WANDB_IMAGE_COLS)
                    images = wandb.Image(
                        concatenated_images,
                        caption=f'Generated images at epoch {epoch}')
                    logged_items['generated_images'] = images
                wandb.log(logged_items)
                tmp_generator_filename = '/tmp/gan_generator.h5'
                tmp_discriminator_filename = '/tmp/gan_discriminator.h5'
                self.save_model(tmp_generator_filename,
                                tmp_discriminator_filename)
                wandb.save(tmp_generator_filename)
                wandb.save(tmp_discriminator_filename)
            history.history['epoch'].append(epoch)
            history.history['loss'].append(loss)
            history.history['gen_loss'].append(gen_loss)
            history.history['dis_loss'].append(dis_loss)
        if use_wandb:
            best_gen_epoch = min(
                history.history['epoch'],
                key=lambda epoch: history.history['gen_loss'][epoch])
            best_gen_loss = history.history['gen_loss'][best_gen_epoch]
            best_dis_epoch = min(
                history.history['epoch'],
                key=lambda epoch: history.history['dis_loss'][epoch])
            best_dis_loss = history.history['dis_loss'][best_dis_epoch]
            wandb.run.summary['best_gen_epoch'] = best_gen_epoch
            wandb.run.summary['best_gen_loss'] = best_gen_loss
            wandb.run.summary['best_dis_epoch'] = best_dis_epoch
            wandb.run.summary['best_dis_loss'] = best_dis_loss
        return history

    def generate(self, num_samples: int) -> np.ndarray:
        """Returns a batch of images generated by the (trained) model. The batch
        is generated based on random noise vectors.

        :param num_samples: The number of images to generate.
        :return: A batch of generated images; a tensor of shape num_samples x
            h x w x c, where h is the image height, w is the image width, and c
            is the number of channels.
        """
        # TODO clip output to [0, 1]
        noise = tf.random.normal((num_samples, self.gen_input_dim))
        return self._generator(noise)

    @staticmethod
    def _concatenate_images(images: np.ndarray,
                            rows: int,
                            cols: int) -> np.ndarray:
        """Returns a single image that is the concatenation of the (rows * cols)
        images into the specified number of rows and columns; a tensor of shape
        (h * rows) x (w * cols) x c, where images is of shape
        (rows * cols) x h x w x c.

        :param images: An array of (rows * cols) images; a tensor of shape
            (rows * cols) x h x w x c, where h is the image height, w is the
            image width, and c is the number of channels.
        :param rows: The number of rows in which to display the images.
        :param cols: The number of cols in which to display the images.
        :return: A single image that is the concatenation of the (rows * cols)
            images into the specified number of rows and columns; a tensor of
            shape (h * rows) x (w * cols) x c, where images is of shape
            (rows * cols) x h x w x c.
        """
        result = np.zeros((rows * images.shape[1],
                           cols * images.shape[2],
                           images.shape[3]))
        for row in range(rows):
            for col in range(cols):
                image_num = (row * cols) + col
                row_start = row * images.shape[1]
                row_end = (row + 1) * images.shape[1]
                col_start = col * images.shape[2]
                col_end = (col + 1) * images.shape[2]
                result[row_start:row_end,
                       col_start:col_end,
                       :] = images[image_num]
        return result

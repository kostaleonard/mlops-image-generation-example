"""Demonstrates a generative adversarial network architecture on the pokemon
dataset. The model will generate images of never-before-seen pokemon. The
positive class is used to denote real images.

This model syncs with WandB.
"""

from typing import Optional
import time
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model as tf_load_model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import History
import wandb
from mlops.dataset.versioned_dataset import VersionedDataset
from imagegen.errors import GANShapeError

DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 32
DEFAULT_GEN_INPUT_DIM = 128
DEFAULT_CKPT_PREFIX = 'models/gan/checkpoints/gan'
CROSS_ENTROPY_LOSS = BinaryCrossentropy(from_logits=False)
WANDB_PROJECT_TITLE = 'gan_pokemon'
MAX_NUM_WANDB_IMAGES = 50
WANDB_IMAGE_ROWS = 4
WANDB_IMAGE_COLS = 4


class GAN:
    """Represents a generative adversarial network model."""

    def __init__(self, generator: Model, discriminator: Model) -> None:
        """Instantiates the GAN.

        :param generator: The compiled generator model. Generates new images.
            The input shape must be m x n, where m is the number of examples
            and n is the length of the noise vector that the generator uses as
            input. The output shape must be m x h x w x c, where m is the number
            of examples, h is the image height, w is the image width, and c is
            the image channels. The output must be in the range [0, 1].
        :param discriminator: The compiled discriminator model. Classifies
            images as either real or fake. The input shape must be
            m x h x w x c, the same shape as the output of the generator. The
            output shape must be m x 1, where the output represents the
            probability that each example is real. This output must be in the
            range [0, 1].
        """
        if len(generator.input_shape) != 2:
            raise GANShapeError('Generator input must be of shape (m, n)')
        if len(generator.output_shape) != 4:
            raise GANShapeError('Generator output must be of shape '
                                '(m, h, w, c)')
        if generator.output_shape != discriminator.input_shape:
            raise GANShapeError('Generator output shape must match '
                                'discriminator input shape')
        if discriminator.output_shape[1:] != (1,):
            raise GANShapeError('Discriminator output must be of shape (m, 1)')
        self.gen_output_shape = generator.output_shape[1:]
        self.gen_input_dim = generator.input_shape[1]
        self.model_hyperparams = {
            'gen_input_dim': self.gen_input_dim,
            'gen_output_shape': self.gen_output_shape
        }
        self.generator = generator
        self.discriminator = discriminator

    def save_model(self,
                   generator_filename: str,
                   discriminator_filename: str) -> None:
        """Saves the generator and discriminator networks to the given paths.

        :param generator_filename: The path to which to save the generator.
        :param discriminator_filename: The path to which to save the
            discriminator.
        """
        self.generator.save(generator_filename)
        self.discriminator.save(discriminator_filename)

    @staticmethod
    def load(generator_filename: str, discriminator_filename: str) -> 'GAN':
        """Returns the GAN loaded from the generator and discriminator files.

        :param generator_filename: The path to the saved generator.
        :param discriminator_filename: The path to the saved discriminator.
        :return: The GAN loaded from the generator and discriminator files.
        """
        generator = tf_load_model(generator_filename)
        discriminator = tf_load_model(discriminator_filename)
        return GAN(generator, discriminator)

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
    def _train_step(self, X_batch: np.ndarray) -> tuple[float, float]:
        """Runs one batch of images through the model, computes the loss, and
        applies the gradients to the model; returns the generator and
        discriminator losses on the batch.

        :param X_batch: A batch of input images; a tensor of shape m x h x w x
            c, where m is the batch size, h is the image height, w is the image
            width, and c is the number of channels.
        :return: A 2-tuple of the generator and discriminator losses on the
            batch.
        """
        # pylint: disable=invalid-name
        noise = tf.random.normal((len(X_batch), self.gen_input_dim))
        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(X_batch, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            gen_loss = GAN._generator_loss(fake_output)
            dis_loss = GAN._discriminator_loss(real_output, fake_output)
        gen_gradients = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        dis_gradients = dis_tape.gradient(
            dis_loss, self.discriminator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables))
        self.discriminator.optimizer.apply_gradients(
            zip(dis_gradients, self.discriminator.trainable_variables))
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
                self.generator)
            wandb.run.summary['discriminator_graph'] = wandb.Graph.from_keras(
                self.discriminator)
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
                gen_loss_batch, dis_loss_batch = self._train_step(train_batch)
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
                key=lambda gen_epoch: history.history['gen_loss'][gen_epoch])
            best_gen_loss = history.history['gen_loss'][best_gen_epoch]
            best_dis_epoch = min(
                history.history['epoch'],
                key=lambda dis_epoch: history.history['dis_loss'][dis_epoch])
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
            is the number of channels. All values are in the range [0, 1].
        """
        noise = tf.random.normal((num_samples, self.gen_input_dim))
        return self.generator(noise)

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

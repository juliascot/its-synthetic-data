from tensorly.decomposition import parafac
import numpy as np
from tensor_import.tensor import Tensor
import tensorly as tl


def special_sigmoid(input: any) -> any:
    return 1 / (1 + np.exp(-6 * input + 3))


def special_sigmoid_inverse(input: any) -> any:
    return (3 - np.log(1 / input - 1)) / 6


def create_dense_tensor(
        filename: str, rank: int,
        pre_reconstruction_augmentation_offsets: list[int],
        post_reconstruction_augmentation_values: list[float],
        l2: float = 0
        ) -> np.ndarray:
    
    initial_tensor = Tensor(filename, is_student_outside=True, augment_offsets=pre_reconstruction_augmentation_offsets)

    mask = ~np.isnan(initial_tensor.data_tensor)
    initial_tensor.data_tensor = np.nan_to_num(initial_tensor.data_tensor)

    weights, factors = parafac(initial_tensor.data_tensor, rank=rank, mask=mask, l2_reg=l2)
    reconstructed_tensor = tl.kruskal_to_tensor((weights, factors))

    augmented_tensor = np.copy(reconstructed_tensor)
    augmented_tensor = np.vstack([augmented_tensor] + [val * augmented_tensor for val in post_reconstruction_augmentation_values])

    return special_sigmoid(augmented_tensor)


def gradient_penalty(self, batch_size, real_images, fake_images):
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
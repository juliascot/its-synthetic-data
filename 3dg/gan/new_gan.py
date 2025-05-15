from tensor import Tensor
import tensorflow as tf
from keras import layers, models
from tensorly.decomposition import parafac
import numpy as np
import tensorly as tl




def create_dense_tensor(filename: str, rank: int, l2: float) -> np.ndarray:
    initial_tensor = Tensor(filename, is_student_outside=True, is_augmented=True)

    mask = ~np.isnan(initial_tensor.data_tensor)
    initial_tensor.data_tensor = np.nan_to_num(initial_tensor.data_tensor)

    weights, factors = parafac(initial_tensor.data_tensor, rank=rank, mask=mask, l2_reg=l2)
    reconstructed_tensor = tl.kruskal_to_tensor((weights, factors))

    augmented_tensor = np.copy(reconstructed_tensor)
    augmented_tensor = np.vstack((augmented_tensor, augmented_tensor * 0.9, augmented_tensor * 1.1))

    return augmented_tensor


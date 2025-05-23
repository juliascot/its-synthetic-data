from tensorly.decomposition import parafac
import numpy as np
from tensor_import.tensor import Tensor
import tensorly as tl
import torch
import matplotlib.pyplot as plt
from learning_curve_analysis.learning_curve_analysis import extract_prior_and_acquired_knowledge


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


def graph_student_slices(slices: np.ndarray, epochs: int, rank: int, l2: float = 0) -> None:
    
    all_extracted_info = extract_prior_and_acquired_knowledge(slices)

    # Graph the extracted values
    plt.figure()

    for student_num in range(len(slices)):
        plt.scatter(all_extracted_info[student_num-1][0], all_extracted_info[student_num-1][1])

    # plt.title(f'{len(slices)} Students, {epochs} Epochs, Rank {rank}',fontsize=8)
    plt.suptitle(f'{epochs} Epochs, Rank = {rank}, L2 = {l2}',fontsize=16, y=0.97)
    plt.xlabel("$\t{a}$: prior knowledge")
    plt.ylabel("$\t{b}$: learning rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()


def generate_slices(generator, num_slices, noise_dim=100):
    z = torch.randn((num_slices, noise_dim))
    with torch.no_grad():
        fake_slices = generator(z)
    return fake_slices


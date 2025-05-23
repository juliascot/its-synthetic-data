'''
Create graph to visualize the learning rates and prior knowledge of four different students.

This is how I made one specific graph, but it can easily be modified to create others.
'''

import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt
from tensorly.decomposition import parafac
from tensor_import.tensor import Tensor
from learning_curve_analysis.learning_curve_analysis import extract_prior_and_acquired_knowledge

filename = "Getting_Started_Reordered.csv"
rank = 8
l2 = 0


if __name__ == "__main__":
    # Decompose and reconstruct the tensor
    initial_tensor = Tensor(filename, is_student_outside=True)

    mask = ~np.isnan(initial_tensor.data_tensor)
    initial_tensor.data_tensor = np.nan_to_num(initial_tensor.data_tensor)

    weights, factors = parafac(initial_tensor.data_tensor, rank=rank, mask=mask, l2_reg=l2)
    reconstructed_tensor = tl.kruskal_to_tensor((weights, factors))

    all_extracted_info = extract_prior_and_acquired_knowledge(reconstructed_tensor)

    # Graph the extracted values
    plt.figure()

    interesting_students = [12, 24, 30, 27]
    colors = ['gold', 'mediumblue', 'lightgreen', 'red']

    for i, student_num in enumerate(interesting_students):
        plt.scatter(all_extracted_info[student_num-1][0], all_extracted_info[student_num-1][1], color=colors[i])

    plt.title(f'Across all questions, Rank = {rank}, L2 = {l2}',fontsize=8)
    plt.suptitle(f'Four Students\' Learning Curves',fontsize=16, y=0.97)
    plt.xlabel("$\t{a}$: prior knowledge")
    plt.ylabel("$\t{b}$: learning rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()

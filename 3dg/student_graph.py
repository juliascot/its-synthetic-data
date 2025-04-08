'''
Create graph to visualize the learning rates and prior knowledge of four different students.

This is how I made one specific graph, but it can easily be modified to create others.
'''

import numpy as np
import tensorly as tl
import pandas as pd
import matplotlib.pyplot as plt
from tensorly.decomposition import parafac
from scipy.optimize import curve_fit
from tensor import Tensor

filename = "Getting_Started_Processed.csv"
rank = 8
l2 = 0


# Helper function
def power_law(x, a, b):
    return a * np.power(x, b)


if __name__ == "__main__":
    initial_tensor = Tensor(filename, is_student_outside=True)

    mask = ~np.isnan(initial_tensor.data_tensor)
    initial_tensor.data_tensor = np.nan_to_num(initial_tensor.data_tensor)

    weights, factors = parafac(initial_tensor.data_tensor, rank=rank, mask=mask, l2_reg=l2)
    reconstructed_tensor = tl.kruskal_to_tensor((weights, factors))

    # Extract prior knowledge (a) and acquired knowledge (b)
    all_extracted_info = []

    for student_num, student_matrix in enumerate(reconstructed_tensor):

        extracted_info_a = []
        extracted_info_b = []

        for question in student_matrix:

            X = np.arange(1, len(question) + 1)

            popt, pcov = curve_fit(power_law, X, question, p0=[1, 1], bounds=([0, 0], [1, 1]))

            extracted_info_a.append(popt[0])
            extracted_info_b.append(popt[1])

        
        all_extracted_info.append([extracted_info_a, extracted_info_b])


    plt.figure()

    interesting_students = [12, 24, 30, 27]
    colors = ['gold', 'mediumblue', 'lightgreen', 'red']

    for i, student_num in enumerate(interesting_students):
        plt.scatter(all_extracted_info[student_num-1][0], all_extracted_info[student_num-1][1], color=colors[i])

    plt.title('Across all questions',fontsize=8)
    plt.suptitle(f'Four Students\' Learning Curves',fontsize=16, y=0.97)
    plt.xlabel("$\t{a}$: prior knowledge")
    plt.ylabel("$\t{b}$: learning rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()

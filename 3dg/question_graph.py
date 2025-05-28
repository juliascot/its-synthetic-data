import sys
import numpy as np
import tensorly as tl
import pandas as pd
import matplotlib.pyplot as plt
from tensorly.decomposition import parafac
from scipy.special import expit
from scipy.optimize import curve_fit
from internal_imports.tensor import Tensor
from internal_imports.learning_curve_analysis import extract_prior_and_acquired_knowledge

l2 = 0
filename = "Getting_Started_Reordered.csv"
rank = 6



# Helper functions

def power_law(x, a, b):
    return a * np.power(x, b)


# Load dataset into 3D array

data = pd.read_csv(filename)
# print(data.shape)

num_learners = data['Student_Id'].nunique()
num_questions = data['Question_Id'].nunique()
num_attempts = data['Attempt_Count'].nunique()

shaped_data = np.full((num_questions, num_learners, num_attempts), np.nan)

# Fill in with the data points
for row in range(len(data.index) - 2): # Subtract 2 to avoid header and start at 0
    shaped_data[data['Question_Id'][row]-1][data['Student_Id'][row]-1][data['Attempt_Count'][row]-1] = data['Answer_Score'][row]

orig_mask = ~np.isnan(shaped_data)  # True where data is present, False where it is missing
data_tensor = tl.tensor(shaped_data, dtype=tl.float32)
orig_present_points = np.array(np.where(orig_mask)).T




# Use K-fold cross-validation and ALS to factor the tensor for various ranks




# Decompose and reconstruct the tensor
initial_tensor = Tensor(filename, is_student_outside=True)

mask = ~np.isnan(initial_tensor.data_tensor)
initial_tensor.data_tensor = np.nan_to_num(initial_tensor.data_tensor)

weights, factors = parafac(initial_tensor.data_tensor, rank=rank, mask=mask, l2_reg=l2)
reconstructed_tensor = tl.kruskal_to_tensor((weights, factors))

all_extracted_info = extract_prior_and_acquired_knowledge(reconstructed_tensor)
    
for question_number, question in enumerate(all_extracted_info):


    plt.figure()
    plt.scatter(question[0], question[1], label='Data', c="blue")

    plt.suptitle(f'Question {question_number + 1} Learning Curve',fontsize=16, y=0.97)
    plt.title(f'Across all students, Rank = {rank}, L2 = {l2}', fontsize=8)
    plt.xlabel("$\t{a}$: prior knowledge")
    plt.ylabel("$\t{b}$: learning rate")

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()


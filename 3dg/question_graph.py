import sys
import numpy as np
import tensorly as tl
import pandas as pd
import matplotlib.pyplot as plt
from tensorly.decomposition import parafac
from scipy.special import expit
from scipy.optimize import curve_fit

l2 = 0


# Helper functions

def power_law(x, a, b):
    return a * np.power(x, b)


# Load dataset into 3D array

filename = "Getting_Started_Reordered.csv"
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

ranks = range(6, 9)



# Create train tensors
train_tensor = np.copy(data_tensor)

# Optional: assume if student got it right, they get it right every subsequent attempt (rather than empty value)
for question in train_tensor:
    for student in question:
        for attempt_index in range(len(question)):
            if student[attempt_index] == 1:
                student[attempt_index:] = [1 for _ in student[attempt_index:]]
                break


mask = ~np.isnan(train_tensor)
train_tensor = np.nan_to_num(train_tensor)

# Test on different ranks
for rank in ranks:

    weights, factors = parafac(train_tensor, rank=rank, mask=mask, l2_reg=l2)
    reconstructed_tensor = tl.kruskal_to_tensor((weights, factors))

    # Extract prior knowledge (a) and acquired knowledge (b)
    all_extracted_info = []
    all_errors = []

    for question_number, question_matrix in enumerate(reconstructed_tensor):

        extracted_info_a = []
        extracted_info_b = []

        both_extracted = []

        for student in question_matrix:

            X = np.arange(1, len(student) + 1)

            popt, pcov = curve_fit(power_law, X, student, p0=[1, 1], bounds=([0, 0], [1, 1]))

            extracted_info_a.append(popt[0])
            extracted_info_b.append(popt[1])

            both_extracted.append(list(popt))
        


        plt.figure()
        plt.scatter(extracted_info_a, extracted_info_b, label='Data', c="blue")

        plt.suptitle(f'Question {question_number + 1} Learning Curve',fontsize=16, y=0.97)
        plt.title(f'Across all students, Rank = {rank}, L2 = {l2}', fontsize=8)
        plt.xlabel("$\t{a}$: prior knowledge")
        plt.ylabel("$\t{b}$: learning rate")

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()





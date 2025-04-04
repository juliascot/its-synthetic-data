'''
Tensor class. This will take in the csv file and desired rank of the tensor and process it
'''

import numpy as np
import tensorly as tl
import pandas as pd
from tensorly.decomposition import parafac
from sklearn.model_selection import KFold

class Tensor:
    def __init__(self, filename: str) -> None:

        # Load dataset into 3D array
        data = pd.read_csv(filename)

        num_learners = data['Student_Id'].nunique()
        num_questions = data['Question_Id'].nunique()
        num_attempts = data['Attempt_Count'].nunique()

        shaped_data = np.full((num_questions, num_learners, num_attempts), np.nan)

        # Fill in with the data points
        for row in range(len(data.index) - 2): # Subtract 2 to avoid header and start at 0
            shaped_data[data['Question_Id'][row]-1][data['Student_Id'][row]-1][data['Attempt_Count'][row]-1] = data['Answer_Score'][row]

        self.orig_present_points = np.array(np.where(~np.isnan(shaped_data))).T # a 2D array of coordinates of data points that exist [[0 1 0] [3 28 17] ...]

        # Optional: assume that when a student has gotten a question right, they get that question right every subsequent attempt (rather than empty value)
        # We don't include this in the orig_present_points to avoid using these as test data points
        for question in shaped_data:
            for student in question:
                for attempt_index in range(len(student)):
                    if student[attempt_index] == 1:
                        student[attempt_index:] = [1 for _ in student[attempt_index:]]
                        break

        self.data_tensor = tl.tensor(shaped_data, dtype=tl.float32)
'''
Tensor class. This will take in the csv file (and optionally whether students or questions 
are the outside axes)and process it
'''

import numpy as np
import tensorly as tl
import pandas as pd


class Tensor:
    def __init__(self, filename: str, is_student_outside: bool = False, is_augmented: bool = False) -> None:

        # Load dataset into 3D array
        data = pd.read_csv(filename)

        num_learners = data['Student_Id'].nunique()
        num_questions = data['Question_Id'].nunique()
        num_attempts = data['Attempt_Count'].nunique()

        if is_student_outside:
            num_outside = num_learners
            num_middle = num_questions
            outside_id = 'Student_Id'
            middle_id = 'Question_Id'
        else:
            num_outside = num_questions
            num_middle = num_learners
            outside_id = 'Question_Id'
            middle_id = 'Student_Id'

        shaped_data = np.full((num_outside, num_middle, num_attempts), np.nan)

        # Fill in with the data points
        for row in range(len(data.index) - 2): # Subtract 2 to avoid header and start at 0
            shaped_data[data[outside_id][row]-1][data[middle_id][row]-1][data['Attempt_Count'][row]-1] = data['Answer_Score'][row]

        self.orig_present_points = np.array(np.where(~np.isnan(shaped_data))).T # a 2D array of coordinates of data points that exist [[0 1 0] [3 28 17] ...]

        # Optional: assume that when a student has gotten a question right, they get that question right every subsequent attempt (rather than empty value)
        # We don't include this in the orig_present_points to avoid using these as test data points
        for outside in shaped_data:
            for middle in outside:
                for attempt_index in range(len(middle)):
                    if middle[attempt_index] == 1:
                        middle[attempt_index:] = [1 for _ in middle[attempt_index:]]
                        break

        if is_student_outside and is_augmented:
            shaped_data = self.augment(shaped_data)

        self.data_tensor = tl.tensor(shaped_data, dtype=tl.float32)


    def augment(self, orig_data: np.ndarray) -> np.ndarray:
        augmented_data = np.copy(orig_data)

        augmented_data = np.concatenate(augmented_data, self.shift_attempts_earlier(orig_data, 1), axis=0)
        augmented_data = np.concatenate(augmented_data, self.shift_attempts_earlier(orig_data, -1), axis=0)
        augmented_data = np.concatenate(augmented_data, self.shift_attempts_earlier(orig_data, 2), axis=0)
        augmented_data = np.concatenate(augmented_data, self.shift_attempts_earlier(orig_data, -2), axis=0)

        return augmented_data
    
    
    def shift_attempts_earlier(self, orig_data: np.ndarray, shift_amount: int) -> np.ndarray:
        new_data = np.copy(orig_data)

        for student in new_data:
            for question in student:

                if 1 not in question: # continue if there are no correct answers
                    continue

                first_correct = question.index(1)
                if shift_amount > 0: # if positive shift, then insert 1's to the left
                    for attempt_index in range(np.max(first_correct - shift_amount, 0), first_correct):
                        question[attempt_index] = 1
                else: # if negative shift, then insert 0's to the right
                    for attempt_index in range(first_correct, np.min(first_correct - shift_amount, len(question))):
                        question[attempt_index] = 0
    
        return new_data
import numpy as np
import tensorly as tl
import pandas as pd


class Tensor:
    def __init__(self, filename: str, is_student_outside: bool = True, contains_whether_achieved: bool = False) -> None:
        """
        Tensor class. This will take in the csv file, optionally whether students or milestones are the outside axis, 
        and whether we want to include a slice for whether each milestone was achieved.
        It then processes this all and puts it in a tensor.

        """

        # Load dataset into 3D array
        data = pd.read_csv(filename)

        num_students = data['student'].nunique()
        num_milestones = data['milestone'].nunique()

        if is_student_outside:
            num_outside = num_students
            num_middle = num_milestones
            outside_id = 'student'
            middle_id = 'milestone'
        else:
            num_outside = num_milestones
            num_middle = num_students
            outside_id = 'milestone'
            middle_id = 'student'

        shaped_data = np.full((num_outside, num_middle, 3 if contains_whether_achieved else 2), np.nan)


        # Fill in with the data points

        if contains_whether_achieved:
            shaped_data[:, :, 2] = 0

        for row in range(len(data.index)):
            shaped_data[data[outside_id][row]][data[middle_id][row]][0] = data['timestamp'][row]
            shaped_data[data[outside_id][row]][data[middle_id][row]][1] = data['attempt'][row]
            if contains_whether_achieved:
                shaped_data[data[outside_id][row]][data[middle_id][row]][2] = 1


        self.orig_present_points = np.array(np.where(~np.isnan(shaped_data))).T # a 2D array of coordinates of data points that exist [[0 1 0] [3 28 17] ...]
        self.max_time = max(shaped_data[:, :, 0])
        self.data_tensor = tl.tensor(shaped_data, dtype=tl.float32)
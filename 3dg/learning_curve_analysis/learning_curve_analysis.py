import numpy as np
from scipy.optimize import curve_fit


# Helper function
def power_law(x, a, b):
    return a * np.power(x, b)


def extract_prior_and_acquired_knowledge(tensor: np.ndarray) -> list[list[float]]:
    # Extract prior knowledge (a) and acquired knowledge (b)
    all_extracted_info = []

    for student_num, student_matrix in enumerate(tensor):

        extracted_info_a = []
        extracted_info_b = []

        for question in student_matrix:

            X = np.arange(1, len(question) + 1)

            popt, pcov = curve_fit(power_law, X, question, p0=[1, 1], bounds=([0, 0], [1, 1]))

            extracted_info_a.append(popt[0])
            extracted_info_b.append(popt[1])

        
        all_extracted_info.append([extracted_info_a, extracted_info_b])

    return all_extracted_info
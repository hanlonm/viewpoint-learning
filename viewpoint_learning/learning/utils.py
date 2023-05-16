import numpy as np

def normalize(input_arr: np.ndarray):
    mean = np.mean(input_arr)
    std = np.std(input_arr)
    input_arr = (input_arr - mean) / std

    return input_arr, mean, std

def standardize(input_array: np.ndarray, max_value: float):
    input_array = input_array / max_value

    input_array= np.clip(input_array, None, 1.0)

    return input_array


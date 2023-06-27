import numpy as np
from tqdm import tqdm

def normalize(input_arr: np.ndarray):
    mean = np.mean(input_arr)
    std = np.std(input_arr)
    input_arr = (input_arr - mean) / std

    return input_arr, mean, std

def standardize(input_array: np.ndarray, max_value: float):
    input_array = input_array / max_value

    input_array= np.clip(input_array, None, 1.0)

    return input_array

def pre_process(histogram_data):
    ranges = histogram_data[:,:2]
    range_1 = standardize(histogram_data[:, 0], 2000)
    range_2 = standardize(histogram_data[:, 1], 2000)
    min_dist_hist = histogram_data[:,2:12]
    max_dist_hist = histogram_data[:,12:22]
    min_ang_hist =histogram_data[:,22:32]
    max_ang_hist = histogram_data[:,32:42]
    min_ang_diff_hist =histogram_data[:,42:52]
    max_ang_diff_hist =histogram_data[:,52:62]
    heatmaps = histogram_data[:,62:126]
    heatmaps = standardize(histogram_data[:,62:126], 1000)
    px_u_hist=histogram_data[:,126:136]
    px_v_hist=histogram_data[:,136:146]


    histogram_data = np.hstack((np.array([range_1]).T, np.array([range_2]).T, min_dist_hist, max_dist_hist, min_ang_hist, max_ang_hist, min_ang_diff_hist, 
                                max_ang_diff_hist,heatmaps,px_u_hist, px_v_hist))

    # histogram_data = np.hstack((np.array([range_1]).T, np.array([range_2]).T, min_dist_hist, max_dist_hist, min_ang_hist, max_ang_hist,min_ang_diff_hist,max_ang_diff_hist,heatmaps))
    
    return histogram_data

def remove_nan_rows(array):
    nan_rows = np.isnan(array).any(axis=1)
    array = array[~nan_rows]

    return array, nan_rows

def create_dataset(hf, environments, max_error):
    histograms = []
    errors = []
    for environment in environments:
        environment_data = hf[environment]
        print(environment_data.keys())
        histograms.append(environment_data["histogram_data"][:])
        errors.append(environment_data["errors"][:])
        
    histogram_data = np.vstack(histograms)
    histogram_data, nan_rows = remove_nan_rows(histogram_data)
    errors = np.vstack(errors)
    errors = errors[~nan_rows]
    histogram_data = pre_process(histogram_data)

    e_trans = np.array([np.linalg.norm(errors[:,:3], axis=1)]).T
    e_rot = np.array([errors[:,3]]).T

    errors = np.hstack((e_trans, e_rot))
    trans_errors = errors[:,0] 

    trans_errors = np.clip(trans_errors, None, max_error)

    trans_errors = standardize(trans_errors, max_error)
    trans_errors = np.array([trans_errors]).T

    return histogram_data, trans_errors, e_rot

def create_transformer_dataset(hf, environments, max_error):
    inputs = []
    errors = []
    for environment in environments:
        environment_data = hf[environment]
        print(environment_data.keys())
        data = environment_data["token_data"][:]


        inputs.append(data)
        errors.append(environment_data["errors"][:])
        
    token_data = np.vstack(inputs)
    # token_data, nan_rows = remove_nan_rows(token_data)
    errors = np.vstack(errors)
    # errors = errors[~nan_rows]
    # token_data = pre_process(token_data)

    e_trans = np.array([np.linalg.norm(errors[:,:3], axis=1)]).T
    e_rot = np.array([errors[:,3]]).T

    errors = np.hstack((e_trans, e_rot))
    trans_errors = errors[:,0] 

    trans_errors = np.clip(trans_errors, None, max_error)

    trans_errors = standardize(trans_errors, max_error)
    trans_errors = np.array([trans_errors]).T

    return token_data, trans_errors, e_rot


def create_variable_transformer_dataset(hf, environments, max_error):
    inputs = []
    errors = []
    for environment in tqdm(environments):
        environment_data = hf[environment]
        errors.append(environment_data["errors"][:])

    for environment in tqdm(environments):
        group = hf[environment]["token_data"]
        keys = list(group.keys())
        keys.sort(key=int)
        for key in keys:
            inputs.append(group[key][:])
        
    token_data = np.array(inputs, dtype="object")
    # token_data, nan_rows = remove_nan_rows(token_data)
    errors = np.vstack(errors)
    # errors = errors[~nan_rows]
    # token_data = pre_process(token_data)

    e_trans = np.array([np.linalg.norm(errors[:,:3], axis=1)]).T
    e_rot = np.array([errors[:,3]]).T

    errors = np.hstack((e_trans, e_rot))
    trans_errors = errors[:,0] 

    trans_errors = np.clip(trans_errors, None, max_error)

    trans_errors = standardize(trans_errors, max_error)
    trans_errors = np.array([trans_errors]).T

    return token_data, trans_errors, e_rot


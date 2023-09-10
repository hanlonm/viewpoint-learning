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

def replace_nan_rows(array):
    nan_rows = np.isnan(array).any(axis=1)
    array[nan_rows] = np.zeros((1, array.shape[1]))
    return array

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

def create_rank_dataset(hf, environments, max_error, weight_factor=5, num_pos=100, num_viewpoints=50 ,samples_per_point=50):
    histograms = []
    errors = []
    dataset = []
    for environment in environments:
        environment_data = hf[environment]
        print(environment_data.keys())
        histograms.append(environment_data["histogram_data"][:])
        errors.append(environment_data["errors"][:])
        
    histogram_data = np.vstack(histograms)
    histogram_data = replace_nan_rows(histogram_data)
    errors = np.vstack(errors)
    histogram_data = pre_process(histogram_data)

    e_trans = np.array([np.linalg.norm(errors[:,:3], axis=1)]).T
    e_rot = np.array([errors[:,3]]).T

    errors = np.hstack((e_trans, e_rot))
    trans_errors = errors[:,0] 

    trans_errors = np.clip(trans_errors, None, max_error)

    trans_errors = standardize(trans_errors, max_error)

    histogram_data = histogram_data.reshape((len(environments), num_pos, num_viewpoints, -1))
    trans_errors = trans_errors.reshape((len(environments), num_pos, num_viewpoints, -1))

    for env_idx in tqdm(range(histogram_data.shape[0])):
        for point_idx in range(histogram_data.shape[1]):
            hists = histogram_data[env_idx,point_idx]
            errs = trans_errors[env_idx,point_idx]
            
            for _ in range(samples_per_point):
                idx_1, idx_2 = np.random.choice(len(hists), size=2, replace=False)
                
                if errs[idx_1] < errs[idx_2]:
                    label = 1.0
                    diff = errs[idx_2] - errs[idx_1]
                    weight = weight_factor * diff[0]
                    dataset.append((hists[idx_1], hists[idx_2], label, weight))
                    dataset.append((hists[idx_2], hists[idx_1], label-1.0, weight))
                elif errs[idx_1] > errs[idx_2]:
                    label = 0.0
                    diff = errs[idx_1] - errs[idx_2]
                    weight = weight_factor * diff[0]
                    dataset.append((hists[idx_1], hists[idx_2], label, weight))
                    dataset.append((hists[idx_2], hists[idx_1], label+1.0, weight))
                    
                else:
                    label = 0.5
                    diff = 0.1
                    weight = weight_factor * diff
                    dataset.append((hists[idx_1], hists[idx_2], label, weight))
    
    return dataset

def create_rank_trf_dataset(hf, environments, max_error, weight_factor=5,num_pos=100, num_viewpoints=50 ,samples_per_point=50):
    inputs = []
    errors = []
    dataset = []
    target_rows = 1024
    for environment in environments:
        environment_data = hf[environment]
        print(environment_data.keys())
        group = hf[environment]["token_data"]
        keys = list(group.keys())
        keys.sort(key=int)
        for key in keys:
            token = group[key][:]
            token = token[:,:9]
            if token.shape[0] > target_rows:
                random_indices = np.random.choice(token.shape[0], target_rows, replace=False)
                token = token[random_indices]
            elif token.shape[0] < target_rows:
                random_indices = np.random.choice(token.shape[0], target_rows, replace=True)
                token = token[random_indices]
            inputs.append(token)
        errors.append(environment_data["errors"][:])
        
    token_data = np.array(inputs)
    errors = np.vstack(errors)

    e_trans = np.array([np.linalg.norm(errors[:,:3], axis=1)]).T
    e_rot = np.array([errors[:,3]]).T

    errors = np.hstack((e_trans, e_rot))
    trans_errors = errors[:,0] 

    trans_errors = np.clip(trans_errors, None, max_error)

    trans_errors = standardize(trans_errors, max_error)

    token_data = token_data.reshape((len(environments), num_pos, num_viewpoints, target_rows, -1))
    trans_errors = trans_errors.reshape((len(environments), num_pos, num_viewpoints, -1))

    for env_idx in tqdm(range(token_data.shape[0])):
        for point_idx in range(token_data.shape[1]):
            toks = token_data[env_idx,point_idx]
            errs = trans_errors[env_idx,point_idx]
            
            for _ in range(samples_per_point):
                idx_1, idx_2 = np.random.choice(len(toks), size=2, replace=False)
                
                if errs[idx_1] < errs[idx_2]:
                    label = 1.0
                    diff = errs[idx_2] - errs[idx_1]
                    weight = weight_factor * diff[0]
                    dataset.append((toks[idx_1], toks[idx_2], label, weight))
                    dataset.append((toks[idx_2], toks[idx_1], label-1.0, weight))
                elif errs[idx_1] > errs[idx_2]:
                    label = 0.0
                    diff = errs[idx_1] - errs[idx_2]
                    weight = weight_factor * diff[0]
                    dataset.append((toks[idx_1], toks[idx_2], label, weight))
                    dataset.append((toks[idx_2], toks[idx_1], label+1.0, weight))
                    
                else:
                    label = 0.5
                    diff = 0.1
                    weight = weight_factor * diff
                    dataset.append((toks[idx_1], toks[idx_2], label, weight))
                    
    
    return dataset


            


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
            token = group[key][:]
            token = token[:,:73]
            inputs.append(token)
        
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


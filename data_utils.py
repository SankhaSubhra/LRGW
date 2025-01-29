import os
import json
import random
import numpy as np

def gather_data_files(run_config):

    data_path_dict = {}
    for file in os.listdir(run_config['data_base_path']):
        # Two .npy files corresponding to two datasets are expected to be in the folder.
        if file.endswith(".npy"):
            data_name = file[:-4]
            data_path_dict[data_name] = os.path.join(run_config['data_base_path'], file)
        # One JSON file with a common noise config is expected to be in the folder.
        elif file.endswith(".json"):
            noise_config_path = os.path.join(run_config['data_base_path'], file)

    return data_path_dict, noise_config_path


def data_load(data_path_dict):

    # Loads and process the datasets.
    # :param: data_path_dict: dict[string, string]: dict of data file names in npy format.
    # Each data file contains a matrix of points sampled from an image. 
    # :returns: [np.array]: A set of point clouds.
    # :returns: int: The number of point clouds. 

    data_dict = {}
    for key, value in data_path_dict.items():
        data_dict[key] = np.load(value).astype(float)

    data_dict, number_of_samples = data_process(data_dict)
    return data_dict, number_of_samples

def data_process(data_sets):

    # Ensures that all datasets have equal number of samples.
    # :param: data_sets: dict[ string, np.array(ns, ndim)]: Matrix of ns samples in ndim dimensions.
    # :number_of_data_sets: int: Length of data_sets.
    # :returns: data_sets: [np.array(ns, ndim)]: Sampled datasets all havings number_of_samples instances.
    # :returns: number_of_samples: int: Common number of samples for the sampled datasets.

    number_of_data_samples = [len(value) for _, value in data_sets.items()]
    number_of_samples = min(number_of_data_samples)

    for key, value in data_sets.items():
        if len(value) > number_of_samples:
            random_index = np.sort(random.permutation(np.arange(len(value)))[:number_of_samples])
            data_sets[key] = value[random_index, :]

    return data_sets, number_of_samples

def free_form_outlier_perturbation(X, 
                                   num_outliers, 
                                   outlier_mode=None,
                                   data_json_config=None,
                                   normal_scale=0.01,
                                   normal_outlier_scale=0.2,
                                   cauchy_outlier_scale=0.1,
                                   outside_outlier_scale=10,
                                   cauchy_data_deviation_scale=5, 
                                   outside_data_deviation_scale=1.25):
    
    # Perturb samples, creating free-form outliers randomly distributed across the data range.
    # :param: X: np.array(ns, ndim): Dataset of ns samples each of ndim dimensions.
    # :param: num_outliers: int: Number of total outliers. 
    # :param: outlier_mode: str: Outlier distribution, can be Gaussian, Cauchy, or Outside.
    # :param: data_json_config: str: JSON config for noise addition. 
    # :param: normal_scale: float: Std of normal distribution for mild noise..
    # :param: normal_outlier_scale: float: Spread of Gaussian outliers.
    # :param: cauchy_outlier_scale: float: Spread of Cauchy outliers.
    # :param: outside_outlier_scal: float: Spread of outside Gaussian outliers.
    # :param: cauchy_data_deviation_scale: float: Extend of data deviation from Noisefree max/min.
    # :param: outside_data_deviation_scale: float: Extend of data deviation from Noisefree max/min.
    # :returns: X_perturbed: np.array(ns, ndim): Perturbed dataset.
    
    if data_json_config is not None:
        with open(data_json_config, 'r') as file:
            config = json.load(file)
        normal_scale = config['normal_scale']
        normal_outlier_scale = config['normal_outlier_scale']
        cauchy_outlier_scale = config['cauchy_outlier_scale']
        outside_outlier_scale = config['outside_outlier_scale']
        cauchy_data_deviation_scale = config['cauchy_data_deviation_scale']
        outside_data_deviation_scale = config['outside_data_deviation_scale']

    X_perturbed = X.copy()
    n_samples, n_features = X.shape
    
    # Generate mild Gaussian noise for all samples
    gaussian_noise = np.random.normal(0, normal_scale, X.shape)
    X_perturbed += gaussian_noise
    
    # Randomly select indices to perturb as outliers
    outlier_indices = np.random.choice(n_samples, num_outliers, replace=False)
    
    # Determine the data range for each feature
    data_min = np.min(X, axis=0)
    data_max = np.max(X, axis=0)
    data_range = data_max - data_min
    data_center = (data_max + data_min) / 2
    
    if outlier_mode == 'Gaussian':
        # Generate outliers randomly across the data space
        for idx in outlier_indices:
            # Place each outlier freely across an extended range of the dataset
            random_position = np.random.uniform(data_min - normal_outlier_scale * data_range, 
                                                data_max + normal_outlier_scale * data_range, 
                                                n_features)
            X_perturbed[idx] = random_position
    
    elif outlier_mode == 'Cauchy':
        for idx in outlier_indices:
            cauchy_noise = np.random.standard_cauchy(n_features)
            scaled_noise = cauchy_noise * cauchy_outlier_scale * data_range
            initial_point = data_center + scaled_noise
            extended_data_max = data_max * cauchy_data_deviation_scale
            extended_data_min = data_min * cauchy_data_deviation_scale
            # corrections for extreme shift in dimensions
            for d in range(n_features):
                while initial_point[d] > extended_data_max[d] or initial_point[d] < extended_data_min[d]:
                    initial_point[d] = data_center[d] + np.random.standard_cauchy() * cauchy_outlier_scale * data_range[d]
            X_perturbed[idx] = initial_point
    
    elif outlier_mode == 'Outside':
        for idx in outlier_indices:
            extended_data_max = data_max * outside_data_deviation_scale
            random_position = np.random.normal(extended_data_max, outside_outlier_scale, X_perturbed[idx].shape)
            X_perturbed[idx] = random_position

    else:
        raise NotImplementedError
    
    return X_perturbed

def generate_noisy_samples(data, 
                           number_of_samples, 
                           noise_levels, 
                           noise_types, 
                           noise_config=None):

    # Calls the free-form outlier creator to generate the noisy datasets.
    # :param: data: np.array(ns, ndim): Dataset of ns samples of ndim dimensions.
    # :param: number_of_samples: int: Number of samples in each dataset.
    # :param: noise_levels: [str]: List of number of outliers.
    # :param: noise_types: [str]: List of outlier types.
    # :param: noise_config: str: Path to config JSON file.
    # :returns: noisy_datasets: [np.array(ns, ndim)]: All datasets including noisy and real ones.
    # :returns: num_noise_levels: int: Number of outlier variants.
    # :returns: num_noise_types: int: Number of the type of noise addition.

    noise_levels_actual = {}
    for noise_level in noise_levels:
        if noise_level == '5':
            noise_levels_actual[noise_level] = (int(number_of_samples*0.05))
        elif noise_level == '10':
             noise_levels_actual[noise_level] = (int(number_of_samples*0.10))
        elif noise_level == '15':
            noise_levels_actual[noise_level] = (int(number_of_samples*0.15))
        elif noise_level == '20':
            noise_levels_actual[noise_level] = (int(number_of_samples*0.20))
        elif noise_level == '25':
            noise_levels_actual[noise_level] = (int(number_of_samples*0.25))
        else:
            raise NotImplementedError
        
    noisy_datasets = {}
    noisy_datasets['Noisefree'] = data
    for noise_type in noise_types:
        _noisy_datasets = {}
        for noise_level in noise_levels:   
            _noisy_datasets[noise_level] = free_form_outlier_perturbation(data, 
                                                                 num_outliers=noise_levels_actual[noise_level], 
                                                                 outlier_mode=noise_type,
                                                                 data_json_config=noise_config)
        noisy_datasets[noise_type] = _noisy_datasets
    return noisy_datasets
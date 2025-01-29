import os
import json
import numpy as np
import scipy as sp
import ot
import copy
import data_utils
import plot_utils
import _utils
from scipy.spatial.distance import cdist
from GWComputationsClass import GWComputations

def config_settings(output_folder_name="output"):

    # Initializer for the run.
    # :param: output_folder_name: str: folder name to store output, deffault 'output'.
    # :returns: dict: dictionary containing run configeration.
    # Edit here accordingly to pass the parameters.

    # Please update here according to your choice.
    run_config = {
        'summary_plotting_only': False,
        'data_base_path': "data_npy",
        'noise_levels': ['5', '10', '15', '20', '25'],
        'noise_types': ['Gaussian', 'Cauchy', 'Outside'],
        'prevention_choice_list': ['95P', '98P', '3LT'],
        'number_of_runs': 20,
        'all_output_paths': _utils.path_creation(output_folder_name),
        'result_load_path': None}
    
    return run_config

def distance_metric_preperation(X, prevention_choices_list=None):

    C = sp.spatial.distance.cdist(X, X, metric='euclidean')

    if prevention_choices_list is not None:
        cutoff_values = {}
        for choice in prevention_choices_list:
            if choice == '90P':
                cutoff_values[choice] = np.percentile(C, 90)
            elif choice == '95P':
                cutoff_values[choice] = np.percentile(C, 95)
            elif choice == '98P':
                cutoff_values[choice] = np.percentile(C, 98)
            elif choice == '3LT':
                median_dist = np.median(C)
                mad = np.mean(np.abs(C - median_dist))
                cutoff_values[choice] = median_dist + 3 * mad
            else:
                raise NotImplementedError
            
        cutoff_C = {}
        for key, value in cutoff_values.items():
            _C = copy.deepcopy(C)
            _C[_C > value] = value
            _C = _C / _C.max()
            cutoff_C[key] = _C
        
        processed_C = {'pdmats': cutoff_C, 'cutoffs': cutoff_values, 'dmat': C}

    else: 
        processed_C = {'pdmats': None, 'cutoffs': None, 'dmat': C}

    return processed_C

def gw_runner(PC1, PC2, C1, C2, p, q, M, joint_keys, result):

    gw_object = GWComputations(PC1, PC2, C1, C2, p, q, M)
    gw_object.distance_computation_run()
    if joint_keys not in list(result.keys()):
        result[joint_keys] = [gw_object.cost_dict]
    else:
        result[joint_keys].append(gw_object.cost_dict)
    
    return result, gw_object.cost_dict

def only_lrgw_runner(PC1, PC2, C1, C2, p, q, M, joint_keys, result, past_gw_object_cost_dict):

    gw_object = GWComputations(PC1, PC2, C1, C2, p, q, M)
    key, loss = gw_object.compute_lrgw_distance()
    _past_gw_object_cost_dict = copy.deepcopy(past_gw_object_cost_dict)
    _past_gw_object_cost_dict[key] = loss

    print('------------- Distances -------------')
    print("GW: " + str(round(_past_gw_object_cost_dict['gw'], 5)))
    print("UGW: " + str(round(_past_gw_object_cost_dict['ugw'], 5)))
    print("FGW: " + str(round(_past_gw_object_cost_dict['fgw'], 5)))
    print("PGW: " + str(round(_past_gw_object_cost_dict['pgw'], 5)))
    print("LRGW (Ours): " + str(round(_past_gw_object_cost_dict['lrgw'], 5)))
    
    if joint_keys not in list(result.keys()):
        result[joint_keys] = [_past_gw_object_cost_dict]
    else:
        result[joint_keys].append(_past_gw_object_cost_dict)
    
    return result

def gw_barycenter_runner_and_plotter(X1, X2, PC1, PC2, C1, C2, p, q, joint_keys, output_path, steps=5):
    
    gw_object = GWComputations(PC1, PC2, C1, C2, p, q)
    gw_object.barycenter_computation_run()

    npos_gw = [_utils.smacof_mds(gw_object.barycenters['gw'][s], 2) for s in range(steps-1)]
    npos_lrgw = [_utils.smacof_mds(gw_object.barycenters['lrgw'][s], 2) for s in range(steps-1)]

    plot_utils.plot_barycenters(X1, X2, npos_gw, npos_lrgw, steps, joint_keys, output_path)

def check_completion(completed_list, source_list, target_list):

    for run_name in completed_list:
        if run_name == source_list + target_list or run_name == target_list + source_list:
            # Already completed in one direction.
            # GW is a metric thus the reverse distance calculation is not needed. 
            return True
        
        # Switch the noise type and check if one side is already performed.
        _source_list = copy.deepcopy(source_list)
        _target_list = copy.deepcopy(target_list)
        _source_list[1] = target_list[1]
        _target_list[1] = source_list[1]

        # Example: abc and def means abc -> def equals to
        # def -> abc switch data and noise type with order
        # aec -> dbf switch noise type
        # dbf -> aec switch noise type and order
        if run_name == _source_list + _target_list or run_name == _target_list + _source_list:
            return True
    return False

def run(data_sets, number_of_samples, run_config, noise_config_path):

    result = {}
    noisy_datasets = {}
    p, q = ot.unif(number_of_samples), ot.unif(number_of_samples)

    b_counter = 0
    for current_run in range(run_config['number_of_runs']):
        total_counter = 0
        print("------------- Current Run " + str(current_run) + "/" + str(run_config['number_of_runs']) + " -------------")
        distance_matrix_dict = {}
        for key, value in data_sets.items():
            noisy_datasets[key] = data_utils.generate_noisy_samples(value,
                                number_of_samples,
                                run_config['noise_levels'], 
                                run_config['noise_types'],
                                noise_config_path)

            if current_run == 0:
                plot_utils.plot_noisy_samples(noisy_datasets[key], 
                    run_config['noise_levels'], 
                    run_config['noise_types'], 
                    data_name=key,
                    output_path=run_config['all_output_paths']['noisy_dataset_path'])

            distance_matrix_dict[(key, 'Noisefree', '0')] = distance_metric_preperation(
                        noisy_datasets[key]['Noisefree'])
            
            for noise_type in run_config['noise_types']:
                for noise_level in run_config['noise_levels']:
                    
                    distance_matrix_dict[(key, noise_type, noise_level)] = distance_metric_preperation(
                        noisy_datasets[key][noise_type][noise_level],
                        run_config['prevention_choice_list'])
                    
                    if current_run == 0:
                        heatmap_name = key + '_' + noise_level + '_' + noise_type + '_distance_heatmap' '.pdf'   
                        plot_utils.plot_heat_maps(distance_matrix_dict[(key, noise_type, noise_level)], 
                                    run_config['prevention_choice_list'], 
                                    run_config['all_output_paths']['heatmap_save_path'],
                                    heatmap_name)
                        distance_hist_name = key + '_' + noise_level + '_' + noise_type + '_distance_frequency' '.pdf'
                        plot_utils.plot_histogram(distance_matrix_dict[(key, noise_type, noise_level)],
                                    run_config['all_output_paths']['distance_mat_cutoffs_save_path'],
                                    distance_hist_name)

        distance_matrix_dict_keys = list(distance_matrix_dict.keys())
        completed_list = []
        for source in distance_matrix_dict_keys:
            for target in distance_matrix_dict_keys:
                if source[0] != target[0]:
                    
                    source_list = list(source)
                    target_list = list(target)

                    C1 = distance_matrix_dict[source]['dmat']
                    C2 = distance_matrix_dict[target]['dmat']
                    
                    M = cdist(data_sets[source_list[0]], data_sets[target_list[0]], metric='euclidean')
                    M = M / M.max()

                    if check_completion(completed_list, source_list, target_list) is False:
                        completed_list.append(source_list+target_list)
                            
                        if 'Noisefree' not in source_list and 'Noisefree' not in target_list:
                            # Edit this conditions as per your choice.
                            # Only LRGW will change over different prevision meeasures. 
                            # Calculate other disanctes only once.
                            local_counter = 0
                            for source_prevention, source_processed in distance_matrix_dict[source]['pdmats'].items():
                                PC1 = source_processed
                                for target_prevention, target_processed in distance_matrix_dict[target]['pdmats'].items():
                                    PC2 = target_processed                            
                                    joint_keys = "_".join(tuple((source_list + target_list + [source_prevention] + [target_prevention])))
                                    print('Data pair: ' + joint_keys)
                                    total_counter = total_counter + 1
                                    # Limit the number of costly barycenter computation
                                    if current_run == 0 and source_prevention == target_prevention:
                                        # Only consider two noise levels for now.
                                        if source_list[2] in ['25', '15'] and target_list[2] in ['25', '15']:
                                            if source_list[2] == target_list[2]:
                                                b_counter = b_counter + 1
                                                gw_barycenter_runner_and_plotter(noisy_datasets[source_list[0]][source_list[1]][source_list[2]], 
                                                    noisy_datasets[target_list[0]][target_list[1]][target_list[2]],
                                                    PC1, PC2, C1, C2, p, q, joint_keys, run_config['all_output_paths']['gw_barycenter_save_path'])
                                    if local_counter == 0:
                                        # Calculate all the distances.
                                        result , past_cost_dict = gw_runner(PC1, PC2, C1, C2, p, q, M, joint_keys, result)
                                    else:
                                        # Calculate only LRGW and reuse previously computed distances.
                                        result = only_lrgw_runner(PC1, PC2, C1, C2, p, q, M, joint_keys, result, past_cost_dict)
                                    local_counter = local_counter + 1


    with open(os.path.join(run_config['all_output_paths']['gw_metrics_save_path'], "results.json"), "w") as fp:
        json.dump(result, fp, indent = 4)

    print(total_counter, b_counter)

    return result

def result_process(result, run_config):

    # helper code for results processing.
    mean_result = {}
    max_tracker = 0
    for key, value in result.items():
        gw_distances = []
        ugw_distances = []
        pgw_distances = []
        fgw_distances = []
        lrgw_distances = []
        for idx in range(run_config['number_of_runs']):
            gw_distances.append(value[idx]['gw'])
            ugw_distances.append(value[idx]['ugw'])
            pgw_distances.append(value[idx]['pgw'])
            fgw_distances.append(value[idx]['fgw'])
            lrgw_distances.append(value[idx]['lrgw'])

        mean_result[key] = {'gw_mean': np.mean(np.array(gw_distances)),
                            'gw_std': np.std(np.array(gw_distances)),
                            'ugw_mean': np.mean(np.array(ugw_distances)),
                            'ugw_std': np.std(np.array(ugw_distances)),
                            'pgw_mean': np.mean(np.array(pgw_distances)),
                            'pgw_std': np.std(np.array(pgw_distances)),
                            'fgw_mean': np.mean(np.array(fgw_distances)),
                            'fgw_std': np.std(np.array(fgw_distances)),
                            'lrgw_mean': np.mean(np.array(lrgw_distances)),
                            'lrgw_std': np.std(np.array(lrgw_distances))}
        
        max_tracker = max(max_tracker, 
                          mean_result[key]['gw_mean'],
                          mean_result[key]['ugw_mean'],
                          mean_result[key]['pgw_mean'],
                          mean_result[key]['fgw_mean'],
                          mean_result[key]['lrgw_mean'])
        
    print("All possible maximum distance is: " + str(max_tracker))
    _mean_result = {}
    for key, value in mean_result.items():
        _mean_result[key] = {'gw_mean': mean_result[key]['gw_mean']/max_tracker,
                            'ugw_mean': mean_result[key]['ugw_mean']/max_tracker,
                            'pgw_mean': mean_result[key]['pgw_mean']/max_tracker,
                            'fgw_mean': mean_result[key]['fgw_mean']/max_tracker,
                            'lrgw_mean': mean_result[key]['lrgw_mean']/max_tracker}
        
    with open(os.path.join(run_config['all_output_paths']['gw_metrics_summary_save_path'], "summary.json"), "w") as fp:
        json.dump(mean_result, fp, indent = 4)

    with open(os.path.join(run_config['all_output_paths']['gw_metrics_summary_save_path'], "norm_summary.json"), "w") as fp:
        json.dump(_mean_result, fp, indent = 4)

    plot_utils.plot_heatmaps_of_distances(_mean_result, run_config)

    plot_utils.plot_error_bars(mean_result, run_config)

def main():

    # Initialize config.
    # If you have a output directory in mind please pass it instead of the default.
    run_config = config_settings()
    data_path_dict, noise_config_path = data_utils.gather_data_files(run_config)
    data_sets, number_of_samples = data_utils.data_load(data_path_dict)

    # Obtain the results.
    if run_config['summary_plotting_only'] is False:
        result = run(data_sets, number_of_samples, run_config, noise_config_path)
    
    # Once you have the results please run the code again with necessary config update for analysis/
    elif run_config['summary_plotting_only'] is True:
        with open(run_config['result_load_path']) as fp:
            result = json.load(fp)

        result_process(result, run_config)

    
if __name__ == "__main__":
    main()
    
    

    


import os
import copy
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio   
pio.kaleido.scope.mathjax = None

import seaborn as sns

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times"})

def plot_noisy_samples(noisy_datasets, 
                                noise_levels, 
                                noise_types,
                                data_name=None,
                                output_path=None):
    
    # :param: noisy_datasets: [np.array]: List of datasets including the Noisefree one at the first.
    # :param: noise_levels: [str]: Number of outlier variants.
    # :param: noise_types: [str]: Type of noise additions.
    # :param: data_name: str: Name of the dataest, default None.
    # :param: output_path: str: Name of the output folder, default None.

    titile_format = {'fontsize': 10}

    plt.scatter(noisy_datasets['Noisefree'][:, 0], 
                             noisy_datasets['Noisefree'][:, 1], 
                             color="b",
                             s=0.20)
    plt.grid()
    plt.title("Noisefree dataset", fontdict=titile_format)
    plt.savefig(os.path.join(output_path, ('Noisefree_visualization_' + data_name + '.pdf')))
    plt.close()
    
    titile_format = {'fontsize': 10}
    fig, ax = plt.subplots(len(noise_types), len(noise_levels), figsize=(12, 6))
    for i, noise_type in enumerate(noise_types):
        for j, noise_level in enumerate(noise_levels):
            ax[i][j].grid(True)
            ax[i][j].scatter(noisy_datasets[noise_type][noise_level][:, 0], 
                             noisy_datasets[noise_type][noise_level][:, 1], 
                             color="b",
                             s=0.20)
            ax[i][j].set_title(noise_level + "$\%$ " + noise_type + " Outliers",
                               fontdict=titile_format)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, ('noisy_visualization_' + data_name + '.pdf')))
    plt.close()

def plot_heat_maps(processed_C, prevention_choices, heatmap_save_path, heat_map_name):

    # Plot heatmap of distance matrix after applying cut-off.
    titile_format = {'fontsize': 7}
    fig, ax = plt.subplots(1, len(prevention_choices) + 1)

    ax[0].grid(False)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].imshow(processed_C['dmat'] / processed_C['dmat'].max())
    ax[0].set_title("Noisefree", fontdict=titile_format)
    for i in range(len(prevention_choices)):
        ax[i+1].grid(False)
        ax[i+1].set_xticks([])
        ax[i+1].set_yticks([])
        ax[i+1].imshow(processed_C['pdmats'][prevention_choices[i]])
        ax[i+1].set_title(prevention_choices[i], fontdict=titile_format)
    
    plt.tight_layout()
    plt.savefig(os.path.join(heatmap_save_path, heat_map_name))
    plt.close()

def plot_histogram(processed_C, distance_mat_cutoffs_save_path, distance_hist_name):

    # Plot the histogram of data distance matrix.
    # Plot histogram with KDE
    utm = processed_C['dmat'][np.triu_indices(processed_C['dmat'].shape[0], k = 0)]
    sns.histplot(utm, stat='density', kde=True, color='blue', alpha=0.3)

    # Plot threshold lines
    thresholds = {
        # Ignore 90P for now, that is not proving that much effective.
        # '90': (processed_C['cutoffs']['90P'], '90th percentile'),
        '95': (processed_C['cutoffs']['95P'], '95th percentile'),
        '98': (processed_C['cutoffs']['98P'], '98th percentile'),
        '3lt': (processed_C['cutoffs']['3LT'], 'Median + 3*MAD')}
    
    colors = ['red', 'blue', 'green', 'purple']
    for (key, (value, label)), color in zip(thresholds.items(), colors):
        linestyle = '--'
        plt.axvline(x=value, color=color, linestyle=linestyle,
                    label=f'{label} = {value:.3f}')
    
    plt.xlabel('Distance Distortion $d_X(x,x\')$')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Set x-axis to start at 0 since we're using absolute distortions
    plt.xlim(left=0)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(distance_mat_cutoffs_save_path, distance_hist_name))
    plt.close()

def plot_barycenters(X1, X2, npos_gw, npos_lrgw, steps, joint_keys, output_path):

    # Plot the barycenters. 
    titile_format = {'fontsize': 10}
    fig, ax = plt.subplots(2, steps+1, figsize=(10, 4))

    for i in range(2):
        for j in range(steps+1):
            if j == 0:
                D = X1
                color = 'b'
                title_string = 'Source t=' + str(j) + "/" + str(steps)
            elif j == steps:
                D = X2
                color = 'r'
                title_string = 'Target t=' + str(j) + "/" + str(steps)
            elif j < steps and j > 0:
                color = 'm'
                if i == 0:
                    D = npos_gw[steps - j -1]
                    title_string = 'GW t=' + str(j) + "/" + str(steps)
                elif i == 1:
                    D = npos_lrgw[steps - j -1]
                    title_string = 'LRGW t=' + str(j) + "/" + str(steps)

            ax[i][j].grid(True)
            ax[i][j].scatter(D[:, 0], D[:, 1], color=color, s=0.20)
            ax[i][j].set_title(title_string,
                            fontdict=titile_format)

    output_file = joint_keys + '.pdf'
    save_path = os.path.join(output_path, output_file)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def key_process(key):

    # Helper function to identify the component of a key of the result.
    combinations = []
    _key = copy.deepcopy(key)
    splitted_key = _key.split('_')
    source_data = splitted_key[0] + '_' + splitted_key[1]
    source_noise_type = splitted_key[2]
    source_noise_level = splitted_key[3]
    target_data = splitted_key[4] + '_' + splitted_key[5]
    target_noise_type = splitted_key[6]
    target_noise_level = splitted_key[7]
    source_prevention = splitted_key[8]
    target_prevention = splitted_key[9]

    combinations.append(tuple([source_data, source_noise_type, source_noise_level,
                          target_data, target_noise_type, target_noise_level, 
                          source_prevention, target_prevention]))

    return combinations

def tag2map(tag, noise_levels, preventions):

    # Helper function to identify a run.
    _tag = copy.deepcopy(tag)
    splitted_tag = _tag.split("_")
    source_noise_level = splitted_tag[3]
    target_noise_level = splitted_tag[7]
    source_prevention = splitted_tag[8]
    target_prevention = splitted_tag[9]
    row = (len(preventions)*(noise_levels.index(source_noise_level))) + preventions.index(source_prevention)
    col = (len(preventions)*(noise_levels.index(target_noise_level))) + preventions.index(target_prevention)

    return row, col

def _plot_heatmap_of_distances_plot_helper(res_dict, run_config, tag_1, tag_2, save_name):

    # Heatmap of ditances. 
    titile_format_1 = {'fontsize': 10}
    titile_format_2 = {'fontsize': 14}
    fig, ax = plt.subplots(1, 5, figsize=(22, 5))

    tilte_list = ['GW', 'UGW', 'PGW', 'FGW', 'LRGW (Ours)']
    key_list = ['gw', 'ugw', 'pgw', 'fgw', 'lrgw']
    axis_ticks_lrgw = ['5\% 95P', '5\% 98P', '5\% 3LT',
                  '10\% 95P', '10\% 98P', '10\% 3LT',
                  '15\% 95P', '15\% 98P', '15\% 3LT',
                  '20\% 95P', '20\% 98P', '20\% 3LT',
                  '25\% 95P', '25\% 98P', '25\% 3LT']
    axis_ticks_regular = [' ', '5\%', ' ',
                          ' ', '10%', ' ',
                          ' ', '15\%', ' ',
                          ' ', '20\%', ' ',
                          ' ', '25\%', ' ']

    for i in range(5):
        ax[i].grid(False)
        im = ax[i].pcolormesh(res_dict[key_list[i]], cmap='viridis', edgecolors='white', linewidth=1)
        plt.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)
        if i == 4:
            ax[i].set_xticks(list(range(15)), labels=axis_ticks_lrgw, rotation=90, fontdict=titile_format_1)
            ax[i].set_yticks(list(range(15)), labels=axis_ticks_lrgw, fontdict=titile_format_1)
        else:
            ax[i].set_xticks(list(range(15)), labels=axis_ticks_regular, fontdict=titile_format_1)
            ax[i].set_yticks(list(range(15)), labels=axis_ticks_regular, fontdict=titile_format_1)
        ax[i].set_xlabel(tag_1, fontdict=titile_format_2)
        ax[i].set_ylabel(tag_2, fontdict=titile_format_2)
        ax[i].set_title(tilte_list[i], fontdict=titile_format_2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_config['all_output_paths']['gw_metrics_summary_save_path'], save_name))
    plt.close()


def plot_heatmaps_of_distances(norm_mean_result, run_config):

    # Plot the mean distances as heatmap for all noise strength and prevention methods. 
    cauchy_outside_plot_dict = {'gw': np.zeros((15, 15)),
                                'ugw': np.zeros((15, 15)),
                                'pgw': np.zeros((15, 15)),
                                'fgw': np.zeros((15, 15)),
                                'lrgw': np.zeros((15, 15))}
    
    # Enable the following as per required. 
    # cauchy_cauchy_plot_dict = copy.deepcopy(cauchy_outside_plot_dict)
    # outside_outside_plot_dict = copy.deepcopy(cauchy_outside_plot_dict)

    gaussian_gaussian_plot_dict = copy.deepcopy(cauchy_outside_plot_dict)

    completed_key = []
    for key, value in norm_mean_result.items():

        if key not in completed_key:
            completed_key.append(key)
            combinations = key_process(key)

            if combinations[0][1] == combinations[0][4]:

                _key = "_".join(combinations[0])
                row, col = tag2map(_key, run_config['noise_levels'], run_config['prevention_choice_list'])
                gaussian_gaussian_plot_dict['gw'][row][col] = value['gw_mean']
                gaussian_gaussian_plot_dict['ugw'][row][col] = value['ugw_mean']
                gaussian_gaussian_plot_dict['pgw'][row][col] = value['pgw_mean']
                gaussian_gaussian_plot_dict['fgw'][row][col] = value['fgw_mean']
                gaussian_gaussian_plot_dict['lrgw'][row][col] = value['lrgw_mean']
                
    _plot_heatmap_of_distances_plot_helper(gaussian_gaussian_plot_dict, run_config, 'Gaussian', 'Gaussian', 'gaussian_gaussian_distances.pdf')
    
    # Enable this as per requirement.
    # For Cauchy in source and Outside in target.
    # completed_key = []
    # for key, value in norm_mean_result.items():

    #     if key not in completed_key:
    #         completed_key.append(key)
    #         combinations = key_process(key)

    #         if combinations[0][1] != combinations[0][4]:

    #             _key = "_".join(combinations[0])
    #             row, col = tag2map(_key, run_config['noise_levels'], run_config['prevention_choice_list'])
    #             cauchy_outside_plot_dict['gw'][row][col] = value['gw_mean']
    #             cauchy_outside_plot_dict['ugw'][row][col] = value['ugw_mean']
    #             cauchy_outside_plot_dict['pgw'][row][col] = value['pgw_mean']
    #             cauchy_outside_plot_dict['fgw'][row][col] = value['fgw_mean']
    #             cauchy_outside_plot_dict['lrgw'][row][col] = value['lrgw_mean']
                
    # _plot_heatmap_of_distances_plot_helper(cauchy_outside_plot_dict, run_config, 'Cauchy', 'Outside', 'cauchy_outside_distances.pdf')

    # For Cauchy in both source and target.
    # completed_key = []
    # for key, value in norm_mean_result.items():

    #     if key not in completed_key:
    #         completed_key.append(key)
    #         combinations = key_process(key)

    #         if combinations[0][1] == combinations[0][4] and combinations[0][1] == 'Cauchy':

    #             _key = "_".join(combinations[0])
    #             row, col = tag2map(_key, run_config['noise_levels'], run_config['prevention_choice_list'])
    #             cauchy_cauchy_plot_dict['gw'][row][col] = value['gw_mean']
    #             cauchy_cauchy_plot_dict['ugw'][row][col] = value['ugw_mean']
    #             cauchy_cauchy_plot_dict['pgw'][row][col] = value['pgw_mean']
    #             cauchy_cauchy_plot_dict['fgw'][row][col] = value['fgw_mean']
    #             cauchy_cauchy_plot_dict['lrgw'][row][col] = value['lrgw_mean']
                
    # _plot_heatmap_of_distances_plot_helper(cauchy_cauchy_plot_dict, run_config, 'Cauchy', 'Cauchy', 'cauchy_cauchy_distances.pdf')

    # For Outside in both source and target.
    # completed_key = []
    # for key, value in norm_mean_result.items():

    #     if key not in completed_key:
    #         completed_key.append(key)
    #         combinations = key_process(key)

    #         if combinations[0][1] == combinations[0][4] and combinations[0][1] == 'Outside':

    #             _key = "_".join(combinations[0])
    #             row, col = tag2map(_key, run_config['noise_levels'], run_config['prevention_choice_list'])
    #             outside_outside_plot_dict['gw'][row][col] = value['gw_mean']
    #             outside_outside_plot_dict['ugw'][row][col] = value['ugw_mean']
    #             outside_outside_plot_dict['pgw'][row][col] = value['pgw_mean']
    #             outside_outside_plot_dict['fgw'][row][col] = value['fgw_mean']
    #             outside_outside_plot_dict['lrgw'][row][col] = value['lrgw_mean']
                
    # _plot_heatmap_of_distances_plot_helper(outside_outside_plot_dict, run_config, 'Outside', 'Outside', 'outside_outside_distances.pdf')

def plot_error_bars(mean_result, run_config):

    # Plot the error bars over multiple run for the distances over different noise strength.
    # Case 1
    # cases = [("cat_500_Cauchy_", "_heart_500_Cauchy_15_3LT_3LT"),
    #          ('cat_500_Cauchy_', '_heart_500_Outside_15_98P_3LT'),
    #          ('cat_500_Outside_', '_heart_500_Outside_5_3LT_3LT'),
    #          ('cat_500_Outside_', '_heart_500_Outside_5_3LT_98P'),
    #          ('cat_500_Outside_', '_heart_500_Outside_15_3LT_98P')]
    
    # Case 2
    cases = [("cat_500_Gaussian_", "_heart_500_Gaussian_15_3LT_3LT"),
             ('cat_500_Gaussian_', '_heart_500_Gaussian_15_98P_3LT'),
             ('cat_500_Gaussian_', '_heart_500_Gaussian_25_3LT_3LT'),
             ('cat_500_Gaussian_', '_heart_500_Gaussian_5_95P_98P'),
             ('cat_500_Gaussian_', '_heart_500_Gaussian_20_3LT_98P')]

    _mean_result = {}
    for idx in range(len(cases)):
        _plot_data = {}
        for method in ['gw', 'ugw', 'pgw', 'fgw', 'lrgw']:
            _method_data = {}
            for noise_level in run_config['noise_levels']: 
                _method_data[noise_level] = [mean_result[cases[idx][0] + noise_level + cases[idx][1]][method + "_mean"],
                                     mean_result[cases[idx][0] + noise_level + cases[idx][1]][method + "_std"]]
            _plot_data[method] = _method_data
        _mean_result[idx] = _plot_data
    
    # Change here accordingly to the cases, for example case 1.
    # figname = os.path.join(run_config['all_output_paths']['gw_metrics_summary_save_path'], "cat_500_Cauchy_X_heart_500_Cauchy_15_3LT_3LT.pdf")
    # plot_gw_metrics_comparison(_mean_result[0], run_config['noise_levels'], figname)

    # figname = os.path.join(run_config['all_output_paths']['gw_metrics_summary_save_path'], 'cat_500_Cauchy_X_heart_500_Outside_15_98P_3LT.pdf')
    # plot_gw_metrics_comparison(_mean_result[1], run_config['noise_levels'], figname)

    # figname = os.path.join(run_config['all_output_paths']['gw_metrics_summary_save_path'], 'cat_500_Outside_X_heart_500_Outside_5_3LT_3LT.pdf')
    # plot_gw_metrics_comparison(_mean_result[2], run_config['noise_levels'], figname)

    # figname = os.path.join(run_config['all_output_paths']['gw_metrics_summary_save_path'], 'cat_500_Outside_X_heart_500_Outside_5_3LT_98P.pdf')
    # plot_gw_metrics_comparison(_mean_result[3], run_config['noise_levels'], figname)

    # figname = os.path.join(run_config['all_output_paths']['gw_metrics_summary_save_path'], 'cat_500_Outside_X_heart_500_Outside_15_3LT_98P.pdf')
    # plot_gw_metrics_comparison(_mean_result[4], run_config['noise_levels'], figname)

    # Change here accordingly to the cases, for example case 2.
    figname = os.path.join(run_config['all_output_paths']['gw_metrics_summary_save_path'], "cat_500_Gaussian_X_heart_500_Gaussian_15_3LT_3LT.pdf")
    plot_gw_metrics_comparison(_mean_result[0], run_config['noise_levels'], figname)

    figname = os.path.join(run_config['all_output_paths']['gw_metrics_summary_save_path'], 'cat_500_Gaussian_X_heart_500_Gaussian_15_98P_3LT.pdf')
    plot_gw_metrics_comparison(_mean_result[1], run_config['noise_levels'], figname)

    figname = os.path.join(run_config['all_output_paths']['gw_metrics_summary_save_path'], 'cat_500_Gaussian_X_heart_500_Gaussian_25_3LT_3LT.pdf')
    plot_gw_metrics_comparison(_mean_result[2], run_config['noise_levels'], figname)

    figname = os.path.join(run_config['all_output_paths']['gw_metrics_summary_save_path'], 'cat_500_Gaussian_X_heart_500_Gaussian_5_95P_98P.pdf')
    plot_gw_metrics_comparison(_mean_result[3], run_config['noise_levels'], figname)

    figname = os.path.join(run_config['all_output_paths']['gw_metrics_summary_save_path'], 'cat_500_Gaussian_X_heart_500_Gaussian_20_3LT_98P.pdf')
    plot_gw_metrics_comparison(_mean_result[4], run_config['noise_levels'], figname)

def plot_gw_metrics_comparison(loss_data, noisy_sample_counts, figname):
    """Create plot comparing all GW variants."""
    colors = ['rgb(102,194,165)', 
              'rgb(252,141,98)', 
              'rgb(141,160,203)',
              'rgb(231,138,195)', 
              'rgb(166,216,84)', 
              'rgb(100,100,100)']
    
    method_rename = {'gw': 'GW', 'ugw': 'UGW', 'pgw': 'PGW', 'fgw': 'FGW', 'lrgw': 'LRGW (Ours)'}
    int_sample_counts = [int(count) for count in noisy_sample_counts]
    
    fig = make_subplots()
    
    for i, (metric, metric_data) in enumerate(loss_data.items()):
        means = [metric_data[count][0] for count in noisy_sample_counts]
        stds = [metric_data[count][1] for count in noisy_sample_counts]
        
        fig.add_trace(go.Scatter(
            x=int_sample_counts,
            y=means,
            mode='lines+markers',
            name=method_rename[metric],
            line=dict(color=colors[i]),
            error_y=dict(
                type='data',
                array=stds,
                visible=True
            )
        ))
        
        # Add error bands
        upper_bound = np.array(means) + np.array(stds)
        lower_bound = np.array(means) - np.array(stds)
        
        fig.add_trace(go.Scatter(
            x=int_sample_counts + int_sample_counts[::-1],
            y=np.concatenate([upper_bound, lower_bound[::-1]]),
            fill='toself',
            fillcolor=colors[i],
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            opacity=0.2
        ))

    fig.update_layout(
        font_family="Times",
        font_color="black",
        xaxis_title='Percentage of Outliers',
        yaxis_title='Average Distance',
        font=dict(size=12),
        hovermode="x unified"
    )

    noisy_samples_labels = ['5%', '10%', '15%', '20%', '25%']
    
    fig.update_xaxes(tickmode='array', tickvals=noisy_samples_labels)
    fig.write_image(figname, width=800, height=600, scale=3)









            




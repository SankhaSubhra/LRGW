import os
import numpy as np
from sklearn import manifold

# Function taken from PyOT example of  computing GW Barycenter.
def smacof_mds(C, dim, max_iter=3000, eps=1e-9):

    # Returns an interpolated point cloud following the dissimilarity matrix C
    # using SMACOF multidimensional scaling (MDS) in specific dimensioned
    # target space

    # Parameters
    # ----------
    # C : ndarray, shape (ns, ns)
    #     dissimilarity matrix
    # dim : int
    #       dimension of the targeted space
    # max_iter :  int
    #     Maximum number of iterations of the SMACOF algorithm for a single run
    # eps : float
    #     relative tolerance w.r.t stress to declare converge

    # Returns
    # -------
    # npos : ndarray, shape (R, dim)
    #        Embedded coordinates of the interpolated point cloud (defined with
    #        one isometry)

    rng = np.random.RandomState(seed=3)

    mds = manifold.MDS(
        dim, max_iter=max_iter, eps=1e-9, dissimilarity="precomputed", n_init=1
    )
    pos = mds.fit(C).embedding_

    nmds = manifold.MDS(
        2,
        max_iter=max_iter,
        eps=1e-9,
        dissimilarity="precomputed",
        random_state=rng,
        n_init=1,
    )
    npos = nmds.fit_transform(C, init=pos)

    return npos

def path_creation(output_folder_name='Output'):

    # Creates the output path structure.

    output_path = output_folder_name
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    noisy_dataset_folder_name = 'noisy_datasets'
    noisy_dataset_path = os.path.join(output_path, noisy_dataset_folder_name)
    if not os.path.exists(noisy_dataset_path):
        os.makedirs(noisy_dataset_path)

    heatmap_save_folder_name = 'heatmap'
    heatmap_save_path = os.path.join(output_path, heatmap_save_folder_name)
    if not os.path.exists(heatmap_save_path):
        os.makedirs(heatmap_save_path)

    distance_mat_cutoffs_folder_name = 'distance_cutoffs'
    distance_mat_cutoffs_save_path = os.path.join(output_path, distance_mat_cutoffs_folder_name)
    if not os.path.exists(distance_mat_cutoffs_save_path):
        os.makedirs(distance_mat_cutoffs_save_path)    

    gw_metrics_save_folder_name = 'gw_results'
    gw_metrics_save_path = os.path.join(output_path, gw_metrics_save_folder_name)
    if not os.path.exists(gw_metrics_save_path):
        os.makedirs(gw_metrics_save_path)

    gw_metrics_summary_folder_name = 'gw_summary'
    gw_metrics_summary_save_path = os.path.join(output_path, gw_metrics_summary_folder_name)
    if not os.path.exists(gw_metrics_summary_save_path):
        os.makedirs(gw_metrics_summary_save_path)
    
    gw_barycenter_save_folder_name = 'gw_barycenter'
    gw_barycenter_save_path = os.path.join(output_path, gw_barycenter_save_folder_name)
    if not os.path.exists(gw_barycenter_save_path):
        os.makedirs(gw_barycenter_save_path)

    all_paths = {
        'output_path': output_path,
        'noisy_dataset_path': noisy_dataset_path,
        'heatmap_save_path': heatmap_save_path,
        'distance_mat_cutoffs_save_path': distance_mat_cutoffs_save_path,
        'gw_metrics_save_path': gw_metrics_save_path,
        'gw_metrics_summary_save_path': gw_metrics_summary_save_path,
        'gw_barycenter_save_path': gw_barycenter_save_path}
    
    return all_paths
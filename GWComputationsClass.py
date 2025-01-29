import ot 
import torch 
from unbalanced_gromov_wasserstein.unbalancedgw.vanilla_ugw_solver import exp_ugw_sinkhorn
from unbalanced_gromov_wasserstein.unbalancedgw._vanilla_utils import ugw_cost as compute_ugw_cost
import concurrent.futures
import threading
# from scipy.spatial.distance import cdist

class GWComputations():

    # Class to run GW distance measures and barycenters. 
    # Supports concurrency. 

    def __init__(self, PC1, PC2, C1, C2, p, q, M=None):

        # Initializer for the class. 
        # :param: PC1: np.array(n, n): distance matrix C1 processed by cut-off. 
        # :param: PC2: np.array(n, n): distance matrix C2 processed by cut-off. 
        # :param: C1: np.array(n, n): original distance matrix for the source data. 
        # :param: C2: np.array(n, n): original distance matrix for the target data. 
        # :param: p: np.array(n, 1): probability distribution over source data. 
        # :param: q: np.array(n, 1): probability distribution over target data. 
        # :param: M: np.array(n, n): distance matrix between source and target, only used by Fused GW. 

        self.PC1 = PC1
        self.PC2 = PC2
        self.C1 = C1
        self.C2 = C2
        self.p = p
        self.q = q
        self.M = M

    def compute_gw_barycenters(self, idx):  

        # GW Barycenter calculation.  

        return ot.gromov_barycenters(self.n, [self.C1, self.C2], [self.p, self.q], self.r, self.lambdast[idx], 
                'square_loss', max_iter=100, tol=1e-3)
    
    def compute_lrgw_barycenters(self, idx):

        # Locally Robust Gromov-Wasserstein Barycenter calculation. 

        return ot.gromov_barycenters(self.n, [self.PC1, self.PC2], [self.p, self.q], self.r, self.lambdast[idx], 
                'square_loss', max_iter=100, tol=1e-3)

    def compute_lrgw_distance(self,):

        # Compute Locally Robust Gromov-Wasserstein distance.

        _, log0 = ot.gromov.gromov_wasserstein(self.PC1, self.PC2, self.p, self.q, log=True)
        return 'lrgw', log0['loss'][-1]

    def compute_gw_distance(self,):

        # Compute Gromov-Wasserstein distance.

        _, log0 = ot.gromov.gromov_wasserstein(self.C1, self.C2, self.p, self.q, log=True)
        return 'gw', log0['loss'][-1]

    def compute_ugw_distance(self, loss='square', eps=5.0, rho=1.0, rho2=1.0):
        
        # Compute Unbalanced Gromov-Wasserstein distance.

        a = torch.from_numpy(self.p).float()
        b = torch.from_numpy(self.q).float()
        dx = torch.from_numpy(self.C1).float()
        dy = torch.from_numpy(self.C2).float()

        pi, gamma = exp_ugw_sinkhorn(a, dx, b, dy, loss=loss, init=None, eps=eps,
                                    rho=rho, rho2=rho2,
                                    nits_plan=10000, tol_plan=1e-11,
                                    nits_sinkhorn=10000, tol_sinkhorn=1e-12,
                                    two_outputs=True)

        # Use the renamed function here
        cost = compute_ugw_cost(pi, gamma, a, dx, b, dy, eps=eps, rho=rho, rho2=rho2)
        return 'ugw', cost.item()

    def compute_pgw_distance(self,):

        # Compute Partial Gromov-Wasserstein distance.
        # Please use the function from the modified partial for stability.

        _, log0 = ot.partial.partial_gromov_wasserstein(self.C1, self.C2, self.p, self.q, nb_dummies=1, log=True)
        return 'pgw', log0['partial_gw_dist']

    def compute_fgw_distance(self,):
     
        # Compute Fused Gromov-Wasserstein distance.

        _, log0 = ot.fused_gromov_wasserstein(self.M, self.C1, self.C2, self.p, self.q, log=True)
        return 'fgw', log0['fgw_dist']
    
    def _process_function_result(self, func):
        
        # Computes cost of different GW variants.
        try:
            tag, value = func()
            with self._cost_dict_lock:
                self.cost_dict[tag] = value
        except Exception as e:
            print(f"Error processing function: {str(e)}")

    def _execute_gw_barycenters(self, idx):

        # Executer for parallel GW Barycenter calculation.
        try:
            result = self.compute_gw_barycenters(idx)
            
            # Safely store the result in the shared dictionary
            with self._barycenters_lock:
                self.barycenters['gw'][idx] = result
                
            print("GW Barycenter computed: " + str(idx) + "/steps")
            
        except Exception as e:
            print(f"Error processing GW Barycenter for {idx}: {str(e)}")

    def _execute_lrgw_barycenters(self, idx):

        # Executer for parallel LRGW Barycenter calculation.
        try:
            result = self.compute_gw_barycenters(idx)
            
            # Safely store the result in the shared dictionary
            with self._barycenters_lock:
                self.barycenters['lrgw'][idx] = result
                
            print("LRGW Barycenter computed: " + str(idx) + "/steps")
            
        except Exception as e:
            print(f"Error processing LRGW Barycenter for {idx}: {str(e)}")

    def parallel_compute_gw_barycenters(self, idx_list):

        # Calling the executers of GW.

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            # Submit all tasks
            futures = [
                executor.submit(self._execute_gw_barycenters, idx)
                for idx in idx_list
            ]
            
            # Wait for all tasks to complete
            concurrent.futures.wait(futures)

    def parallel_compute_lrgw_barycenters(self, idx_list):

        # Calling the executers of LRGW.
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            # Submit all tasks
            futures = [
                executor.submit(self._execute_lrgw_barycenters, idx)
                for idx in idx_list
            ]
            
            # Wait for all tasks to complete
            concurrent.futures.wait(futures)

    def distance_computation_run(self,):

        # Parallel computation of GW variant distances.

        self.cost_dict = {}
        self._cost_dict_lock = threading.Lock()

        tasks = [
            self.compute_gw_distance,
            self.compute_ugw_distance,
            self.compute_pgw_distance,
            self.compute_fgw_distance,
            self.compute_lrgw_distance]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all tasks and store the future objects
            futures = [executor.submit(self._process_function_result, task) 
                      for task in tasks]
            
            # Wait for all tasks to complete
            concurrent.futures.wait(futures)

        print('------------- Distances -------------')
        print("GW: " + str(round(self.cost_dict['gw'], 5)))
        print("UGW: " + str(round(self.cost_dict['ugw'], 5)))
        print("FGW: " + str(round(self.cost_dict['fgw'], 5)))
        print("PGW: " + str(round(self.cost_dict['pgw'], 5)))
        print("LRGW (Ours): " + str(round(self.cost_dict['lrgw'], 5)))

    def barycenter_computation_run(self, steps=5):

        # Parallel computation of GW variant barycenters.

        self.barycenters = {}
        self._barycenters_lock = threading.Lock()

        self.n = len(self.C1)
        self.r = ot.unif(self.n)
        self.lambdast = [[float(i) / steps, float(steps - i) / steps] for i in range(steps+1)][1:-1]

        self.barycenters['gw'] = [0 for _ in range(steps-1)]
        self.barycenters['lrgw'] = [0 for _ in range(steps-1)]

        idx_list = list(range(steps-1))

        print('------------- Barycenters -------------')
        self.parallel_compute_gw_barycenters(idx_list)
        self.parallel_compute_lrgw_barycenters(idx_list)

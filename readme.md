# Locally Robust Gromov-Wasserstein

This work follows from this previous work:
```
@article{chakrabarty2024robust,
  title={On Robust Cross Domain Alignment},
  author={Chakrabarty, Anish and Basu, Arkaprabha and Das, Swagatam},
  journal={arXiv preprint arXiv:2412.15861},
  year={2024}
}
```
Essentially, the robustness in this case is achieved by directly imposing a outlier disrading constraint (in the form of a cut-off) on the distance matrices themselves. 

### Folder Structure
```
LRGW
|-- unbalanced_gromov_wasserstein
|   Edited codebase for better stability and square loss.
|-- PyOT-custom
|   Two custom modifications of POT. Replace in the installation environment.
|   PyOT-custom/_utils.py is an optional modification.
|   PyOT-custom/partial.py should be replaced.
|-- requirement.txt: pip3 package requirements. 
|-- main.py: driver code.
|-- GWComputationsClass.py: distance and barycenter computing class.
|-- plot_utils.py: plotting utility functions. 
|-- data_utils.py: data utility functions.
|-- _utils.py: general utility functions.
|-- data_npy: datasets and configurations in json.
|-- output: saves the output, examples in this case.
```

### Getting Started

1. First create an environment say POTEnv.
```
python3 -m venv /path_to_environemnt/POTEnv
source /path_to_environment/POTEnv/bin/activate
```
2. Install dependencies. 
```
pip3 install -r requirement.txt
```
3. Make the changes in the codebase as per need. Specifically you need to modify the `run_config` in main. 
4. Execute `main.py`.
%. If you already have a result set available just plot them by changing the config in `run_config` accordingly. 
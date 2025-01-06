# change line248,608, convertor199, display_remove_tail 156,1052, yml file
import time, functools, torch, os, sys, random, fnmatch, psutil, argparse, tqdm, yaml
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
#plt.use('Agg')
# Enable LaTeX
plt.rcdefaults()
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'  # Optional: Change to a LaTeX-compatible font
plt.rcParams['mathtext.fontset'] = 'stix'

from prettytable import PrettyTable
#torch.manual_seed(1234)
#np.random.seed(1234)
# Pytorch libs
seed = int(time.time())
# Set the seed in NumPy
np.random.seed(seed)
print(f"Random seed (from time): {seed}")

myseed = torch.seed()
print(f"Random seed (from torch): {myseed}")
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, RAdam
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader

# WandB setup
import wandb
os.environ['WANDB_NOTEBOOK_NAME'] = 'NCSM_condor'
wandb.login()

# TDSM libs
# Adds util directory to directories interpreter will search for modules
#sys.path.insert(0, '/afs/cern.ch/work/j/jthomasw/private/NTU/fast_sim/tdsm_encoder/util')
#sys.path.insert(0, '/home/ken91021615/tdsm_encoder_sweep0512/util')
sys.path.insert(1, 'util') # Local path / user independent
sys.path.insert(2, 'toy_model')
import data_utils as utils
import score_model as score_model
import sdes as sdes
import display_remove_tail as display 
import samplers as samplers
import Convertor_quantile as Convertor
from Convertor_quantile import Preprocessor


output_directory = '/eos/user/c/chenhua/copy_tdsm_encoder_sweep16/sampling_result/sampling_quantile_fulldataset20241226_0017_output'
            #output_directory = os.path.join(workingdir,'sampling_20240708_2258_output')


Geant4_files = []
for i in range(5):
    # Create an instance of Convertor for the current file
    # converter = Convertor.Convertor(file_path, 0.0, preprocessor=args.preprocessor)
    
    # # Perform the desired operations
    # converter.invert(-99)
    # converter.digitize()
    
    # Optionally, you can plot or perform other actions here
    # converter.plot_re(r_gen, E_gen, z_gen)

    # Save the output to an H5 file with a unique name for each file
    output_file = os.path.join(output_directory, f'Reference_{i}.h5')
    Geant4_files.append(output_file)
    #converter.to_h5py(output_file)

import h5py

output_file = os.path.join(output_directory, f'combined_Geant4.h5')

# Open a new HDF5 file to collect all datasets from Geant4_files
with h5py.File(output_file, 'w') as h5_out:
    combined_showers = []
    combined_energies = []

    for file_idx, h5_file in enumerate(Geant4_files):
        with h5py.File(h5_file, 'r') as h5_in:
            # Accumulate all showers and incident_energies datasets
            combined_showers.append(h5_in['showers'][:])
            combined_energies.append(h5_in['incident_energies'][:])

    # Concatenate all the collected data
    combined_showers = np.concatenate(combined_showers, axis=0)
    combined_energies = np.concatenate(combined_energies, axis=0)

    # Create datasets in the output file
    h5_out.create_dataset('showers', data=combined_showers)
    h5_out.create_dataset('incident_energies', data=combined_energies)

print(f"All files have been combined into {output_file}")



Gen_file = os.path.join(output_directory, 'Gen.h5')
Geant4_file = os.path.join(output_directory, 'Reference.h5')
plot = display.High_class_feature_plot_test(Gen_file,Geant4_files,output_directory)
plot_energy_r_plt = plot.plot_energy_r()

plot_energy_r_plt.savefig(os.path.join(output_directory, 'energy_r.png'))
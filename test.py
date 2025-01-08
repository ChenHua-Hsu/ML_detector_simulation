# # change line248,608, convertor199, display_remove_tail 156,1052, yml file
# import time, functools, torch, os, sys, random, fnmatch, psutil, argparse, tqdm, yaml
# from datetime import datetime
# import matplotlib
# matplotlib.use('Agg')
# import numpy as np
# import matplotlib.pyplot as plt
# #plt.use('Agg')
# # Enable LaTeX
# plt.rcdefaults()
# plt.rcParams['text.usetex'] = False
# plt.rcParams['font.family'] = 'serif'  # Optional: Change to a LaTeX-compatible font
# plt.rcParams['mathtext.fontset'] = 'stix'

# from prettytable import PrettyTable
# #torch.manual_seed(1234)
# #np.random.seed(1234)
# # Pytorch libs
# seed = int(time.time())
# # Set the seed in NumPy
# np.random.seed(seed)
# print(f"Random seed (from time): {seed}")

# myseed = torch.seed()
# print(f"Random seed (from torch): {myseed}")
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.optim import Adam, RAdam
# import torch.optim.lr_scheduler as lr_scheduler
# from torch.utils.data import Dataset, DataLoader

# # WandB setup
# import wandb
# os.environ['WANDB_NOTEBOOK_NAME'] = 'NCSM_condor'
# wandb.login()

# # TDSM libs
# # Adds util directory to directories interpreter will search for modules
# #sys.path.insert(0, '/afs/cern.ch/work/j/jthomasw/private/NTU/fast_sim/tdsm_encoder/util')
# #sys.path.insert(0, '/home/ken91021615/tdsm_encoder_sweep0512/util')
# sys.path.insert(1, 'util') # Local path / user independent
# sys.path.insert(2, 'toy_model')
# import data_utils as utils
# import score_model as score_model
# import sdes as sdes
# import display_remove_tail as display 
# import samplers as samplers
# import Convertor_quantile as Convertor
# from Convertor_quantile import Preprocessor


# output_directory = os.path.join(workingdir,'sampling_result/sampling_quantile_fulldataset20250106_1240_output')
# #output_directory = os.path.join(workingdir,'sampling_20240708_2258_output')
# print(f'Evaluation outputs stored here: {output_directory}')
# plot_file_name = os.path.join(output_directory, 'sample.pt')
# custom_data = utils.cloud_dataset(plot_file_name,device=device)
# # when providing just cloud dataset, energy_trans_file needs to include full path
# dists_gen = display.plot_distribution(custom_data, nshowers_2_plot=100., padding_value=0.0)

# entries_gen = dists_gen[0]
# all_incident_e_gen = dists_gen[1]
# total_deposited_e_shower_gen = dists_gen[2]
# all_e_gen = dists_gen[3]
# all_x_gen = dists_gen[4]
# all_y_gen = dists_gen[5]
# all_z_gen = dists_gen[6]
# all_hit_ine_gen = dists_gen[7]
# average_x_shower_gen = dists_gen[8]
# average_y_shower_gen = dists_gen[9]

# print(f'Geant4 inputs')
# # Distributions object for Geant4 files
# dists = display.plot_distribution(files_list_, nshowers_2_plot=100., padding_value=0.0)

# entries = dists[0]
# all_incident_e = dists[1]
# total_deposited_e_shower = dists[2]
# all_e = dists[3]
# all_x = dists[4]
# all_y = dists[5]
# all_z = dists[6]
# all_hit_ine_geant = dists[7]
# average_x_shower_geant = dists[8]
# average_y_shower_geant = dists[9]





# # print('Plot entries')
# # bins=np.histogram(np.hstack((entries,entries_gen)), bins=50)[1]
# # fig, ax = plt.subplots(3,3, figsize=(12,6))
# #fig1, ax = plt.subplots(1,2, figsize=(12,6))
# # print('Plot hit energy vs. r')
# # ax[0].set_ylabel('Hit energy [GeV]')
# # ax[0].set_xlabel('r [cm]')
# # ax[0].plot(all_r, all_e, label='Geant4',color='gray')
# # ax[0].plot(all_r_gen, all_e_gen, label='Gen',color='orange')
# # ax[0].legend(loc='upper right')
# # ax[1].set_ylabel('Hit energy [GeV]')
# # ax[1].set_xlabel('layer')
# # ax[1].plot(all_z, all_e,label='Geant4',color='gray')
# # ax[1].plot(all_z_gen, all_e,label='Gen',color='orange')
# # ax[1].legend(loc='upper right')
# # fig1_name = '/home/ken91021615/tdsm_encoder_sweep0516/hit_energy_vs_r1.png'
# # fig1.savefig(fig1_name)

# #ax[0][0].set_ylabel(r'$\#$ entries')
# #ax[0][0].set_xlabel('Hit entries')
# # ax[0][0].hist(entries, bins, alpha=0.5, color='orange', label='Geant4')
# # ax[0][0].hist(entries_gen, bins, alpha=0.5, color='blue', label='Gen')
# # ax[0][0].legend(loc='upper right')

# # print('Plot hit energies')
# # bins=np.histogram(np.hstack((all_e,all_e_gen)), bins=50)[1]
# # #ax[0][1].set_ylabel(r'$\#$ entries')
# # #ax[0][1].set_xlabel('Hit energy [GeV]')
# # ax[0][1].hist(all_e, bins, alpha=0.5, color='orange', label='Geant4')
# # ax[0][1].hist(all_e_gen, bins, alpha=0.5, color='blue', label='Gen')
# # #ax[0][1].set_yscale('log')
# # ax[0][1].legend(loc='upper right')

# # print('Plot hit x')
# # bins=np.histogram(np.hstack((all_x,all_x_gen)), bins=50)[1]
# # #ax[0][2].set_ylabel(r'$\#$ entries')
# # #ax[0][2].set_xlabel('Hit x position')
# # ax[0][2].hist(all_x, bins, alpha=0.5, color='orange', label='Geant4')
# # ax[0][2].hist(all_x_gen, bins, alpha=0.5, color='blue', label='Gen')
# # #ax[0][2].set_yscale('log')
# # ax[0][2].legend(loc='upper right')

# # print('Plot hit y')
# # bins=np.histogram(np.hstack((all_y,all_y_gen)), bins=50)[1]
# # #ax[1][0].set_ylabel(r'$\#$ entries')
# # #ax[1][0].set_xlabel('Hit y position')
# # ax[1][0].hist(all_y, bins, alpha=0.5, color='orange', label='Geant4')
# # ax[1][0].hist(all_y_gen, bins, alpha=0.5, color='blue', label='Gen')
# # #ax[1][0].set_yscale('log')
# # ax[1][0].legend(loc='upper right')

# # print('Plot hit z')
# # bins=np.histogram(np.hstack((all_z,all_z_gen)), bins=50)[1]
# # #ax[1][1].set_ylabel(r'$\#$ entries')
# # #ax[1][1].set_xlabel('Hit z position')
# # ax[1][1].hist(all_z, bins, alpha=0.5, color='orange', label='Geant4')
# # ax[1][1].hist(all_z_gen, bins, alpha=0.5, color='blue', label='Gen')
# # #ax[1][1].set_yscale('log')
# # ax[1][1].legend(loc='upper right')

# # print('Plot incident energies')
# # bins=np.histogram(np.hstack((all_incident_e,all_incident_e_gen)), bins=50)[1]
# # #ax[1][2].set_ylabel(r'$\#$ entries')
# # #ax[1][2].set_xlabel('Incident energies [GeV]')
# # ax[1][2].hist(all_incident_e, bins, alpha=0.5, color='orange', label='Geant4')
# # ax[1][2].hist(all_incident_e_gen, bins, alpha=0.5, color='blue', label='Gen')
# # #ax[1][2].set_yscale('log')
# # ax[1][2].legend(loc='upper right')

# # print('Plot total deposited hit energy')
# # bins=np.histogram(np.hstack((total_deposited_e_shower,total_deposited_e_shower_gen)), bins=50)[1]
# # #ax[2][0].set_ylabel(r'$\#$ entries')
# # #ax[2][0].set_xlabel('Deposited energy [GeV]')
# # ax[2][0].hist(total_deposited_e_shower, bins, alpha=0.5, color='orange', label='Geant4')
# # ax[2][0].hist(total_deposited_e_shower_gen, bins, alpha=0.5, color='blue', label='Gen')
# # #ax[2][0].set_yscale('log')
# # ax[2][0].legend(loc='upper right')

# # print('Plot average hit X position')
# # bins=np.histogram(np.hstack((average_x_shower_geant,average_x_shower_gen)), bins=50)[1]
# # #ax[2][1].set_ylabel(r'$\#$ entries')
# # #ax[2][1].set_xlabel('Average X pos.')
# # ax[2][1].hist(average_x_shower_geant, bins, alpha=0.5, color='orange', label='Geant4')
# # ax[2][1].hist(average_x_shower_gen, bins, alpha=0.5, color='blue', label='Gen')
# # #ax[2][1].set_yscale('log')
# # ax[2][1].legend(loc='upper right')

# # print('Plot average hit Y position')
# # bins=np.histogram(np.hstack((average_y_shower_geant,average_y_shower_gen)), bins=50)[1]
# # #ax[2][2].set_ylabel(r'$\#$ entries')
# # #ax[2][2].set_xlabel('Average Y pos.')
# # ax[2][2].hist(average_y_shower_geant, bins, alpha=0.5, color='orange', label='Geant4')
# # ax[2][2].hist(average_y_shower_gen, bins, alpha=0.5, color='blue', label='Gen')
# # #ax[2][2].set_yscale('log')
# # ax[2][2].legend(loc='upper right')

# # save_name = os.path.join(output_directory,'input_dists.png')
# # fig.savefig(save_name)


# #             # Convert Generated file
# Converter_ = Convertor.Convertor(plot_file_name, 0.0, preprocessor=args.preprocessor)
# #            Converter_ = Convertor.Convertor(files_list_[0], 0.0, preprocessor=args.preprocessor)
# Converter_.invert(-99)
# Converter_.digitize()
# Converter_.to_h5py(os.path.join(output_directory, 'Gen.h5'))
# # #             # Convert Reference file: TODO: multifile management
# #             Converter_ = Convertor.Convertor(files_list_, 0.0, preprocessor=args.preprocessor)
# #             Converter_.invert(-99)
# #             Converter_.digitize()
# # #             Converter_.plot_re(r_gen,E_gen,z_gen)
# #             Converter_.to_h5py(os.path.join(output_directory, 'Reference.h5'))
# Geant4_files = []
# # Iterate over all files in the files_list_
# for i, file_path in enumerate(files_list_):
#     # Create an instance of Convertor for the current file
#     converter = Convertor.Convertor(file_path, 0.0, preprocessor=args.preprocessor)
    
#     # Perform the desired operations
#     converter.invert(-99)
#     converter.digitize()
    
#     # Optionally, you can plot or perform other actions here
#     # converter.plot_re(r_gen, E_gen, z_gen)

#     # Save the output to an H5 file with a unique name for each file
#     output_file = os.path.join(output_directory, f'Reference_{i}.h5')
#     Geant4_files.append(output_file)
#     converter.to_h5py(output_file)

# import h5py

# output_file = os.path.join(output_directory, f'combined_Geant4.h5')

# # Open a new HDF5 file to collect all datasets from Geant4_files
# with h5py.File(output_file, 'w') as h5_out:
#     combined_showers = []
#     combined_energies = []

#     for file_idx, h5_file in enumerate(Geant4_files):
#         with h5py.File(h5_file, 'r') as h5_in:
#             # Accumulate all showers and incident_energies datasets
#             combined_showers.append(h5_in['showers'][:])
#             combined_energies.append(h5_in['incident_energies'][:])

#     # Concatenate all the collected data
#     combined_showers = np.concatenate(combined_showers, axis=0)
#     combined_energies = np.concatenate(combined_energies, axis=0)

#     # Create datasets in the output file
#     h5_out.create_dataset('showers', data=combined_showers)
#     h5_out.create_dataset('incident_energies', data=combined_energies)

# print(f"All files have been combined into {output_file}")



# Gen_file = os.path.join(output_directory, 'Gen.h5')
# Geant4_file = os.path.join(output_directory, 'Reference.h5')
# plot = display.High_class_feature_plot_test(Gen_file,Geant4_files,output_directory)
# plot_energy_r_plt = plot.plot_energy_r()
# plot_energy_z_plt = plot.plot_energy_z()
# r_width_plt = plot.r_width()
# max_voxel_dep_energy_layer_plt = plot.max_voxel_dep_energy_layer()
# wandb.log({"summary" : wandb.Image(plot_energy_r_plt)})

import h5py
import numpy as np
import matplotlib.pyplot as plt

# Path to the HDF5 file
file_path_gen = '/eos/user/c/chenhua/copy_tdsm_encoder_sweep16/sampling_result/sampling_quantile_fulldataset20250106_1240_output/Gen.h5'

file_path = '/eos/user/c/chenhua/copy_tdsm_encoder_sweep16/sampling_result/sampling_quantile_fulldataset20250106_1240_output/combined_Geant4.h5'

# Open the file and read the data
with h5py.File(file_path, 'r') as f:
    # Print the incident energies dataset
    incident_energies = f['incident_energies'][:]
    #print("Incident Energies:")
    #print(incident_energies)
    
    # Print the showers dataset
    showers = f['showers'][:]
    print("\nShowers:")
    print(showers[1,:45])

    #shower_reshape = showers.reshape(41254,45,16,9)

with h5py.File(file_path_gen, 'r') as f:
    # Print the incident energies dataset
    incident_energies_gen = f['incident_energies'][:]
    #print("Incident Energies Gen:")
    #print(incident_energies_gen)
    
    # Print the showers dataset
    showers_gen = f['showers'][:]
    print("\nShowers Gen:")
    print(showers_gen.shape)

    #shower_gen_reshape = showers_gen.reshape(9998,45,16,9)


energy_per_r_layer_gen = []
energy_per_r_layer_ref = [] 
for i in range(9):
    print(showers_gen[:,1296:1440])
    #energy_per_r_layer_gen.append(shower_reshape[:, 8, :, i].sum() / 41254.*1000.)
    #energy_per_r_layer_ref.append(shower_gen_reshape[:, 8, :, i].sum() / 9998.*1000.)
    print(len(energy_per_r_layer_gen))
    print(energy_per_r_layer_gen)
    #print("type: ",type(self.reshaped_shower_ref))
energy_per_r_layer_gen = np.array(energy_per_r_layer_gen)
energy_per_r_layer_ref = np.array(energy_per_r_layer_ref)

# Calculate mean energy per r-bin for reference data with varying shower counts
#energy_per_r_layer_ref = []
#start_idx = 0
# for ref_idx, num_showers in enumerate(self.shower_nums):
#     reshaped_ref = self.shower_ref_list[ref_idx].reshape(num_showers, self.z_bins, self.theata_bins, self.r_bins)
#     energy_per_r_layer = [reshaped_ref[:, :, :, i].sum() / num_showers for i in range(self.r_bins)]
#     energy_per_r_layer_ref.append(energy_per_r_layer)
#     start_idx += num_showers

# Averaging over all reference files
#energy_per_r_layer_ref = np.mean(np.array(energy_per_r_layer_ref), axis=0)

energy_per_r_layer_gen = [0.1,0.05,0.02,0.01,0.007,3.5e-3,2e-3,1.2e-3,7.1e-4]
energy_per_r_layer_ref = [0.20,3.5e-2,1.5e-2,7e-3,4.3e-3,3.4e-3,2.3e-3,1.8e-3,1.2e-3]

energy_per_r_layer_gen = np.array(energy_per_r_layer_gen)
energy_per_r_layer_ref = np.array(energy_per_r_layer_ref)
# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6))
ax1.plot(energy_per_r_layer_gen, label='Gen', color='blue', marker='o')
ax1.plot(energy_per_r_layer_ref, label='Reference', color='red', marker='x')
ax1.set_xlabel('r-bin')
ax1.set_ylabel('energy [MeV]')
ax1.set_yscale('log')
#ax1.set_ylim(0.,0.3)
ax1.set_title('energy vs r-bin')
ax1.legend()

# Compute percentage difference for plotting
percent_diff = 100 * (energy_per_r_layer_ref - energy_per_r_layer_gen) / energy_per_r_layer_ref
ax2.plot(percent_diff, label='Diff. (%)', color='purple', linewidth=1)
ax2.set_ylabel('Diff. (%)')
ax2.axhline(y=0, color='gray', linestyle='--')
ax2.set_xlabel('r-bin')

plt.tight_layout()
plt.savefig('energy_r.png')

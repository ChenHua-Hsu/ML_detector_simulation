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
import display as display 
import samplers as samplers
import Convertor_quantile as Convertor
from Convertor_quantile import Preprocessor
#import fid_score1 as fid_score1


def train_log(loss, batch_ct, epoch):
    wandb.log({"epoch": epoch, "loss": loss}, step=batch_ct)

def build_dataset(filename, train_ratio, batch_size, device):
    # Build dataset
    custom_data = utils.cloud_dataset(filename, device=device)
    #custom_data.clean(20)
    train_size = int(train_ratio * len(custom_data.data))
    test_size = len(custom_data.data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(custom_data, [train_size, test_size])
    shower_loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    shower_loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return shower_loader_train, shower_loader_test

def check_mem():
    # Resident set size memory (non-swap physical memory process has used)
    process = psutil.Process(os.getpid())
    # Print bytes in GB
    print('Memory usage of current process 0 [GB]: ', process.memory_info().rss/(1024 * 1024 * 1024))
    return

def train_model(files_list_, device='cpu',serialized_model=False):

    # access all HPs through wandb.config, so logging matches execution!
    config = wandb.config
    # if config.batch_size == 64:
    #     config.num_encoder_blocks = 8
    print(f'training config: {config}')

    wd = os.getcwd()
    #wd = '/afs/cern.ch/work/j/jthomasw/private/NTU/fast_sim/tdsm_encoder/'
    output_files = './training_result/training_quantile_fulldataset'+datetime.now().strftime('%Y%m%d_%H%M')+'_output/'
    output_directory = os.path.join(wd, output_files)
    print('Training directory: ', output_directory)
    if not os.path.exists(output_directory):
        print(f'Making new dir . . . . . ')
        os.makedirs(output_directory)

    # Instantiate stochastic differential equation
    if config.SDE == 'subVP':
        sde = sdes.subVPSDE(beta_max=config.sigma_max, beta_min=config.sigma_min, device=device)
    if config.SDE == 'VP':
        sde = sdes.VPSDE(beta_max=config.sigma_max, beta_min=config.sigma_min, device=device)
    if config.SDE == 'VE':
        sde = sdes.VESDE(sigma_max=config.sigma_max,device=device)
    marginal_prob_std_fn = functools.partial(sde.marginal_prob)

    # Instantiate model
    loss_fn = score_model.ScoreMatchingLoss()
    if not serialized_model:
        model = score_model.Gen(config.n_feat_dim, config.embed_dim, config.hidden_dim, config.num_encoder_blocks, config.num_attn_heads, config.dropout_gen, marginal_prob_std=marginal_prob_std_fn)
    else:
        model = score_model.get_seq_model(config.n_feat_dim, config.embed_dim, config.hidden_dim, config.num_encoder_blocks, config.num_attn_heads, config.dropout_gen, marginal_prob_std=marginal_prob_std_fn)
    #model = score_model.Gen(config.n_feat_dim, config.embed_dim, config.hidden_dim, config.num_encoder_blocks, config.num_attn_heads, config.dropout_gen, marginal_prob_std=marginal_prob_std_fn)

    table = PrettyTable(['Module name', 'Parameters listed'])
    t_params = 0
    for name_ , para_ in model.named_parameters():
        if not para_.requires_grad: continue
        param = para_.numel()
        table.add_row([name_, param])
        t_params+=param
    print(table)
    print(f'Sum of trainable parameters: {t_params}')    
    
    if torch.cuda.device_count() > 1:
        print(f'Lets use {torch.cuda.device_count()} GPUs!')
        model = nn.DataParallel(model)

    # Optimiser needs to know model parameters for to optimise
    optimiser = RAdam(model.parameters(),lr=config.lr)
    scheduler = lr_scheduler.ExponentialLR(optimiser, gamma=0.99)

    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, loss_fn, log="all", log_freq=10)
    
    eps_ = []
    batch_ct = 0
    for epoch in range(0, config.epochs ):
        sys.stdout.write('\r')
        sys.stdout.write('Progress: %d/%d'%((epoch+1), config.epochs)) # Local Progress Tracker
        sys.stdout.flush()
        eps_.append(epoch)

        # Create/clear per epoch variables
        cumulative_epoch_loss = 0.
        file_counter = 0
        training_batches_per_epoch = 0
        testing_batches_per_epoch = 0

        # Load files
        for filename in files_list_:
            file_counter+=1

            # Build dataset
            shower_loader_train, shower_loader_test = build_dataset(filename, config.train_ratio, config.batch_size, device)

            # Accumuate number of batches per epoch
            training_batches_per_epoch += len(shower_loader_train)
            testing_batches_per_epoch += len(shower_loader_test)
            
            # Load shower batch for training
            for i, (shower_data,incident_energies) in enumerate(shower_loader_train,0):
                batch_ct+=1
                # Move model to device and set dtype as same as data (note torch.double works on both CPU and GPU)
                model.to(device, shower_data.dtype)
                model.train()
                shower_data.to(device)
                incident_energies.to(device)
                if len(shower_data) < 1:
                    continue

                # Zero any gradients from previous steps
                optimiser.zero_grad()
                # Loss average for each batch
                loss = loss_fn(model, shower_data, incident_energies, marginal_prob_std_fn, padding_value=0.0, device=device, diffusion_on_mask=False,serialized_model=False, cp_chunks=4)
                # collect dL/dx for any parameters (x) which have requires_grad = True via: x.grad += dL/dx
                loss.backward()
                cumulative_epoch_loss+=loss.item()
                # Update value of x += -lr * x.grad
                optimiser.step()
                # Report metrics every 5th batch
                if ((batch_ct + 1) % 5) == 0:
                    train_log(loss, batch_ct, epoch)
            
            # Testing on subset of file
            for i, (shower_data,incident_energies) in enumerate(shower_loader_test,0):
                with torch.no_grad():
                    model.to(device, shower_data.dtype)
                    model.eval()
                    shower_data = shower_data.to(device)
                    incident_energies = incident_energies.to(device)
                    test_loss = score_model.loss_fn(model, shower_data, incident_energies, marginal_prob_std_fn, padding_value=0.0, device=device,serialized_model=False, cp_chunks=4)

        scheduler.step()
        
        # Save checkpoints
        if epoch%10 == 0:
            torch.save(model.state_dict(), os.path.join(output_directory, 'ckpt_tmp_'+str(epoch)+'.pth' ))
    
    save_name = os.path.join(output_directory, 'ckpt_tmp_'+str(epoch)+'.pth' )
    torch.save(model.state_dict(), save_name)
    return save_name


def generate(files_list_, load_filename, device='cpu', serialized_model=False):
    sample_savename = '/eos/user/c/chenhua/copy_tdsm_encoder_sweep16/sampling_result/sampling_quantile_fulldataset20241216_0942_output/sample.pt'
    output_directory = '/eos/user/c/chenhua/copy_tdsm_encoder_sweep16/sampling_result/sampling_quantile_fulldataset20241216_0942_output/'
    gen_data = utils.cloud_dataset(sample_savename,device=device)
    # Generated distributions
    #dists_gen = display.plot_distribution(gen_data, nshowers_2_plot=100, padding_value=0.0)
    # Distributions object for Geant4 files
    #Help me randomely generate dists

    
    #generate ramdome data
    z = torch.normal(0,1,size=100)
    dists = [z,z,z,z,z,z,z,z,z,z,z]
    #dists = display.plot_distribution(files_list_, nshowers_2_plot=100, padding_value=0.0)
    comparison_fig = display.comparison_summary(dists, dists, output_directory)#, erange=(-5,3), xrange=(-2.5,2.5), yrange=(-2.5,2.5), zrange=(0,1))
    # Add evaluation plots to keep on wandb
    #et_correlation_gen = display.correlation(dists_gen[3],dists_gen[5],output_directory)
    #rt_correlation_gen = display.correlation(dists_gen[4],dists_gen[5],output_directory)
    #zt_correlation_gen = display.correlation(dists_gen[6],dists_gen[5],output_directory)
    #test = fid_score1.Score(dists[3],dists[4],dists[5],dists[6],dists_gen[3],dists_gen[4],dists_gen[5],dists_gen[6])
    #score_fid = test.FID_score()
    #score_fid_4D = test.FID_score_4D()
    #wandb.log({"summary" : wandb.Image(comparison_fig),"FID_e" : score_fid[0], "FID_x" : score_fid[1], "FID_y" : score_fid[2], "FID_z" : score_fid[3], "FID" : score_fid_4D, "time_consuming" : elapsed_time})
    wandb.log({"summary" : wandb.Image(comparison_fig)})



    return output_directory
def main(config=None):
    
    indir = args.inputs
    switches_ = int('0b'+args.switches,2)
    switches_str = bin(int('0b'+args.switches,2))
    print(f"Random seed (from time): {seed}")
    print(f"Random seed (from torch): {myseed}")
    trigger = 0b0001
    print(f'switches trigger: {switches_str}')
    if switches_ & trigger:
        print('input_feature_plots = ON')
    if switches_>>1 & trigger:
        print('training_switch = ON')
    if switches_>>2 & trigger:
        print('sampling_switch = ON')
    if switches_>>3 & trigger:
        print('evaluation_plots_switch = ON')

    print('torch version: ', torch.__version__)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Running on device: ', device)
    if torch.cuda.is_available():
        print('Cuda used to build pyTorch: ',torch.version.cuda)
        print('Current device: ', torch.cuda.current_device())
        print('Cuda arch list: ', torch.cuda.get_arch_list())
    
    print('Working directory: ' , os. getcwd())

    # Useful when debugging gradient issues
    torch.autograd.set_detect_anomaly(True)

    padding_value = 0.0

    # List of training input files
    training_file_path = os.path.join(indir) # change indir to be absolute path
    files_list_ = []
    print(f'Training files found in: {training_file_path}')
    for filename in os.listdir(training_file_path):
        if fnmatch.fnmatch(filename, 'dataset_2_padded_transform_incident_later_nentry130*.pt'):
            files_list_.append(os.path.join(training_file_path,filename))
    print(f'Files: {files_list_}')
    
    with wandb.init(config=config, project='testsweep', entity='calo_tNCSM'):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config
        if config.batch_size == 64:
            config.num_encoder_blocks = 8

        #### Input plots ####
        if switches_ & trigger:
            # Limited to n_showers_2_gen showers in for plots
            # Transformed variables
            dists_trans = display.plot_distribution(files_list_, nshowers_2_plot=config.n_showers_2_gen, padding_value=padding_value)
            entries = dists_trans[0]
            all_incident_e_trans = dists_trans[1]
            total_deposited_e_shower_trans = dists_trans[2]
            all_e_trans = dists_trans[3]
            all_x_trans = dists_trans[4]
            all_y_trans = dists_trans[5]
            all_z_trans = dists_trans[6]
            all_hit_ine_trans = dists_trans[7]
            average_x_shower_trans = dists_trans[8]
            average_y_shower_trans = dists_trans[9]

            ### 1D histograms
            fig, ax = plt.subplots(3,3, figsize=(12,12))
            print('Plot entries')
            #ax[0][0].set_ylabel(r'$\#$ entries')
            #ax[0][0].set_xlabel(r'$\#$ Hit entries')
            ax[0][0].hist(entries, 50, color='orange', label='Geant4')
            ax[0][0].legend(loc='upper right')

            print('Plot hit energies')
            #ax[0][1].set_ylabel(r'$\#$ entries')
            #ax[0][1].set_xlabel(r'$\#$ Hit energy [GeV]')
            ax[0][1].hist(all_e_trans, 50, color='orange', label='Geant4')
            ax[0][1].set_yscale('log')
            ax[0][1].legend(loc='upper right')

            print('Plot hit x')
            #ax[0][2].set_ylabel(r'$\#$ entries')
            #ax[0][2].set_xlabel(r'$\#$ Hit x position')
            ax[0][2].hist(all_x_trans, 50, color='orange', label='Geant4')
            ax[0][2].set_yscale('log')
            ax[0][2].legend(loc='upper right')

            print('Plot hit y')
            #ax[1][0].set_ylabel(r'$\#$ entries')
            #ax[1][0].set_xlabel(r'$\#$ Hit y position')
            ax[1][0].hist(all_y_trans, 50, color='orange', label='Geant4')
            ax[1][0].set_yscale('log')
            ax[1][0].legend(loc='upper right')

            print('Plot hit z')
            #ax[1][1].set_ylabel(r'$\#$ entries')
            #ax[1][1].set_xlabel(r'$\#$ Hit z position')
            ax[1][1].hist(all_z_trans, color='orange', label='Geant4')
            ax[1][1].set_yscale('log')
            ax[1][1].legend(loc='upper right')

            print('Plot incident energies')
            #ax[1][2].set_ylabel(r'$\#$ entries')
            #ax[1][2].set_xlabel(r'Incident energies [GeV]')
            ax[1][2].hist(all_incident_e_trans, 50, color='orange', label='Geant4')
            ax[1][2].set_yscale('log')
            ax[1][2].legend(loc='upper right')

            print('Plot total deposited hit energy per shower')
            #ax[2][0].set_ylabel(r'$\#$ entries')
            #ax[2][0].set_xlabel(r'Deposited energy [GeV]')
            ax[2][0].hist(total_deposited_e_shower_trans, 50, color='orange', label='Geant4')
            ax[2][0].set_yscale('log')
            ax[2][0].legend(loc='upper right')

            print('Plot av. X position per shower')
            #ax[2][1].set_ylabel(r'$\#$ entries')
            #ax[2][1].set_xlabel(r'Average X position [GeV]')
            ax[2][1].hist(average_x_shower_trans, 50, color='orange', label='Geant4')
            ax[2][1].set_yscale('log')
            ax[2][1].legend(loc='upper right')

            print('Plot av. Y position per shower')
            #ax[2][2].set_ylabel(r'$\#$ entries')
            #ax[2][2].set_xlabel(r'Average Y position [GeV]')
            ax[2][2].hist(average_y_shower_trans, 50, color='orange', label='Geant4')
            ax[2][2].set_yscale('log')
            ax[2][2].legend(loc='upper right')

            save_name = os.path.join(training_file_path,'input_dists_transformed.png')
            fig.savefig(save_name)
        
        #### Sampling ####
        if switches_>>2 & trigger:
            # If a new training was run and you want to use it
            if switches_>>1 & trigger:
                output_directory = generate(files_list_, load_filename=trained_model_name, device=device)
            # To use an older training file
            # n.b. you'll need to make sure the config hyperparams are the same as the model being used
            else:
#                trained_model_name = 'training_20240408_1350_output/ckpt_tmp_299.pth'
                trained_model_name = '/eos/user/c/chenhua/copy_tdsm_encoder_sweep16/training_result/training_quantile_fulldataset20241215_1512_output/ckpt_tmp_199.pth'
                output_directory = generate(files_list_, load_filename=trained_model_name, device=device)
            

        #### Evaluation plots ####
        if switches_>>3 & trigger:
            # Distributions object for generated files
            print(f'Generated inputs')
            workingdir = os.getcwd()
            #out = './'
            if not switches_>>2 & trigger:
              output_directory = os.path.join(workingdir,'sampling_result/sampling_quantile_fulldataset20241216_0942_output')
            #output_directory = os.path.join(workingdir,'sampling_20240708_2258_output')
            print(f'Evaluation outputs stored here: {output_directory}')
            plot_file_name = os.path.join(output_directory, 'sample.pt')
            custom_data = utils.cloud_dataset(plot_file_name,device=device)
            # when providing just cloud dataset, energy_trans_file needs to include full path
            dists_gen = display.plot_distribution(custom_data, nshowers_2_plot=config.n_showers_2_gen, padding_value=padding_value)

            entries_gen = dists_gen[0]
            all_incident_e_gen = dists_gen[1]
            total_deposited_e_shower_gen = dists_gen[2]
            all_e_gen = dists_gen[3]
            all_x_gen = dists_gen[4]
            all_y_gen = dists_gen[5]
            all_z_gen = dists_gen[6]
            all_hit_ine_gen = dists_gen[7]
            average_x_shower_gen = dists_gen[8]
            average_y_shower_gen = dists_gen[9]

            print(f'Geant4 inputs')
            # Distributions object for Geant4 files
            dists = display.plot_distribution(files_list_, nshowers_2_plot=config.n_showers_2_gen, padding_value=padding_value)

            entries = dists[0]
            all_incident_e = dists[1]
            total_deposited_e_shower = dists[2]
            all_e = dists[3]
            all_x = dists[4]
            all_y = dists[5]
            all_z = dists[6]
            all_hit_ine_geant = dists[7]
            average_x_shower_geant = dists[8]
            average_y_shower_geant = dists[9]

            



            print('Plot entries')
            bins=np.histogram(np.hstack((entries,entries_gen)), bins=50)[1]
            fig, ax = plt.subplots(3,3, figsize=(12,6))
            #fig1, ax = plt.subplots(1,2, figsize=(12,6))
            # print('Plot hit energy vs. r')
            # ax[0].set_ylabel('Hit energy [GeV]')
            # ax[0].set_xlabel('r [cm]')
            # ax[0].plot(all_r, all_e, label='Geant4',color='gray')
            # ax[0].plot(all_r_gen, all_e_gen, label='Gen',color='orange')
            # ax[0].legend(loc='upper right')
            # ax[1].set_ylabel('Hit energy [GeV]')
            # ax[1].set_xlabel('layer')
            # ax[1].plot(all_z, all_e,label='Geant4',color='gray')
            # ax[1].plot(all_z_gen, all_e,label='Gen',color='orange')
            # ax[1].legend(loc='upper right')
            # fig1_name = '/home/ken91021615/tdsm_encoder_sweep0516/hit_energy_vs_r1.png'
            # fig1.savefig(fig1_name)

            #ax[0][0].set_ylabel(r'$\#$ entries')
            #ax[0][0].set_xlabel('Hit entries')
            ax[0][0].hist(entries, bins, alpha=0.5, color='orange', label='Geant4')
            ax[0][0].hist(entries_gen, bins, alpha=0.5, color='blue', label='Gen')
            ax[0][0].legend(loc='upper right')

            print('Plot hit energies')
            bins=np.histogram(np.hstack((all_e,all_e_gen)), bins=50)[1]
            #ax[0][1].set_ylabel(r'$\#$ entries')
            #ax[0][1].set_xlabel('Hit energy [GeV]')
            ax[0][1].hist(all_e, bins, alpha=0.5, color='orange', label='Geant4')
            ax[0][1].hist(all_e_gen, bins, alpha=0.5, color='blue', label='Gen')
            #ax[0][1].set_yscale('log')
            ax[0][1].legend(loc='upper right')

            print('Plot hit x')
            bins=np.histogram(np.hstack((all_x,all_x_gen)), bins=50)[1]
            #ax[0][2].set_ylabel(r'$\#$ entries')
            #ax[0][2].set_xlabel('Hit x position')
            ax[0][2].hist(all_x, bins, alpha=0.5, color='orange', label='Geant4')
            ax[0][2].hist(all_x_gen, bins, alpha=0.5, color='blue', label='Gen')
            #ax[0][2].set_yscale('log')
            ax[0][2].legend(loc='upper right')

            print('Plot hit y')
            bins=np.histogram(np.hstack((all_y,all_y_gen)), bins=50)[1]
            #ax[1][0].set_ylabel(r'$\#$ entries')
            #ax[1][0].set_xlabel('Hit y position')
            ax[1][0].hist(all_y, bins, alpha=0.5, color='orange', label='Geant4')
            ax[1][0].hist(all_y_gen, bins, alpha=0.5, color='blue', label='Gen')
            #ax[1][0].set_yscale('log')
            ax[1][0].legend(loc='upper right')

            print('Plot hit z')
            bins=np.histogram(np.hstack((all_z,all_z_gen)), bins=50)[1]
            #ax[1][1].set_ylabel(r'$\#$ entries')
            #ax[1][1].set_xlabel('Hit z position')
            ax[1][1].hist(all_z, bins, alpha=0.5, color='orange', label='Geant4')
            ax[1][1].hist(all_z_gen, bins, alpha=0.5, color='blue', label='Gen')
            #ax[1][1].set_yscale('log')
            ax[1][1].legend(loc='upper right')

            print('Plot incident energies')
            bins=np.histogram(np.hstack((all_incident_e,all_incident_e_gen)), bins=50)[1]
            #ax[1][2].set_ylabel(r'$\#$ entries')
            #ax[1][2].set_xlabel('Incident energies [GeV]')
            ax[1][2].hist(all_incident_e, bins, alpha=0.5, color='orange', label='Geant4')
            ax[1][2].hist(all_incident_e_gen, bins, alpha=0.5, color='blue', label='Gen')
            #ax[1][2].set_yscale('log')
            ax[1][2].legend(loc='upper right')

            print('Plot total deposited hit energy')
            bins=np.histogram(np.hstack((total_deposited_e_shower,total_deposited_e_shower_gen)), bins=50)[1]
            #ax[2][0].set_ylabel(r'$\#$ entries')
            #ax[2][0].set_xlabel('Deposited energy [GeV]')
            ax[2][0].hist(total_deposited_e_shower, bins, alpha=0.5, color='orange', label='Geant4')
            ax[2][0].hist(total_deposited_e_shower_gen, bins, alpha=0.5, color='blue', label='Gen')
            #ax[2][0].set_yscale('log')
            ax[2][0].legend(loc='upper right')

            print('Plot average hit X position')
            bins=np.histogram(np.hstack((average_x_shower_geant,average_x_shower_gen)), bins=50)[1]
            #ax[2][1].set_ylabel(r'$\#$ entries')
            #ax[2][1].set_xlabel('Average X pos.')
            ax[2][1].hist(average_x_shower_geant, bins, alpha=0.5, color='orange', label='Geant4')
            ax[2][1].hist(average_x_shower_gen, bins, alpha=0.5, color='blue', label='Gen')
            #ax[2][1].set_yscale('log')
            ax[2][1].legend(loc='upper right')

            print('Plot average hit Y position')
            bins=np.histogram(np.hstack((average_y_shower_geant,average_y_shower_gen)), bins=50)[1]
            #ax[2][2].set_ylabel(r'$\#$ entries')
            #ax[2][2].set_xlabel('Average Y pos.')
            ax[2][2].hist(average_y_shower_geant, bins, alpha=0.5, color='orange', label='Geant4')
            ax[2][2].hist(average_y_shower_gen, bins, alpha=0.5, color='blue', label='Gen')
            #ax[2][2].set_yscale('log')
            ax[2][2].legend(loc='upper right')

            save_name = os.path.join(output_directory,'input_dists.png')
            fig.savefig(save_name)

            
#             # Convert Generated file
            Converter_ = Convertor.Convertor(plot_file_name, 0.0, preprocessor=args.preprocessor)
#            Converter_ = Convertor.Convertor(files_list_[0], 0.0, preprocessor=args.preprocessor)
            Converter_.invert(-99)
            Converter_.digitize()
            Converter_.to_h5py(os.path.join(output_directory, 'Gen.h5'))
# #             # Convert Reference file: TODO: multifile management
#             Converter_ = Convertor.Convertor(files_list_, 0.0, preprocessor=args.preprocessor)
#             Converter_.invert(-99)
#             Converter_.digitize()
# #             Converter_.plot_re(r_gen,E_gen,z_gen)
#             Converter_.to_h5py(os.path.join(output_directory, 'Reference.h5'))
            Geant4_files = []
            # Iterate over all files in the files_list_
            for i, file_path in enumerate(files_list_):
                # Create an instance of Convertor for the current file
                converter = Convertor.Convertor(file_path, 0.0, preprocessor=args.preprocessor)
                
                # Perform the desired operations
                converter.invert(-99)
                converter.digitize()
                
                # Optionally, you can plot or perform other actions here
                # converter.plot_re(r_gen, E_gen, z_gen)

                # Save the output to an H5 file with a unique name for each file
                output_file = os.path.join(output_directory, f'Reference_{i}.h5')
                Geant4_files.append(output_file)
                converter.to_h5py(output_file)
            
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
            plot_energy_z_plt = plot.plot_energy_z()
            r_width_plt = plot.r_width()
            max_voxel_dep_energy_layer_plt = plot.max_voxel_dep_energy_layer()
            wandb.log({"summary" : wandb.Image(plot_energy_r_plt)})
            wandb.log({"summary" : wandb.Image(plot_energy_z_plt)})
            wandb.log({"summary" : wandb.Image(r_width_plt)})
            wandb.log({"summary" : wandb.Image(max_voxel_dep_energy_layer_plt)}) 
            os.system('python3 util/evaluate_image_based.py -m all --output_dir {outdir} --input_file {Gen_file} --reference_file {Geant4_file} --dataset 2'.format(Gen_file = os.path.join(output_directory, 'Gen.h5'), Geant4_file = os.path.join(output_directory, 'Reference_0.h5'), outdir = os.path.join(output_directory, 'calo_score')))
            wandb.log({"summary" : wandb.Image(os.path.join(output_directory, 'calo_score', 'reference_average_shower_dataset_2.png'))})
            wandb.log({"summary" : wandb.Image(os.path.join(output_directory, 'calo_score', 'average_shower_dataset_2.png'))})
            wandb.log({"summary":  wandb.Image(os.path.join(output_directory, 'calo_score', 'voxel_energy_dataset_2.png'))})
   
            # os.system('python3 util/evaluate_image_based.py -m all --output_dir {outdir} --input_file {Gen_file} --reference_file {Geant4_file} --dataset 2'.format(Gen_file = os.path.join(output_directory, 'Gen.h5'), Geant4_file = os.path.join(output_directory, 'Reference.h5'), outdir = os.path.join(output_directory, 'calo_score')))
            # wandb.log({"summary" : wandb.Image(os.path.join(output_directory, 'calo_score', 'reference_average_shower_dataset_2.png'))})
            # wandb.log({"summary" : wandb.Image(os.path.join(output_directory, 'calo_score', 'average_shower_dataset_2.png'))})
            # Gen_file = os.path.join(output_directory, 'Gen.h5')
            # Geant4_file = os.path.join(output_directory, 'Reference.h5')
            # plot = display.High_class_feature_plot(Gen_file,Geant4_file,output_directory)
            # plot_energy_r_plt = plot.plot_energy_r()
            # plot_energy_z_plt = plot.plot_energy_z()
            # r_width_plt = plot.r_width()
            # max_voxel_dep_energy_layer_plt = plot.max_voxel_dep_energy_layer()
            # wandb.log({"summary" : wandb.Image(plot_energy_r_plt)})
            # wandb.log({"summary" : wandb.Image(plot_energy_z_plt)})
            # wandb.log({"summary" : wandb.Image(r_width_plt)})
            # wandb.log({"summary" : wandb.Image(max_voxel_dep_energy_layer_plt)}) 
            # wandb.log({"summary":  wandb.Image(os.path.join(output_directory, 'calo_score', 'voxel_energy_dataset_2.png'))})

if __name__=='__main__':

    usage=''
    argparser = argparse.ArgumentParser(usage)
    argparser.add_argument('-s','--switches',dest='switches', help='Binary representation of switches that run: evaluation plots, training, sampling, evaluation plots', default='0000', type=str)
    argparser.add_argument('-i','--inputs',dest='inputs', help='Path to input directory', default='', type=str)
    argparser.add_argument('-c', '--config', dest='config', help='Configuration file for parameter monitoring (relative path)', default='', type=str)
    argparser.add_argument('-p', '--preprocessor', dest='preprocessor', help='pickle files of preprocessor', default='', type=str)
    argparser.add_argument('--condor', dest = 'condor', default = 0, type=int)
    parsed, unknown = argparser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
        # you can pass any arguments to add_argument
            argparser.add_argument(arg.split('=')[0])

    args = argparser.parse_args()

    print(args)
    if args.condor == 1: # Use condor to run
      main(args)


    else: # Local run

    # WandB configuration
      cfg_name = args.config
      print(f'Using config: {cfg_name}')

      project_name = cfg_name.split('.')[0].split('_', 1)[1].replace("/", "_")
      print(f'Starting project: {project_name}')

   #   if not os.path.exists(cfg_name):
   #     cfg_name = os.path.join('../configs', cfg_name)


      with open(cfg_name) as ymlfile:
        sweep_yml = yaml.safe_load(ymlfile)
    
    # Run main function using sweep agents reading from configs
    # Sweeps run by setting range of parameter values to explore, else set single parameter value
    # Running from yaml files facilitates submitting (several) jobs to condor
      n_runs = 2
      sweep_id = wandb.sweep(sweep_yml, project="NCSM-"+project_name)
      wandb.agent(sweep_id, main, count=n_runs)
    


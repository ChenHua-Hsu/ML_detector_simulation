import torch, sys, os
#sys.path.insert(1, '../')
import data_utils as utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
#plt.use('Agg')
# Enable LaTeX
plt.rcdefaults()
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'  # Optional: Change to a LaTeX-compatible font
plt.rcParams['mathtext.fontset'] = 'cm'
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from typing import Union
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler, PowerTransformer, minmax_scale
from pickle import load
from matplotlib import cm
import evaluate_image_based as eib 

def invert_transform_e(e_):
    original_e = 0.5 * np.log( (1+np.array(e_)) / (1-np.array(e_)) )
    original_e = np.nan_to_num(original_e)
    original_e = np.reshape(original_e,(-1,))
    return original_e

# Pass a list of plots, bins and titles and function will recursively loop through and plot
class recursive_plot:
    def __init__(self, n_plots, name1, vals_list, x_titles, n_bins=0, xvals_list=[], y_titles=[], colors=[]):
        '''
        Plot list of any number plots
        Args:
            nplots: number of plots to make
            name1: save name
            vals_list: list of lists/arrays of values/datapoints to plot
            n_bins: number of bins
            x_titles: x-axis label for each histogram
        '''
        self.n_plots = n_plots
        self.fig, self.ax = plt.subplots(1,n_plots, figsize=(25,4))
        self.fig.suptitle(name1)
        self.vals_list = vals_list
        self.xvals_list = xvals_list
        self.n_bins = n_bins
        self.x_titles = x_titles
        self.y_titles = y_titles
        self.colors = colors
    
    def rec_plot(self):
        if len(self.vals_list) == 0:
            return None
        plot_idx = self.n_plots-len(self.vals_list)
        self.ax[plot_idx].hist(self.vals_list[0], bins=self.n_bins[0])
        self.ax[plot_idx].set_xlabel(self.x_titles[0])
        self.vals_list.pop(0)
        self.n_bins.pop(0)
        self.x_titles.pop(0)
        self.ax[plot_idx].set_yscale('log')
        self.rec_plot()

    def rec_scatter(self):
        if len(self.vals_list) == 0:
            return None
        if len(self.xvals_list) == 0:
            print('WARNING: no xvals provided for scatter plot')
            return None
        plot_idx = self.n_plots-len(self.vals_list)
        self.ax[plot_idx].scatter(self.xvals_list[0],self.vals_list[0])
        self.ax[plot_idx].set_xlabel(self.x_titles[0])
        self.ax[plot_idx].set_ylabel(self.y_titles[0])
        self.vals_list.pop(0)
        self.xvals_list.pop(0)
        self.y_titles.pop(0)
        self.x_titles.pop(0)
        self.ax[plot_idx].set_yscale('log')
        self.rec_scatter()

    def save(self, savename):
        self.fig.savefig(savename)
        return

def plot_loss_vs_epoch(eps_, train_losses, test_losses, odir='', zoom=False):
    
    fig_, ax_ = plt.subplots(ncols=1, figsize=(4,4))
    
    if zoom==True:
        # Only plot the last 80% of the epochs
        ax_.set_title('zoom')
        zoom_split = int(len(train_losses) * 0.8)
    else:
        ax_.set_title('Loss vs. epoch')
        zoom_split = 0
        
    ax_.set_ylabel('Loss')
    ax_.set_xlabel('Epoch')
    ax_.set_yscale('log')
    eps_zoom = eps_[zoom_split:]
    train_loss_zoom = train_losses[zoom_split:]
    test_loss_zoom = test_losses[zoom_split:]
    ax_.plot(eps_zoom,train_loss_zoom, c='blue', label='training')
    ax_.plot(eps_zoom,test_loss_zoom, c='red', label='testing')
    ax_.legend(loc='upper right')
    
    if zoom==True:
        z = np.polyfit(eps_zoom, train_loss_zoom, 1)
        trend = np.poly1d(z)
        ax_.plot(eps_zoom,trend(eps_zoom), c='black', label='trend')
        fig_.savefig(os.path.join(odir,'loss_v_epoch_zoom.png'))
    else:
        fig_.savefig(os.path.join(odir,'loss_v_epoch.png'))
    
    return

def plot_distribution(files_:Union[ list , utils.cloud_dataset], nshowers_2_plot=100, padding_value=0.0, batch_size=1, energy_trans=False, masking=True):
    
    '''
    files_ = can be a list of input files or a cloud dataset object
    nshowers_2_plot = # showers you want to plot. Limits memory required. Samples evenly from several files if files input is used.
    padding_value = value used for padded entries
    batch_size = # showers to load at a time
    energy_trans_file = pickle file containing fitted input transformation function. Only provide a file name if you want to plot distributions where the transformation has been inverted and applied to inputs to transform back to the original distributions.
    '''
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    shower_counter = 0
    shower_hit_energies = []
    shower_hit_x = []
    shower_hit_y = []
    all_z = []
    shower_hit_ine = []
    total_deposited_e_shower = []
    sum_x_shower = []
    sum_y_shower = []
    sum_z_shower = []
    
    sum_e_shower = []
    mean_x_shower = []
    mean_y_shower = []
    mean_z_shower = []
    
    all_incident_e = []
    entries = []
    GeV = 1/1000
    print(f'# showers to plot: {nshowers_2_plot}')
    if type(files_) == list:
        print(f'plot_distribution running on input type \'files\'')
        
        # Using several files so want to take even # samples from each file for plots
        n_files = len(files_)
        print(f'# files: {n_files}')
        nshowers_per_file = [1311,6685,774,613,615]
        #r_ = nshowers_2_plot % nshowers_per_file[0]
        #nshowers_per_file[-1] = nshowers_per_file[-1]+r_
        print(f'# showers per file: {nshowers_per_file}')
        
        for file_idx in range(len(files_)):
            filename = files_[file_idx]
            shower_counter=0
            print(f'File: {filename}')
            fdir = filename.rsplit('/',1)[0]
            custom_data = utils.cloud_dataset(filename, device=device)
            # Note: Shuffling can be turned off if you want to see exactly the same showers before and after transformation
            point_clouds_loader = DataLoader(custom_data, batch_size=batch_size, shuffle=True)
            
            # For each batch in file
            print(f'# batches: {len(point_clouds_loader)}')
            for i, (shower_data,incident_energies) in enumerate(point_clouds_loader,0): 
                valid_hits = []
                data_np = shower_data.cpu().numpy().copy()
                incident_energies = incident_energies.cpu().numpy().copy()
                
                # Mask for padded values
               # mask = ~(data_np[:,:,0] <= 0.01)
                mask = ~(data_np[:,:,0] == padding_value)

                incident_energies = np.array(incident_energies).reshape(-1,1)
                incident_energies = incident_energies.flatten().tolist()
                
                # For each shower in batch
                for j in range(len(data_np)):
                    if shower_counter >= nshowers_per_file[file_idx]:
                        break
                    shower_counter+=1
                    
                    # Only use non-padded values for plots
                    valid_hits = data_np[j]#[mask[j]]
                    if masking:
                        valid_hits = data_np[j][mask[j]]
                    # To transform back to original energies for plots
                    all_e = np.array(valid_hits[:,0]).reshape(-1,1)
                    all_x = np.array(valid_hits[:,1]).reshape(-1,1)
                    all_y = np.array(valid_hits[:,2]).reshape(-1,1)
                    
                    all_e = all_e.flatten().tolist()
                    all_x = all_x.flatten().tolist()
                    all_y = all_y.flatten().tolist()
                    
                    # Store features of individual hits in shower
                    shower_hit_energies.extend( all_e )
                    shower_hit_x.extend( all_x )
                    shower_hit_y.extend( all_y )
                    all_z.extend( ((valid_hits).copy()[:,3]).flatten().tolist() )
                    hits_ine = [ incident_energies[j] for x in range(0,len(valid_hits[:,0])) ]
                    shower_hit_ine.extend( hits_ine )
                    
                    # Store full shower features
                    # Number of valid hits
                    entries.extend( [len(valid_hits)] )
                    # Total energy deposited by shower
                    total_deposited_e_shower.extend([ sum(all_e) ])
                    # Incident energies
                    all_incident_e.extend( [incident_energies[j]] )
                    # Hit means
                    sum_e_shower.extend( [np.sum(all_e)] )
                    mean_x_shower.extend( [np.mean(all_x)] )
                    mean_y_shower.extend( [np.mean(all_y)] )
                    mean_z_shower.extend( [np.mean(all_z)] )
                    
    elif type(files_) == utils.cloud_dataset:
        print(f'plot_distribution running on input type \'cloud_dataset\'')
        
        # Note: Shuffling can be turned off if you want to see specific showers
        point_clouds_loader = DataLoader(files_, batch_size=batch_size, shuffle=False)

        for i, (shower_data,incident_energies) in enumerate(point_clouds_loader,0):
            valid_hits = []
            data_np = shower_data.cpu().numpy().copy()
            energy_np = incident_energies.cpu().numpy().copy()

            with open ("data.txt", "a") as file:
                file.write(str(data_np))
                file.write("\n")
                file.write(str(energy_np))
                file.write("\n")
            
            mask = ~(data_np[:,:,0] == padding_value)
            # For each shower in batch
            for j in range(len(data_np)):
                if shower_counter >= nshowers_2_plot:
                    break
                    
                shower_counter+=1
                valid_hits = data_np[j]#[mask[j]]
                if masking:
                    valid_hits = data_np[j][mask[j]]
                # To transform back to original energies for plots                    
                all_e = np.array(valid_hits[:,0]).reshape(-1,1)
                all_x = np.array(valid_hits[:,1]).reshape(-1,1)
                all_y = np.array(valid_hits[:,2]).reshape(-1,1)
                    
                all_e = all_e.flatten().tolist()
                all_x = all_x.flatten().tolist()
                all_y = all_y.flatten().tolist()
                
                # Store features of individual hits in shower
                shower_hit_energies.extend( all_e )
                shower_hit_x.extend( all_x )
                shower_hit_y.extend( all_y )
                all_z.extend( ((valid_hits).copy()[:,3]).flatten().tolist() )
                
                shower_hit_ine.extend( [energy_np[j] for x in valid_hits[:,0]] ) #Use CPU version of incident_energies
                
                # Number of valid hits
                entries.extend( [len(valid_hits)] )
                # Total energy deposited by shower
                total_deposited_e_shower.extend( [sum(all_e)] )
                # Incident energies
                all_incident_e.extend( [energy_np[j]] )
                # Hit means
                sum_e_shower.extend( [np.sum(all_e)] )
                mean_x_shower.extend( [np.mean(all_x)] )
                mean_y_shower.extend( [np.mean(all_y)] )
                mean_z_shower.extend( [np.mean(all_z)] )

    return [entries, all_incident_e, shower_hit_ine, shower_hit_energies, shower_hit_x, shower_hit_y, all_z, sum_e_shower, mean_x_shower, mean_y_shower, mean_z_shower]

def perturbation_1D(distributions, titles, outdir=''):
    xlabel = distributions[0][0]
    p0, p1, p2, p3, p4, p5 = distributions[0][1]
    
    fig, axs_1 = plt.subplots(1,5, figsize=(24,8), sharey=True)
    bins=np.histogram(np.hstack((p0,p1)), bins=25)[1]
    axs_1[0].set_title(titles[0])
    axs_1[0].set_xlabel(xlabel)
    axs_1[0].hist(p0, bins, alpha=0.5, color='orange', label='un-perturbed')
    axs_1[0].hist(p1, bins, alpha=0.5, color='red', label='perturbed')
    axs_1[0].set_yscale('log')
    axs_1[0].legend(loc='upper right')
    
    bins=np.histogram(np.hstack((p0,p2)), bins=25)[1]
    axs_1[1].set_title(titles[1])
    axs_1[1].hist(p0, bins, alpha=0.5, color='orange', label='un-perturbed')
    axs_1[1].hist(p2, bins, alpha=0.5, color='red', label='perturbed')
    axs_1[1].set_yscale('log')
    axs_1[1].legend(loc='upper right')
    
    bins=np.histogram(np.hstack((p0,p3)), bins=25)[1]
    axs_1[2].set_title(titles[2])
    axs_1[2].hist(p0, bins, alpha=0.5, color='orange', label='un-perturbed')
    axs_1[2].hist(p3, bins, alpha=0.5, color='red', label='perturbed')
    axs_1[2].set_yscale('log')
    axs_1[2].legend(loc='upper right')

    bins=np.histogram(np.hstack((p0,p4)), bins=25)[1]
    axs_1[3].set_title(titles[3])
    axs_1[3].hist(p0, bins, alpha=0.5, color='orange', label='un-perturbed')
    axs_1[3].hist(p4, bins, alpha=0.5, color='red', label='perturbed')
    axs_1[3].set_yscale('log')
    axs_1[3].legend(loc='upper right')
    
    bins=np.histogram(np.hstack((p0,p5)), bins=25)[1]
    axs_1[4].set_title(titles[4])
    axs_1[4].hist(p0, bins, alpha=0.5, color='orange', label='un-perturbed')
    axs_1[4].hist(p5, bins, alpha=0.5, color='red', label='perturbed')
    axs_1[4].set_yscale('log')
    axs_1[4].legend(loc='upper right')
    
    fig.show()
    save_name = xlabel+'_perturbation_1D.png'
    save_name = save_name.replace(' ','').replace('[','').replace(']','')
    save_name = os.path.join(outdir,save_name)
    print(f'save_name: {save_name}')
    fig.savefig(save_name)
    return

def create_axes():

    # define the axis for the first plot
    left, width = 0.1, 0.22
    bottom, height = 0.1, 0.7
    bottom_h = height + 0.15
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter = plt.axes(rect_scatter)
    ax_histx = plt.axes(rect_histx)
    ax_histy = plt.axes(rect_histy)
    
    # define the axis for the first colorbar
    left, width_c = width + left + 0.1, 0.01
    rect_colorbar = [left, bottom, width_c, height]
    ax_colorbar = plt.axes(rect_colorbar)
    
    # define the axis for the transformation plot
    left = left + width_c + 0.2
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter_trans = plt.axes(rect_scatter)
    ax_histx_trans = plt.axes(rect_histx)
    ax_histy_trans = plt.axes(rect_histy)

    # define the axis for the second colorbar
    left, width_c = left + width + 0.1, 0.01
    rect_colorbar_trans = [left, bottom, width_c, height]
    ax_colorbar_trans = plt.axes(rect_colorbar_trans)
    
    return (
        (ax_scatter, ax_histy, ax_histx),
        (ax_scatter_trans, ax_histy_trans, ax_histx_trans),
        (ax_colorbar, ax_colorbar_trans)
    )

def plot_xy(axes, X1, X2, y, ax_colorbar, hist_nbins=50, zlabel="", x0_label="", x1_label="", name=""):
    
    # scale the output between 0 and 1 for the colorbar
    y_full = y
    y = minmax_scale(y_full)
    
    # The scatter plot
    cmap = cm.get_cmap('winter')
    
    ax, hist_X2, hist_X1 = axes
    ax.set_title(name)
    ax.set_xlabel(x0_label)
    ax.set_ylabel(x1_label)
    
    colors = cmap(y)
    ax.scatter(X1, X2, alpha=0.5, marker="o", s=5, lw=0, c=colors)

    # Aesthetics
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    # Histogram for x-axis (along top)
    hist_X1.set_xlim(ax.get_xlim())
    hist_X1.hist(X1, bins=hist_nbins, orientation="vertical", color="red", ec="red")
    hist_X1.axis("off")
    
    # Histogram for y-axis (along RHS)
    hist_X2.set_ylim(ax.get_ylim())
    hist_X2.hist(X2, bins=hist_nbins, orientation="horizontal", color="grey", ec="grey")
    hist_X2.axis("off")
    
    norm = Normalize(min(y_full), max(y_full))
    cb1 = ColorbarBase(
        ax_colorbar,
        cmap=cmap,
        norm=norm,
        orientation="vertical",
        label=zlabel,
    )
    return

def make_plot(distributions, outdir=''):
    
    fig = plt.figure(figsize=(12, 10))
    
    X1, X2, y_X, T1, T2, y_T = distributions[0][1]
    xlabel, ylabel, zlabel = distributions[0][0]

    ax_X, ax_T, ax_colorbar = create_axes()
    axarr = (ax_X, ax_T)
    
    title = 'Non-transformed'
    plot_xy(
        axarr[0],
        X1,
        X2,
        y_X,
        ax_colorbar[0],
        hist_nbins=200,
        x0_label=xlabel,
        x1_label=ylabel,
        zlabel=zlabel,
        name=title
    )
    
    title='Transformed'
    plot_xy(
        axarr[1],
        T1,
        T2,
        y_T,
        ax_colorbar[1],
        hist_nbins=200,
        x0_label=xlabel,
        x1_label=ylabel,
        zlabel=zlabel,
        name=title
    )
    
    save_name = xlabel+'_'+ylabel+'.png'
    save_name = save_name.replace(' ','').replace('[','').replace(']','')
    print(f'save_name: {save_name}')
    fig.savefig(os.path.join(outdir,save_name))
    
    return

def create_axes_diffusion(n_plots):
    
    axes_ = ()
    
    # Define the axis for the first plot
    # Histogram width
    width_h = 0.02
    left = 0.02
    width_buffer = 0.05
    # Scatter plot width
    width = (1-(n_plots*(width_h+width_buffer))-left)/n_plots
    bottom, height = 0.1, 0.7
    bottom_h = height + 0.15
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, width_h, height]

    # Add axes with dimensions above in normalized units
    ax_scatter = plt.axes(rect_scatter)
    # Horizontal histogram along x-axis (Top)
    ax_histx = plt.axes(rect_histx)
    # Vertical histogram along y-axis (RHS)
    ax_histy = plt.axes(rect_histy)
    
    axes_ += ((ax_scatter, ax_histy, ax_histx),)
    
    # define the axis for the next plots
    for idx in range(0,n_plots-1):
        #left = left + width + 0.22
        left = left + width + width_h + width_buffer
        left_h = left + width + 0.01

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.1]
        rect_histy = [left_h, bottom, width_h, height]

        ax_scatter_diff = plt.axes(rect_scatter)
        # Horizontal histogram along x-axis (Top)
        ax_histx_diff = plt.axes(rect_histx)
        # Vertical histogram along y-axis (RHS)
        ax_histy_diff = plt.axes(rect_histy)
        
        axes_ += ((ax_scatter_diff, ax_histy_diff, ax_histx_diff),)
    
    return axes_

def plot_diffusion_xy(axes, X1, X2, GX1, GX2, hist_nbins=50, x0_label="", x1_label="", name="", xlim=(-1,1), ylim=(-1,1)):
    
    # The scatter plot
    ax, hist_X1, hist_X0 = axes
    ax.set_title(name)
    ax.set_xlabel(x0_label)
    ax.set_ylabel(x1_label)
    ax.scatter(GX1, GX2, alpha=0.5, marker="o", s=8, lw=0, c='orange',label='Geant4')
    ax.scatter(X1, X2, alpha=0.5, marker="o", s=8, lw=0, c='blue',label='Gen')
    ax.set_xlim(xlim[0],xlim[1])
    ax.set_ylim(ylim[0],ylim[1])
    ax.legend(loc='upper left')

    # Removing the top and the right spine for aesthetics
    # make nice axis layout
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    # Horizontal histogram along x-axis (Top)
    hist_X0.set_xlim(ax.get_xlim())
    hist_X0.hist(X1, bins=hist_nbins, orientation="vertical", color="grey", ec="grey")
    hist_X0.axis("off")
    
    # Vertical histogram along y-axis (RHS)
    hist_X1.set_ylim(ax.get_ylim())
    hist_X1.hist(X2, bins=hist_nbins, orientation="horizontal", color="red", ec="red")
    hist_X1.axis("off")
    
    return

def make_diffusion_plot(distributions, titles=[], outdir=''):
    
    fig = plt.figure(figsize=(50, 10))
    #fig.set_tight_layout(True)

    xlabel, ylabel = distributions[0][0]
    # Geant4/Gen distributions for x- and y-axes
    geant_x, geant_y, gen_x_t1, gen_y_t1, gen_x_t25, gen_y_t25, gen_x_t50, gen_y_t50, gen_x_t75, gen_y_t75, gen_x_t99, gen_y_t99  = distributions[0][1]
    
    # Number of plots depends on the number of diffusion steps to plot
    n_plots = (len(distributions[0][1])-2)/2
    
    # Labels of variables to plot
    ax_X, ax_T1, ax_T2, ax_T3, ax_T4 = create_axes_diffusion(int(n_plots))
    axarr = (ax_X, ax_T1, ax_T2, ax_T3, ax_T4)
    
    x_lim = ( min(min(gen_x_t1),min(geant_x)) , max(max(gen_x_t1),max(geant_x)) )
    y_lim = ( min(min(gen_y_t1),min(geant_y)) , max(max(gen_y_t1),max(geant_y)) )
    
    plot_diffusion_xy(
        axarr[0],
        gen_x_t1,
        gen_y_t1,
        geant_x,
        geant_y,
        hist_nbins=50,
        x0_label=xlabel,
        x1_label=ylabel,
        name=f't={titles[0]} (noisy)',
        xlim=x_lim,
        ylim=y_lim
    )
    
    x_lim = ( min(min(gen_x_t25),min(geant_x)) , max(max(gen_x_t25),max(geant_x)) )
    y_lim = ( min(min(gen_y_t25),min(geant_y)) , max(max(gen_y_t25),max(geant_y)) )
    plot_diffusion_xy(
        axarr[1],
        gen_x_t25,
        gen_y_t25,
        geant_x,
        geant_y,
        hist_nbins=50,
        x0_label=xlabel,
        x1_label=ylabel,
        name=f't={titles[1]}',
        xlim=x_lim,
        ylim=y_lim
    )
    
    x_lim = ( min(min(gen_x_t50),min(geant_x)) , max(max(gen_x_t50),max(geant_x)) )
    y_lim = ( min(min(gen_y_t50),min(geant_y)) , max(max(gen_y_t50),max(geant_y)) )
    plot_diffusion_xy(
        axarr[2],
        gen_x_t50,
        gen_y_t50,
        geant_x,
        geant_y,
        hist_nbins=50,
        x0_label=xlabel,
        x1_label=ylabel,
        name=f't={titles[2]}',
        xlim=x_lim,
        ylim=y_lim
    )
    
    x_lim = ( min(min(gen_x_t75),min(geant_x)) , max(max(gen_x_t75),max(geant_x)) )
    y_lim = ( min(min(gen_y_t75),min(geant_y)) , max(max(gen_y_t75),max(geant_y)) )
    plot_diffusion_xy(
        axarr[3],
        gen_x_t75,
        gen_y_t75,
        geant_x,
        geant_y,
        hist_nbins=50,
        x0_label=xlabel,
        x1_label=ylabel,
        name=f't={titles[3]}',
        xlim=x_lim,
        ylim=y_lim
    )
    
    x_lim = ( min(min(gen_x_t99),min(geant_x)) , max(max(gen_x_t99),max(geant_x)) )
    y_lim = ( min(min(gen_y_t99),min(geant_y)) , max(max(gen_y_t99),max(geant_y)) )
    plot_diffusion_xy(
        axarr[4],
        gen_x_t99,
        gen_y_t99,
        geant_x,
        geant_y,
        hist_nbins=50,
        x0_label=xlabel,
        x1_label=ylabel,
        name=f't={titles[4]}',
        xlim=x_lim,
        ylim=y_lim
    )
    print(f'plt.axis(): {plt.axis()}')

    save_name = xlabel+'_'+ylabel+'.png'
    save_name = save_name.replace(' ','').replace('[','').replace(']','')
    save_name = os.path.join(outdir,save_name)
    print(f'save_name: {save_name}')
    fig.savefig(save_name)

    return

def make_diffusion_plot_v2(distributions, titles=[], outdir='', steps=[]):
  for key in distributions:
    fig = plt.figure(figsize=(50, 10))
    n_plots = len(steps)
    axarr = create_axes_diffusion(n_plots)
    for idx, step in enumerate(steps):
      xlabel, ylabel = distributions[key][step][0]
      geant_x, geant_y, gen_x_t, gen_y_t = distributions[key][step][1]
      x_lim = (min(min(gen_x_t), min(geant_x)), max(max(gen_x_t), max(geant_x)))
      y_lim = (min(min(gen_y_t), min(geant_y)), max(max(gen_y_t), max(geant_y)))
      plot_diffusion_xy(
         axarr[idx],
         gen_x_t,
         gen_y_t,
         geant_x,
         geant_y,
         hist_nbins=50,
         x0_label=xlabel,
         x1_label=ylabel,
         name=f't={step} (noisy)',
         xlim = x_lim,
         ylim = y_lim
      )
    fig.savefig(os.path.join(outdir, '{}_diffusion_2D.png'.format(key)))
  return

def comparison_summary(dists, dists_gen, sampling_directory, erange=(), xrange=(), yrange=(), zrange=()):
    
    entries = dists[0]
    all_incident_e = dists[1]
    all_hit_ine_geant = dists[2]
    all_e = dists[3]
    all_x = dists[4]
    all_y = dists[5]
    all_z = dists[6]
    all_e = np.array(all_e)
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_z = np.array(all_z)
    average_e_shower_geant = dists[7]
    average_x_shower_geant = dists[8]
    average_y_shower_geant = dists[9]
    
    entries_gen = dists_gen[0]
    all_incident_e_gen = dists_gen[1]
    all_hit_ine_gen = dists_gen[2]
    all_e_gen = dists_gen[3]
    all_x_gen = dists_gen[4]
    all_y_gen = dists_gen[5]
    all_z_gen = dists_gen[6]
    average_e_shower_gen = dists_gen[7]
    average_x_shower_gen = dists_gen[8]
    average_y_shower_gen = dists_gen[9]
    all_e_gen = np.array(all_e_gen)
    all_x_gen = np.array(all_x_gen)
    all_y_gen = np.array(all_y_gen)
    all_z_gen = np.array(all_z_gen)

    fig, ax = plt.subplots(3,3, figsize=(12,12))
    bins=np.histogram(np.hstack((entries,entries_gen)), bins=70)[1]
    #ax[0][0].set_ylabel('entries')
    #ax[0][0].set_xlabel('Hit entries')
    ax[0][0].hist(entries, bins, alpha=0.5, color='orange', label='Geant4')
    ax[0][0].hist(entries_gen, bins, alpha=0.5, color='blue', label='Gen')
    ax[0][0].legend(loc='upper right')

    
    #ax[0][1].set_ylabel('entries')
    #ax[0][1].set_xlabel('Hit energy [GeV]')
    # counts, bin_edges = np.histogram(all_e_gen, bins=100)

    # # Calculate the cumulative sum and ratio
    # cumulative_counts = np.cumsum(counts)
    # total_count = cumulative_counts[-1]
    # tail_ratio = 1e-5  # Set your ratio here

    # # Find the cutoff index for the positive side
    # positive_cutoff_index = np.where(cumulative_counts / total_count > tail_ratio)[0][0]
    # positive_cutoff_value = bin_edges[positive_cutoff_index]

    # # Find the cutoff index for the negative side
    # negative_cutoff_index = np.where(cumulative_counts[::-1] / total_count > tail_ratio)[0][0]
    # negative_cutoff_value = bin_edges[-negative_cutoff_index-1]

    # print(negative_cutoff_value)
    # print(positive_cutoff_value)
    # all_e_gen = np.array(all_e_gen)
    # print(all_e_gen)

    # # Filter data
    # filtered_data = all_e_gen[(all_e_gen >= negative_cutoff_value) & (all_e_gen <= positive_cutoff_value)]
    # all_e_gen = filtered_data 
    # counts, bin_edges = np.histogram(all_e_gen, bins=100)
    # print(counts)
    # count_max = np.max(counts)

    # # Find the cutoff index for the positive side
    # min = 0
    # max = 0
    # negative = 0
    # for i in range(len(counts)-1):
        
        
    #     if counts[i] < count_max * 1e-4 and counts[i+1] > count_max * 1e-4 and negative == 0:
    #         negative = 1
    #         min = bin_edges[i]
    #     if counts[i] > count_max * 1e-4 and counts[i+1] < count_max * 1e-4 and negative == 1:
    #         #negative = 0
    #         max = bin_edges[i+1]
    #         break
    # #index = np.where( counts <= count_max* 0.2)
    # filtered_indices = np.where(counts >= count_max * 0.05)
    #all_e_gen = all_e_gen[(all_e_gen >= min) & (all_e_gen <= max)]
    all_e_gen = all_e_gen[(all_e_gen >= -15.) & (all_e_gen <= 0.)]

    

    if len(erange)==0:
        bins=np.histogram(np.hstack((all_e,all_e_gen)), bins=100)[1]
        ax[0][1].hist(all_e, bins=bins, alpha=0.5, color='orange', label='Geant4')
        ax[0][1].hist(all_e_gen, bins=bins, alpha=0.5, color='blue', label='Gen')
    else:
        ax[0][1].hist(all_e, bins=40, range=erange, alpha=0.5, color='orange', label='Geant4')
        ax[0][1].hist(all_e_gen, bins=40, range=erange, alpha=0.5, color='blue', label='Gen')
    ax[0][1].set_yscale('log')
    ax[0][1].legend(loc='upper right')

    
    #ax[0][2].set_ylabel('entries')
    #ax[0][2].set_xlabel('Hit x position')
    all_x_gen = np.array(all_x_gen)
    all_x_gen = all_x_gen[(all_x_gen<=1) & (all_x_gen>=0)]
    if len(xrange) == 0:
        bins=np.histogram(np.hstack((all_x,all_x_gen)), bins=50)[1]
        ax[0][2].hist(all_x, bins=bins, alpha=0.5, color='orange', label='Geant4')
        ax[0][2].hist(all_x_gen, bins=bins, alpha=0.5, color='blue', label='Gen')
    else:
        ax[0][2].hist(all_x, bins=40, range=xrange, alpha=0.5, color='orange', label='Geant4')
        ax[0][2].hist(all_x_gen, bins=40, range=xrange, alpha=0.5, color='blue', label='Gen')
    ax[0][2].set_yscale('log')
    ax[0][2].legend(loc='upper right')

    #ax[1][0].set_ylabel('entries')
    #ax[1][0].set_xlabel('Hit y position')
    all_y_gen = np.array(all_y_gen)
    all_y_gen = all_y_gen[(all_y_gen<=1) & (all_y_gen>=0)]
    if len(yrange)==0:
        bins=np.histogram(np.hstack((all_y,all_y_gen)), bins=50)[1]
        ax[1][0].hist(all_y, bins=bins, alpha=0.5, color='orange', label='Geant4')
        ax[1][0].hist(all_y_gen, bins=bins, alpha=0.5, color='blue', label='Gen')
    else:
        ax[1][0].hist(all_y, bins=40, range=yrange, alpha=0.5, color='orange', label='Geant4')
        ax[1][0].hist(all_y_gen, bins=40, range=yrange, alpha=0.5, color='blue', label='Gen')
    ax[1][0].set_yscale('log')
    ax[1][0].legend(loc='upper right')

    #ax[1][1].set_ylabel('entries')
    #ax[1][1].set_xlabel('Hit z position')
    all_z_gen = all_z_gen[(all_z_gen<=1) & (all_z_gen>=0)]
    if len(zrange)==0:
        bins=np.histogram(np.hstack((all_z,all_z_gen)), bins=50)[1]
        ax[1][1].hist(all_z, bins=bins, alpha=0.5, color='orange', label='Geant4')
        ax[1][1].hist(all_z_gen, bins=bins, alpha=0.5, color='blue', label='Gen')
    else:
        ax[1][1].hist(all_z, bins=10, range=zrange, alpha=0.5, color='orange', label='Geant4')
        ax[1][1].hist(all_z_gen, bins=10, range=zrange, alpha=0.5, color='blue', label='Gen')
    ax[1][1].set_yscale('log')
    ax[1][1].legend(loc='upper right')

    bins=np.histogram(np.hstack((all_incident_e,all_incident_e_gen)), bins=25)[1]
    #ax[1][2].set_ylabel('entries')
    #ax[1][2].set_xlabel('Incident energies [GeV]')
    ax[1][2].hist(all_incident_e, bins, alpha=0.5, color='orange', label='Geant4')
    ax[1][2].hist(all_incident_e_gen, bins, alpha=0.5, color='blue', label='Gen')
    ax[1][2].set_yscale('log')
    ax[1][2].legend(loc='upper right')

    bins=np.histogram(np.hstack((average_e_shower_geant,average_e_shower_gen)), bins=25)[1]
    #ax[2][0].set_ylabel('entries')
    #ax[2][0].set_xlabel('Mean hit energy [GeV]')
    ax[2][0].hist(average_e_shower_geant, bins, alpha=0.5, color='orange', label='Geant4')
    ax[2][0].hist(average_e_shower_gen, bins, alpha=0.5, color='blue', label='Gen')
    ax[2][0].set_yscale('log')
    ax[2][0].legend(loc='upper right')

    bins=np.histogram(np.hstack((average_x_shower_geant,average_x_shower_gen)), bins=25)[1]
    #ax[2][1].set_ylabel('entries')
    #ax[2][1].set_xlabel('Shower Mean X')
    ax[2][1].hist(average_x_shower_geant, bins, alpha=0.5, color='orange', label='Geant4')
    ax[2][1].hist(average_x_shower_gen, bins, alpha=0.5, color='blue', label='Gen')
    ax[2][1].set_yscale('log')
    ax[2][1].legend(loc='upper right')

    bins=np.histogram(np.hstack((average_y_shower_geant,average_y_shower_gen)), bins=25)[1]
    #ax[2][2].set_ylabel('entries')
    #ax[2][2].set_xlabel('Shower Mean Y')
    ax[2][2].hist(average_y_shower_geant, bins, alpha=0.5, color='orange', label='Geant4')
    ax[2][2].hist(average_y_shower_gen, bins, alpha=0.5, color='blue', label='Gen')
    ax[2][2].set_yscale('log')
    ax[2][2].legend(loc='upper right')
    
    print(f'Saving comparison plots to: {sampling_directory}') 
    #plt.savefig( os.path.join(sampling_directory,'comparison.png') )

    return fig
def correlation(variable1,varaible2):
    e = variable1  # energy
    theta = varaible2  # theta (phi in cylindrical coordinates)

    # Calculate correlation coefficient
    e_mean = e.mean(dim=1, keepdim=True)
    theta_mean = theta.mean(dim=1, keepdim=True)
    
    e_centered = e - e_mean
    
    theta_centered = theta - theta_mean
    e_std = e_centered.std(dim=1)
    theta_std = theta_centered.std(dim=1)
    

    covariance_et = (e_centered * theta_centered).mean(dim=1)
    
    
    correlation_et = covariance_et / (e_std * theta_std + 1e-8)  # Avoid division by zero
    

    # Add correlation penalty to the loss (e.g., L2 norm of correlation)
    correlation_penalty_et = torch.mean(correlation_et**2)
    
    #alpha = 0.5 # Weight of the correlation penalty

    return correlation_penalty_et

import h5py
class High_class_feature_plot:
    def __init__(self, source_file, reference_file, output_dir):
        self.r_bins = 9
        self.theata_bins = 16
        self.z_bins = 45
        self.shower_gen, self.ine_gen = eib.extract_shower_and_energy(source_file, which='input')
        self.shower_ref, self.ine_ref = eib.extract_shower_and_energy(reference_file, which='input')
        self.shower_num = self.shower_gen.shape[0]
        self.shower_gen =np.array(self.shower_gen)
        self.shower_ref = np.array(self.shower_ref)
        self.shower_ref = self.shower_ref[:self.shower_num]
        self.reshaped_shower_gen = self.shower_gen.reshape(self.shower_num, self.z_bins, self.theata_bins, self.r_bins)
        self.reshaped_shower_ref = self.shower_ref.reshape(self.shower_num, self.z_bins, self.theata_bins, self.r_bins)
        self.output_dir = output_dir

    def plot_energy_r(self):
        # Plot: Mean energy per cell
        energy_per_r_layer = []
        energy_per_r_layer_ref = [] 
        for i in range(self.r_bins):
            energy_per_r_layer.append(self.reshaped_shower_gen[:,:,:,i].sum() / self.shower_num)
            energy_per_r_layer_ref.append(self.reshaped_shower_ref[:,:,:,i].sum() / self.shower_num)

        energy_per_r_layer = np.array(energy_per_r_layer)
        energy_per_r_layer_ref = np.array(energy_per_r_layer_ref)
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6))
        ax1.plot(energy_per_r_layer, label='Gen', color='blue', marker='o')
        ax1.plot(energy_per_r_layer_ref, label='Reference', color='red', marker = 'x')
        ax1.set_xlabel('r-bin')
        ax1.set_ylabel('energy')
        ax1.set_title('energy vs r-bin')
        ax1.legend()
        percent_diff = 100 * (energy_per_r_layer_ref - energy_per_r_layer) / energy_per_r_layer_ref
        ax2.plot(percent_diff, label='Diff. (%)', color='purple', linewidth=1)
        ax2.set_ylabel('Diff. (%)')
        ax2.axhline(y=0, color='gray', linestyle='--')
        ax2.set_xlabel('layers')

        plt.tight_layout()


        # Save the plot
        plt.savefig('energy_r.png')
        return fig


    def plot_energy_z(self):
        # Plot: Mean energy per cell
        energy_per_z_layer = []
        energy_per_z_layer_ref = []
        for i in range(self.z_bins):
            energy_per_z_layer.append(self.reshaped_shower_gen[:,i,:,:].sum() / self.shower_num)
            energy_per_z_layer_ref.append(self.reshaped_shower_ref[:,i,:,:].sum() / self.shower_num)
        #energy_per_z_layer = self.reshaped_shower_gen.sum(axis=(0, 1, 3)) / self.shower_num
        #energy_per_z_layer_ref = self.reshaped_shower_ref.sum(axis=(0, 1, 3)) / self.shower_num
        energy_per_z_layer = np.array(energy_per_z_layer)
        energy_per_z_layer_ref = np.array(energy_per_z_layer_ref)
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6))
        ax1.plot(energy_per_z_layer, label='Gen', color='blue', marker='o')
        ax1.plot(energy_per_z_layer_ref, label='Reference', color='red', marker = 'x')
        ax1.set_xlabel('layers')
        ax1.set_ylabel('mean dep energy')
        ax1.set_title('mean dep energy vs layers')
        ax1.legend()
        percent_diff = 100 * (energy_per_z_layer_ref - energy_per_z_layer) / energy_per_z_layer_ref
        ax2.plot(percent_diff, label='Diff. (%)', color='purple', linewidth=1)
        ax2.set_ylabel('Diff. (%)')
        ax2.axhline(y=0, color='gray', linestyle='--')
        ax2.set_xlabel('layers')

        plt.tight_layout()

        # Save the plot
        plt.savefig('mean_dep_z.png')
        return fig

    def r_width(self):
        r_width_gen = []
        r_width_ref = []
        r_square_mean_gen = np.zeros_like(self.reshaped_shower_gen)
        r_square_mean_ref = np.zeros_like(self.reshaped_shower_ref)
        r_mean_square_gen = np.zeros_like(self.reshaped_shower_gen)
        r_mean_square_ref = np.zeros_like(self.reshaped_shower_ref)
        print(self.reshaped_shower_gen.shape)
        print(self.reshaped_shower_gen.sum()) 
        for j in range(self.r_bins):
            r_square_mean_gen[:,:,:,j] = self.reshaped_shower_gen[:,:,:,j]*j**2/self.reshaped_shower_gen.sum()
            r_square_mean_ref[:,:,:,j] = self.reshaped_shower_ref[:,:,:,j]*j**2/self.reshaped_shower_ref.sum()
            r_mean_square_gen[:,:,:,j] = (self.reshaped_shower_gen[:,:,:,j]*j/self.reshaped_shower_gen.sum())**2
            r_mean_square_ref[:,:,:,j] = (self.reshaped_shower_ref[:,:,:,j]*j/self.reshaped_shower_ref.sum())**2
                #r_square_mean_gen.append(self.reshaped_shower_gen[:,:,:,j]*j**2/self.reshaped_shower_gen.sum())
                #r_square_mean_ref.append(self.reshaped_shower_ref[:,:,:,j]*j**2/self.reshaped_shower_ref.sum())
                #r_mean_square_gen.append((self.reshaped_shower_gen[:,:,:,j]*j/self.reshaped_shower_gen.sum())**2)
                #r_mean_square_ref.append((self.reshaped_shower_ref[:,:,:,j]*j/self.reshaped_shower_ref.sum())**2)
        for i in range(self.z_bins):
            r_width_gen.append( np.sqrt((r_square_mean_gen[:,i,:,:]-r_mean_square_gen[:,i,:,:]).sum()))
            r_width_ref.append(np.sqrt((r_square_mean_ref[:,i,:,:]-r_mean_square_ref[:,i,:,:]).sum()))
        r_width_gen = np.array(r_width_gen)
        r_width_ref = np.array(r_width_ref)
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6))
        ax1.plot(r_width_gen, label='Gen', color='blue', marker='o')
        ax1.plot(r_width_ref, label='Reference', color='red', marker = 'x')
        ax1.set_xlabel('r-bin')
        ax1.set_ylabel('width')
        ax1.set_title('width vs r-bin')
        ax1.legend()
        percent_diff = 100 * (r_width_ref - r_width_gen) / r_width_ref
        ax2.plot(percent_diff, label='Diff. (%)', color='purple', linewidth=1)
        ax2.set_ylabel('Diff. (%)')
        ax2.axhline(y=0, color='gray', linestyle='--')
        ax2.set_xlabel('r-bins')
        plt.tight_layout()

        # Save the plot
        plt.savefig('r_width.png')
        return fig

    def max_voxel_dep_energy_layer(self):
        # a 2d np array with shape (shower_num, z_bins)
        max_voxel_dep_energy_gen = np.empty((self.shower_num, self.z_bins))
        max_voxel_dep_energy_ref = np.empty((self.shower_num, self.z_bins))
        #max_voxel_dep_energy_gen = np.array
        #max_voxel_dep_energy_ref = []
        print(self.reshaped_shower_gen.shape)
        print(f"Type of self.reshaped_shower_gen: {type(self.reshaped_shower_gen)}")
        for j in range(self.shower_num):
            for i in range(self.z_bins):
                gen = self.reshaped_shower_gen[j,i,:,:]
                max_voxel_dep_energy_gen[j,i]=(self.reshaped_shower_gen[j,i,:,:].max()/(self.reshaped_shower_gen[j,i,:,:].sum()+1e-8))
                max_voxel_dep_energy_ref[j,i]=(self.reshaped_shower_ref[j,i,:,:].max()/(self.reshaped_shower_ref[j,i,:,:].sum()+1e-8))
        max_voxel_dep_energy_gen = np.array(max_voxel_dep_energy_gen.mean(axis=0))
        max_voxel_dep_energy_ref = np.array(max_voxel_dep_energy_ref.mean(axis=0))
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6))
        ax1.plot(max_voxel_dep_energy_gen, label='Gen', color='blue', marker='o')
        ax1.plot(max_voxel_dep_energy_ref, label='Reference', color='red', marker = 'x')
        ax1.set_xlabel('layers')
        ax1.set_ylabel('max voxel dep energy')
        ax1.set_title('max voxel dep energy vs layers')
        ax1.legend()
        percent_diff = 100 * (max_voxel_dep_energy_ref - max_voxel_dep_energy_gen) / max_voxel_dep_energy_ref
        ax2.plot(percent_diff, label='Diff. (%)', color='purple', linewidth=1)
        ax2.set_ylabel('Diff. (%)')
        ax2.axhline(y=0, color='gray', linestyle='--')
        ax2.set_xlabel('layers')

        plt.tight_layout()
        
        # Save the plot
        plt.savefig('max_voxel_dep_energy.png')

        return fig
    
class High_class_feature_plot_test:
    def __init__(self, source_file, reference_files, output_dir):
        self.r_bins = 9
        self.theata_bins = 16
        self.z_bins = 45
        source_file = h5py.File(source_file, 'r')
        self.shower_gen, self.ine_gen = eib.extract_shower_and_energy(source_file, which='input')

        # Initialize lists to store concatenated data and shower counts
        self.shower_ref_list = []
        self.ine_ref_list = []
        self.shower_nums = []

        i = 0 
        self.batches = [1311,6685,774,613,615]#[1311,615,613,774,6685,2]#[774,613,2,6685,615,1311]
        # Load and concatenate all reference files, and keep track of shower counts
        for ref_file in reference_files:
            print(ref_file)
            ref_file = h5py.File(ref_file, 'r')
            shower_ref, ine_ref = eib.extract_shower_and_energy(ref_file, which='input')
            batch = self.batches[i] 
            self.shower_ref_list.append(shower_ref[:batch])
            # if self.shower_ref_list[i] is torch.Tensor:
            #print(self.shower_ref_list[i].shape())
            self.ine_ref_list.append(ine_ref)
            #self.shower_nums.append(self.batches[i])  # Store the number of showers for each reference fileq
            print(self.batches[i])
            print(self.batches)
            print(batch)
            print(i)
            i += 1
            #print(ref_file)
            #print(self.batches[i])
        # Concatenate the data from all reference files
        self.shower_ref = np.concatenate(self.shower_ref_list, axis=0)
        self.ine_ref = np.concatenate(self.ine_ref_list, axis=0)

        # Convert to NumPy arrays
        self.shower_gen = np.array(self.shower_gen)
        self.shower_ref = np.array(self.shower_ref)

        # Ensure the same number of showers in both gen and ref data
        self.shower_num = self.shower_gen.shape[0]
        #self.shower_gen = self.shower_gen[:self.shower_num]
        #self.shower_ref = self.shower_ref[:self.shower_num]

        # Reshape the data
        self.reshaped_shower_gen = self.shower_gen.reshape(self.shower_num, self.z_bins, self.theata_bins, self.r_bins)
        self.reshaped_shower_ref = self.shower_ref.reshape(self.shower_num, self.z_bins, self.theata_bins, self.r_bins)

        self.output_dir = output_dir

    def plot_energy_r(self):
        # Calculate mean energy per r-bin for generated data
        energy_per_r_layer_gen = []
        energy_per_r_layer_ref = [] 
        for i in range(self.r_bins):
            energy_per_r_layer_gen.append(self.reshaped_shower_gen[:, :, :, i].sum() / self.shower_num)
            energy_per_r_layer_ref.append(self.reshaped_shower_ref[:, :, :, i].sum() / self.shower_num)
            print("type: ",type(self.reshaped_shower_ref))
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
        
        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6))
        ax1.plot(energy_per_r_layer_gen, label='Gen', color='blue', marker='o')
        ax1.plot(energy_per_r_layer_ref, label='Reference', color='red', marker='x')
        ax1.set_xlabel('r-bin')
        ax1.set_ylabel('energy')
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
        return fig
    def plot_energy_z(self):
        # Plot: Mean energy per cell
        energy_per_z_layer = []
        energy_per_z_layer_ref = []
        for i in range(self.z_bins):
            energy_per_z_layer.append(self.reshaped_shower_gen[:,i,:,:].sum() / self.shower_num)
            energy_per_z_layer_ref.append(self.reshaped_shower_ref[:,i,:,:].sum() / self.shower_num)
        #energy_per_z_layer = self.reshaped_shower_gen.sum(axis=(0, 1, 3)) / self.shower_num
        #energy_per_z_layer_ref = self.reshaped_shower_ref.sum(axis=(0, 1, 3)) / self.shower_num
        energy_per_z_layer = np.array(energy_per_z_layer)
        energy_per_z_layer_ref = np.array(energy_per_z_layer_ref)
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6))
        ax1.plot(energy_per_z_layer, label='Gen', color='blue', marker='o')
        ax1.plot(energy_per_z_layer_ref, label='Reference', color='red', marker = 'x')
        ax1.set_xlabel('layers')
        ax1.set_ylabel('mean dep energy')
        ax1.set_title('mean dep energy vs layers')
        ax1.legend()
        percent_diff = 100 * (energy_per_z_layer_ref - energy_per_z_layer) / energy_per_z_layer_ref
        ax2.plot(percent_diff, label='Diff. (%)', color='purple', linewidth=1)
        ax2.set_ylabel('Diff. (%)')
        ax2.axhline(y=0, color='gray', linestyle='--')
        ax2.set_xlabel('layers')

        plt.tight_layout()

        # Save the plot
        plt.savefig('mean_dep_z_1.png')

        return fig

    def r_width(self):
        r_width_gen = []
        r_width_ref = []
        r_square_mean_gen = np.zeros_like(self.reshaped_shower_gen)
        r_square_mean_ref = np.zeros_like(self.reshaped_shower_ref)
        r_mean_square_gen = np.zeros_like(self.reshaped_shower_gen)
        r_mean_square_ref = np.zeros_like(self.reshaped_shower_ref)
        print(self.reshaped_shower_gen.shape)
        print(self.reshaped_shower_gen.sum()) 
        for j in range(self.r_bins):
            r_square_mean_gen[:,:,:,j] = self.reshaped_shower_gen[:,:,:,j]*j**2/self.reshaped_shower_gen.sum()
            r_square_mean_ref[:,:,:,j] = self.reshaped_shower_ref[:,:,:,j]*j**2/self.reshaped_shower_ref.sum()
            r_mean_square_gen[:,:,:,j] = (self.reshaped_shower_gen[:,:,:,j]*j/self.reshaped_shower_gen.sum())**2
            r_mean_square_ref[:,:,:,j] = (self.reshaped_shower_ref[:,:,:,j]*j/self.reshaped_shower_ref.sum())**2
                #r_square_mean_gen.append(self.reshaped_shower_gen[:,:,:,j]*j**2/self.reshaped_shower_gen.sum())
                #r_square_mean_ref.append(self.reshaped_shower_ref[:,:,:,j]*j**2/self.reshaped_shower_ref.sum())
                #r_mean_square_gen.append((self.reshaped_shower_gen[:,:,:,j]*j/self.reshaped_shower_gen.sum())**2)
                #r_mean_square_ref.append((self.reshaped_shower_ref[:,:,:,j]*j/self.reshaped_shower_ref.sum())**2)
        for i in range(self.z_bins):
            r_width_gen.append( np.sqrt((r_square_mean_gen[:,i,:,:]-r_mean_square_gen[:,i,:,:]).sum()))
            r_width_ref.append(np.sqrt((r_square_mean_ref[:,i,:,:]-r_mean_square_ref[:,i,:,:]).sum()))
        r_width_gen = np.array(r_width_gen)
        r_width_ref = np.array(r_width_ref)
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6))
        ax1.plot(r_width_gen, label='Gen', color='blue', marker='o')
        ax1.plot(r_width_ref, label='Reference', color='red', marker = 'x')
        ax1.set_xlabel('r-bin')
        ax1.set_ylabel('width')
        ax1.set_title('width vs z-bin')
        ax1.legend()
        percent_diff = 100 * (r_width_ref - r_width_gen) / r_width_ref
        ax2.plot(percent_diff, label='Diff. (%)', color='purple', linewidth=1)
        ax2.set_ylabel('Diff. (%)')
        ax2.axhline(y=0, color='gray', linestyle='--')
        ax2.set_xlabel('z-bins')
        plt.tight_layout()

        # Save the plot
        plt.savefig('r_width_1.png')

        return fig

    def max_voxel_dep_energy_layer(self):
        # a 2d np array with shape (shower_num, z_bins)
        max_voxel_dep_energy_gen = np.empty((self.shower_num, self.z_bins))
        max_voxel_dep_energy_ref = np.empty((self.shower_num, self.z_bins))
        #max_voxel_dep_energy_gen = np.array
        #max_voxel_dep_energy_ref = []
        print(self.reshaped_shower_gen.shape)
        print(f"Type of self.reshaped_shower_gen: {type(self.reshaped_shower_gen)}")
        for j in range(self.shower_num):
            for i in range(self.z_bins):
                gen = self.reshaped_shower_gen[j,i,:,:]
                max_voxel_dep_energy_gen[j,i]=(self.reshaped_shower_gen[j,i,:,:].max()/(self.reshaped_shower_gen[j,i,:,:].sum()+1e-8))
                max_voxel_dep_energy_ref[j,i]=(self.reshaped_shower_ref[j,i,:,:].max()/(self.reshaped_shower_ref[j,i,:,:].sum()+1e-8))
        max_voxel_dep_energy_gen = np.array(max_voxel_dep_energy_gen.mean(axis=0))
        max_voxel_dep_energy_ref = np.array(max_voxel_dep_energy_ref.mean(axis=0))
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6))
        ax1.plot(max_voxel_dep_energy_gen, label='Gen', color='blue', marker='o')
        ax1.plot(max_voxel_dep_energy_ref, label='Reference', color='red', marker = 'x')
        ax1.set_xlabel('layers')
        ax1.set_ylabel('max voxel dep energy')
        ax1.set_title('max voxel dep energy vs layers')
        ax1.legend()
        percent_diff = 100 * (max_voxel_dep_energy_ref - max_voxel_dep_energy_gen) / max_voxel_dep_energy_ref
        ax2.plot(percent_diff, label='Diff. (%)', color='purple', linewidth=1)
        ax2.set_ylabel('Diff. (%)')
        ax2.axhline(y=0, color='gray', linestyle='--')
        ax2.set_xlabel('layers')

        plt.tight_layout()
        
        # Save the plot
        plt.savefig('max_voxel_dep_energy_1.png')

        return fig
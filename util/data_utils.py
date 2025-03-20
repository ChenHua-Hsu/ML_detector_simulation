import h5py, math, torch, fnmatch, os
import numpy as np
from torch.nn.functional import normalize
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler
import torch.nn.functional as F
import random

class cloud_dataset(Dataset):
  def __init__(self, filename, transform=None, transform_y=None, device='cpu'):
    loaded_file = torch.load(filename, map_location=torch.device(device))
    self.data = loaded_file[0]
    self.condition = torch.as_tensor(loaded_file[1]).float()
    if len(loaded_file) > 2:
      self.weight = loaded_file[2]
    else:
      self.weight = [torch.ones(len(self.data[idx])) for idx in range(len(self.data))]
    self.min_y = torch.min(self.condition)
    self.max_y = torch.max(self.condition)
    self.max_nhits = -1
    self.transform = transform
    self.transform_y = transform_y
    self.device = device

  def __getitem__(self, index):
    x = self.data[index]
    y = self.condition[index]
    if self.transform:
        x = self.transform(x,y,self.device)
    if self.transform_y:
       y = self.transform_y(y, self.min_y, self.max_y)
    return x,y
  
  def __len__(self):
    return len(self.data)
  
  # def padding(self, value = 0.0):
  #   for showers in self.data:
  #       if len(showers) > self.max_nhits:
  #           self.max_nhits = len(showers)

  #   padded_showers = []
  #   padded_weights = []
  #   for showers in self.data:
  #     pad_hits = self.max_nhits-len(showers)
  #     padded_shower = F.pad(input = showers, pad=(0,0,0,pad_hits), mode='constant', value = value)
  #     padded_showers.append(padded_shower)
  #   for weights in self.weight:
  #     pad_hits = self.max_nhits - len(weights)
  #     padded_weight = F.pad(input = weights, pad = (0,pad_hits), mode='constant', value=0.0)
  #     padded_weights.append(padded_weight)
  #   self.data = padded_showers
  #   self.weight = padded_weights
  def padding(self, value=0.0):
    # Determine the maximum number of hits across all showers
    for showers in self.data:
        if len(showers) > self.max_nhits:
            self.max_nhits = len(showers)

    # Pad showers and weights
    padded_showers = []
    padded_weights = []
    for showers in self.data:
        pad_hits = self.max_nhits - len(showers)
        padded_shower = F.pad(input=showers, pad=(0, 0, 0, pad_hits), mode='constant', value=value)
        padded_showers.append(padded_shower)

    for weights in self.weight:
        pad_hits = self.max_nhits - len(weights)
        padded_weight = F.pad(input=weights, pad=(0, pad_hits), mode='constant', value=0.0)
        padded_weights.append(padded_weight)

    # Convert padded lists back to tensors
    self.data = [torch.tensor(shower, device=self.device) for shower in padded_showers]
    self.weight = [torch.tensor(weight, device=self.device) for weight in padded_weights]
  def clean(self,threshold):
    count = 0
    for i, showers in enumerate(self.data):
      has_large_values = (showers > threshold).any()
      if has_large_values:
        showers[showers > threshold] = threshold
        count += 1

# Build the combined DataLoader
class PreBatchedShowerDataset(Dataset):
    def __init__(self, datasets, batch_size,shuffle=True):
        """
        Wrap multiple datasets to return pre-batched samples from individual datasets.

        Args:
            datasets (list): A list of individual datasets (one per file).
            batch_size (int): Size of each batch.
        """
        self.datasets = datasets
        self.batch_size = batch_size

        # Create batch indices for each dataset
        self.batches = []
        for dataset_idx, dataset in enumerate(datasets):
            num_samples = len(dataset)
            num_batches = (num_samples+batch_size - 1) // batch_size  # Ceiling division

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx+batch_size, num_samples)
                self.batches.append((dataset_idx, start_idx, end_idx))
        #print(self.batches)
        if shuffle:
            random.shuffle(self.batches)
        #print(self.batches)
    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        """
        Return a batch from a single dataset.

        Args:
            idx (int): Index of the batch.

        Returns:
            tuple: (batch_shower_data, batch_incident_energies)
        """
        dataset_idx, start_idx, end_idx = self.batches[idx]
        dataset = self.datasets[dataset_idx]

        # Fetch samples from the current dataset
        batch = [dataset[i] for i in range(start_idx, end_idx)]

        # Separate showers and incident energies into two lists
        batch_shower_data, batch_incident_energies = zip(*batch)

        # Convert to tensors
        batch_shower_data = torch.stack(batch_shower_data)  # Shape: [batch_size, ...]
        batch_incident_energies = torch.tensor(batch_incident_energies)  # Shape: [batch_size]

        return batch_shower_data, batch_incident_energies
class rescale_conditional:
  '''Convert hit energies to range |01)
  '''
  def __init__(self):
            pass
  def __call__(self, conditional, emin, emax):
     e0 = conditional
     u0 = (e0-emin)/(emax-emin)
     return u0
    
class rescale_energies:
        '''Convert hit energies to range |01)
        '''
        def __init__(self):
            pass

        def __call__(self, features, condition, device='cpu'):
            Eprime = features[:,0]/(2*condition)
            alpha = 1e-06
            x = alpha+(1-(2*alpha))*Eprime
            rescaled_e = torch.log(x/(1-x))
            rescaled_e = torch.nan_to_num(rescaled_e)
            rescaled_e = torch.reshape(rescaled_e,(-1,))
            
            X_ = np.asarray(features[:,1]).reshape(-1, 1)
            Y_ = np.asarray(features[:,2]).reshape(-1, 1)
            
            transform_x = RobustScaler().fit(X_)
            transform_y = RobustScaler().fit(Y_)
            
            x_ = transform_x.transform(X_)
            x_ = torch.from_numpy( x_.flatten() )
            
            y_ = transform_y.transform(Y_)
            y_ = torch.from_numpy( y_.flatten() )

            x_ = normalize(features[:,1], dim=0)
            y_ = normalize(features[:,2], dim=0)

            z_ = features[:,3]
            # Stack tensors along the 'hits' dimension -1 
            stack_ = torch.stack((rescaled_e,x_,y_,z_), -1)
            
            self.features = stack_
            
            return self.features

class unscale_energies:
        '''Undo conversion of hit energies to range |01)
        '''
        def __init__(self):
            pass

        def __call__(self, features, condition):
            alpha = 1e-06
            eR = torch.exp(features[:,0])
            A = eR/(1+eR)
            rescaled_e = (A-alpha)*(2*condition)/(1-(2*alpha))

            x_ = features[:,1]
            y_ = features[:,2]
            z_ = features[:,3]
            
            # Stack tensors along the 'hits' dimension -1 
            stack_ = torch.stack((rescaled_e,x_,y_,z_), -1)
            self.features = stack_
            
            return self.features

# Physics-informed neural network prediction with multiple GPUs

# ====================================================================

# SETUP

# Pick the Unique ID from the training
UID = 'MASK-ONLY'
print('UID of training model:', UID)

# Specify the data to be used in the prediction
# training_fnames = ['overd', 'msrc', 'cellcluster100', 'mask']
training_fnames = ['mask']
n_input = len(training_fnames)
print(f'Training with data: {training_fnames} ({n_input} inputs)')

# Select the root of the project
project_root = "/jmain02/home/J2AD005/jck12/lxs35-jck12/modules/PINION/"

# Path to the simulation files to load
filepath = '/jmain02/home/J2AD005/jck12/lxs35-jck12/data/AI4EoR_244Mpc/'
# Path to where models are saved
modelpath = '/jmain02/home/J2AD005/jck12/lxs35-jck12/modules/PINION/louise_models/'
# Path to where the predicted subvolumes will be saved
export   = '/jmain02/home/J2AD005/jck12/lxs35-jck12/modules/PINION/louise_subvolumes/'

# Depending on your system, you may want to disable the memory mapping
memmap = True

# Size of subcube that each GPU will predict
cube_size = 25
# Total number of subcubes
ndomains = 1000
assert 250%cube_size == 0, f"250 isn't a multiple of {cube_size}!"
assert ndomains == (250//cube_size)**3

# ====================================================================

# TRAINING SETTINGS

from calendar import different_locale
from dataclasses import dataclass

# @dataclass adds generated special methods such as __init__() and __repr__() to user-defined classes
# So that, in the case of __init__(), self.kernel_size = kernel_size and etc.

@dataclass
class Config:
    """Keeps track of the current config."""
    kernel_size: int = 3
    n_pool: int = 2
    nb_train: int = 1000
    nb_test: int = 10
    subvolume_size: int = 19
    batch_size: int = 4600//5
    show: bool = False
    nb_epoch: int = 400
    plot_size: int = 50
    pinn_multiplication_factor: float = 1.0
    fcn_div_factor: int = 2
    n_fcn_layers: int = 5
    n_features: int = 64
    score: str = 'mse' # other choice: r2
    maxpool_size: int = 2
    maxpool_stride: int = 2
    n_input_channel: int = 3
    

    def __str__(self):
        return f"""Run configuration:
                --------------------------------
                kernel_size: {self.kernel_size}
                n_pool: {self.n_pool}
                nb_train: {self.nb_train}
                nb_test: {self.nb_test}
                subvolume_size: {self.subvolume_size}
                batch_size: {self.batch_size}
                show: {self.show}
                nb_epoch: {self.nb_epoch}
                plot_size: {self.plot_size}
                pinn_multiplication_factor: {self.pinn_multiplication_factor}
                fcn_div_factor: {self.fcn_div_factor}
                n_fcn_layers: {self.n_fcn_layers}
                n_features: {self.n_features}
                score: {self.score}
                maxpool_size: {self.maxpool_size}
                maxpool_stride: {self.maxpool_stride}
                n_input_channel: {self.n_input_channel}
                --------------------------------
                """

# Choice of configuration
myconfig = Config(kernel_size=3, n_pool=3, subvolume_size=11, n_features=64,
                  score='r2', maxpool_stride=1, nb_train=4000, nb_test=500,
                  batch_size=4600//5*4, fcn_div_factor=4, n_fcn_layers=5,
                  show=False, n_input_channel=n_input)
    
kernel_size = myconfig.kernel_size
n_pool = myconfig.n_pool
nb_train = myconfig.nb_train
nb_test = myconfig.nb_test
subvolume_size = myconfig.subvolume_size # MUST BE ODD
batch_size = myconfig.batch_size # must be a multiple of 46 if PINN is enabled.
show = myconfig.show # show the plots or not
nb_epoch = myconfig.nb_epoch
plot_size = myconfig.plot_size
pinn_multiplication_factor = myconfig.pinn_multiplication_factor
fcn_div_factor = myconfig.fcn_div_factor
n_fcn_layers = myconfig.n_fcn_layers
n_features = myconfig.n_features
score_type = myconfig.score
maxpool_size = myconfig.maxpool_size
maxpool_stride = myconfig.maxpool_stride
n_input_channel = myconfig.n_input_channel

# ===================================================================

# TRAINING LIBRARIES

import os
os.chdir(project_root)
print("working from: " + os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm.auto import tqdm
from astropy.cosmology import WMAP3
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
import collections
import astropy.units as u
import tools
import central_cnn as cnn

mpl.rcParams["figure.dpi"] = 100

# load the cosmology
cosmo = WMAP3

# ===================================================================

# LOAD DATA

# Get file names
# The ionisation rate is used in the physics-informed training
redshifts_str, files_irate = tools._get_files(filepath, 'irate')
# The ionisation fraction is used as the truth in the training
redshifts_str, files_xHII = tools._get_files(filepath, 'xHII')
# Train with the following input variables:
if 'overd' in training_fnames:
    redshifts_str, files_rho = tools._get_files(filepath, 'overd')
if 'msrc' in training_fnames:
    redshifts_str, files_msrc = tools._get_files(filepath, 'msrc')
if 'mask' in training_fnames:
    redshifts_str, files_mask = tools._get_files(filepath, 'mask')
if 'cellcluster100' in training_fnames:
    redshifts_str, files_cluster = tools._get_files(filepath, 'cellcluster100')

# Load the data
redshifts_arr, irates_arr = tools.load(files_irate, memmap) # 1/s
redshifts_arr, xHII_arr = tools.load(files_xHII, memmap) # unitless

if 'overd' in training_fnames:
    redshifts_arr, overdensity_arr = tools.load(files_rho, memmap) # unitless
    overdensity_arr *= (u.m/u.m)
    rhoc0 = cosmo.critical_density0 # g/cm3
    rho_arr = rhoc0 * (1 + overdensity_arr)
    rho_max = np.max(rho_arr)
    # rho_arr = tools.PBC(rho_arr, ts, xs, ys, zs)
    del overdensity_arr

if 'msrc' in training_fnames:
    redshifts_arr, msrc_arr = tools.load(files_msrc, memmap) # unitless
    msrc_arr *= (u.m/u.m)
    msrc_max = np.max(msrc_arr)
    # msrc_arr = tools.PBC(msrc_arr, ts, xs, ys, zs)

if 'mask' in training_fnames:
    redshifts_arr, mask_arr = tools.load(files_mask, memmap) # unitless
    mask_arr *= (u.m/u.m)
    mask_max = np.max(mask_arr)
    # mask_arr = tools.PBC(mask_arr, ts, xs, ys, zs)

if 'cellcluster100' in training_fnames:
    redshifts_arr, cluster_arr = tools.load(files_cluster, memmap) # unitless
    cluster_arr *= (u.m/u.m)
    cluster_max = np.max(cluster_arr)
    # cluster_arr = tools.PBC(cluster_arr, ts, xs, ys, zs)
    
redshifts_arr *= (u.m/u.m)

# Load the cosmology and convert redshift to time
print(redshifts_arr)
time_arr = np.asarray([cosmo.age(z).to(u.s).value for z in redshifts_arr], dtype=np.float32) * u.s
time_max = np.max(time_arr)
norm_time_arr = time_arr / time_max

# ====================================================================

# GET TRAINING DATA

# Each GPU makes predictions for a subcube
# Here we obtain the training data for that subcube
def get_training_data(exec_idx, verbose=False):

    # 1. Get coordinates for subcube
    # We want subcube size + allowance for predictions at the cube edge
    i,j,k = tools.coord_from_index(exec_idx, cube_size)
    xs = slice(i*cube_size - subvolume_size//2, i*cube_size + subvolume_size//2 + cube_size)
    ys = slice(j*cube_size - subvolume_size//2, j*cube_size + subvolume_size//2 + cube_size)
    zs = slice(k*cube_size - subvolume_size//2, k*cube_size + subvolume_size//2 + cube_size)
    ts = slice(0, 46)

    if verbose==True:
        print("Cube location:")
        print(f"x: {xs}")
        print(f"y: {ys}")
        print(f"z: {zs}")
        print("-------------")

    # 2. Obtain data for subcube
    rho_arr = tools.PBC(rho_arr, ts, xs, ys, zs)
    msrc_arr = tools.PBC(msrc_arr, ts, xs, ys, zs)
    mask_arr = tools.PBC(mask_arr, ts, xs, ys, zs)
    cluster_arr = tools.PBC(cluster_arr, ts, xs, ys, zs)

    # 3. Get the relative coordinates of the points in that subcube
    indices = np.indices((cube_size, cube_size, cube_size)).reshape(3, -1)
    print(f"indices: {indices[:,5]}")
    print(indices.shape)

    # 4. Initialise training sets
    training_set = np.zeros((46*cube_size**3, n_input, subvolume_size,
                             subvolume_size, subvolume_size),
                            dtype=np.float32)
    training_time = np.zeros((46*cube_size**3), dtype=np.float32)

    # 5. Create training batches
    for i in tqdm(range(indices.shape[1]), desc="Creating training batches"):
        
        # Populate training_set array with the training data
        # (Populated alphabetically, according to the training data name)
        train_int = 0
        if 'cellcluster100' in training_fnames:
            training_set[i*46:(i+1)*46, train_int] = cluster_arr[:, indices[0, i]:indices[0, i]+subvolume_size, indices[1, i]:indices[1, i]+subvolume_size, indices[2, i]:indices[2, i]+subvolume_size] / cluster_max
            train_int += 1
        if 'mask' in training_fnames:
            training_set[i*46:(i+1)*46, train_int] = mask_arr[:, indices[0, i]:indices[0, i]+subvolume_size, indices[1, i]:indices[1, i]+subvolume_size, indices[2, i]:indices[2, i]+subvolume_size] / mask_max
            train_int += 1
        if 'msrc' in training_fnames:
            training_set[i*46:(i+1)*46, train_int] = msrc_arr[:, indices[0, i]:indices[0, i]+subvolume_size, indices[1, i]:indices[1, i]+subvolume_size, indices[2, i]:indices[2, i]+subvolume_size] / msrc_max
            train_int += 1
        if 'overd' in training_fnames:
            training_set[i*46:(i+1)*46, train_int] = rho_arr[:, indices[0, i]:indices[0, i]+subvolume_size, indices[1, i]:indices[1, i]+subvolume_size, indices[2, i]:indices[2, i]+subvolume_size] / rho_max
            train_int += 1
        training_time[i*46:(i+1)*46] = norm_time_arr

    # Tell autograd not to record operations on this tensor
    training_set  = torch.from_numpy(training_set).requires_grad_(False)
    training_time = torch.from_numpy(training_time).requires_grad_(False)

    return training_set, training_time

# ====================================================================

# MULTI-GPU SETUP

# Number of GPUs
n_gpus = torch.cuda.device_count()

def run_inference(exec_idx, n_gpus):

    # Prepare for cuda
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = cnn.CentralCNNV2(n_input_channel, 1, n_pool, n_features,
                             kernel_size, subvolume_size, n_fcn_layers,
                             fcn_div_factor, maxpool_size,
                             maxpool_stride)
    model.eval()
    model.load_state_dict(torch.load(f"{modelpath}C-CNN-V2-model-{UID}.pt",
                                     map_location=device))
    model.to(device)

    # Get training data
    training_set, training_time = get_training_data(exec_idx, verbose=False)

    # Initialise array for predictions
    length = training_set.shape[0]//46
    prediction = np.zeros((46*cube_size**3), dtype=np.float32)
    prediction = torch.from_numpy(prediction).requires_grad_(False)

    # check memory status
    # print('check memory:')
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # Make predictions
    with torch.no_grad():
        for batch in tqdm(range(46), desc="iterating timings"):
            train_set  = training_set[batch*length:(batch+1)*length].to(device)
            train_time = training_time[batch*length:(batch+1)*length].view(-1, 1).to(device)
            prediction[batch*length:(batch+1)*length] = model(train_set, train_time)[:,0]

        print(prediction.shape)
        reshaped_prediction = np.zeros((46, cube_size, cube_size, cube_size), dtype=np.float32)
        reshaped_prediction = torch.from_numpy(reshaped_prediction).requires_grad_(False)

        for i in tqdm(range(indices.shape[1]), desc="Creating training batches"):
            reshaped_prediction[:, indices[0, i], indices[1, i], indices[2, i]] = prediction[i*46:(i+1)*46].cpu()

    print(reshaped_prediction.shape, torch.min(reshaped_prediction), torch.max(reshaped_prediction))
    reshaped_prediction = reshaped_prediction.detach().numpy()

    # Prepare the file for saving the result
    file = f'xHII_{UID}_cubesize{cube_size}_idx{exec_idx}.npy'

    filepath_result = export + file
    print("writing: ", filepath_result)

    # Save the result
    with open(filepath_result, 'wb') as nf:
        np.save(nf, reshaped_prediction)

        
mp.set_start_method('spawn')
multi_pool = Pool(processes=n_gpus)
predictions = multi_pool.map(run_inference, np.arange(1000))
multi_pool.close() 
multi_pool.join()

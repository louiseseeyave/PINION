# Pick the Unique ID from the training
# UID = 'TESTNP'
# UID = 'S7X9TJ'
UID = '244P3D'


# training on clustering info or not
use_cluster = False
if use_cluster:
    print('Training with cluster data')
else:
    print('Training without cluster data')


# Selects the root of the project
project_root = "/jmain02/home/J2AD005/jck12/lxs35-jck12/modules/PINION/"

# Choose the path to the simulation files to load, and the folders to save the model and the subvolumes
filepath = '/jmain02/home/J2AD005/jck12/lxs35-jck12/data/AI4EoR_244Mpc/'
# modelpath = '/jmain02/home/J2AD005/jck12/lxs35-jck12/modules/PINION/louise_models/'
modelpath = '/jmain02/home/J2AD005/jck12/lxs35-jck12/modules/PINION/louise_models/'
export   = '/jmain02/home/J2AD005/jck12/lxs35-jck12/modules/PINION/louise_subvolumes/'

# Depending on your system, you may want to disable the memory mapping.
memmap = True

# training settings dataclass
from calendar import different_locale
from dataclasses import dataclass

# @dataclass adds generated special methods such as __init__() and __repr__() to user-defined classes
# so that, in the case of __init__(), self.kernel_size = kernel_size and etc.
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
                --------------------------------
                """


# Choose the relevant configuration
myconfig = Config(kernel_size=3, n_pool=3, subvolume_size=7, n_features=64,
                  score='r2', maxpool_stride=1, nb_train=4000, nb_test=500,
                  batch_size=4600//5*4, fcn_div_factor=4, n_fcn_layers=5, show=False)

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


import os
os.chdir(project_root)
print("working from: " + os.getcwd())

import numpy as np
import torch
import torch.nn as nn
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

# prepare for cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use the information from the script
import sys
if len(sys.argv) > 3:
    print("Program does not understand extra arguments. Expected input:\npython pipeline_full.py {subcube_size} {execution_number}")
    sys.exit()
elif len(sys.argv) == 3:
    cube_size = int(sys.argv[1]) # size of the cube
    exec_idx = int(sys.argv[2]) # TASK_ID i.e. execution number
    assert 300%cube_size == 0, f"300 isn't a multiple of {cube_size}: 300%{cube_size}={300%cube_size}"
    assert exec_idx < (300//cube_size)**3
else:
    print("Program is missing arguments. Expected input:\npython pipeline_full.py {subcube_size} {execution_number}")
    sys.exit()


# Compute the subvolume
i,j,k = tools.coord_from_index(exec_idx, cube_size)
xs = slice(i*cube_size - subvolume_size//2, i*cube_size + subvolume_size//2 + cube_size)
ys = slice(j*cube_size - subvolume_size//2, j*cube_size + subvolume_size//2 + cube_size)
zs = slice(k*cube_size - subvolume_size//2, k*cube_size + subvolume_size//2 + cube_size)
ts = slice(0, 46)

print("Cube location:")
print(f"x: {xs}")
print(f"y: {ys}")
print(f"z: {zs}")
print("-------------")


# prepare the files
redshifts_str, files_irate  = tools._get_files(filepath, 'irate')
redshifts_str, files_xHII   = tools._get_files(filepath, 'xHII')
redshifts_str, files_rho   = tools._get_files(filepath, 'overd')
redshifts_str, files_msrc   = tools._get_files(filepath, 'msrc')
redshifts_str, files_mask  = tools._get_files(filepath, 'mask')
if use_cluster:
    redshifts_str, files_cluster  = tools._get_files(filepath, 'cluster') 

# load overdensity data
redshifts_arr, overdensity_arr = tools.load(files_rho, memmap) # unitless
overdensity_arr               *= (u.m/u.m)
rhoc0                          = cosmo.critical_density0 # g/cm3
rho_arr                        = rhoc0 * (1 + overdensity_arr)
rho_max                        = np.max(rho_arr)
rho_arr                        = tools.PBC(rho_arr, ts, xs, ys, zs)
# overdensity_arr = overdensity_arr[ts, xs, ys, zs]
del overdensity_arr

# load mass of sources (changed from nsrc to msrc)
redshifts_arr, msrc_arr        = tools.load(files_msrc, memmap) # unit mass
msrc_arr                      *= (u.m/u.m)
msrc_max                       = np.max(msrc_arr)
msrc_arr                       = tools.PBC(msrc_arr, ts, xs, ys, zs)
# nsrc_arr = nsrc_arr[ts, xs, ys, zs]

# load mask
redshifts_arr, mask_arr        = tools.load(files_mask, memmap) # unitless
mask_arr                      *= (u.m/u.m)
mask_max                       = np.max(mask_arr)
mask_arr                       = tools.PBC(mask_arr, ts, xs, ys, zs)

# load clustering
if use_cluster:
    redshifts_arr, mask_arr        = tools.load(files_cluster, memmap) # unitless
    cluster_arr                   *= (u.m/u.m)
    cluster_max                    = np.max(cluster_arr)
    cluster_arr                    = tools.PBC(cluster_arr, ts, xs, ys, zs)

redshifts_arr *= (u.m/u.m)


# load the cosmology and convert redshift to time
print(redshifts_arr)
time_arr = np.asarray([cosmo.age(z).to(u.s).value for z in redshifts_arr], dtype=np.float32) * u.s
time_max = np.max(time_arr)
norm_time_arr = time_arr / time_max

# produce the data to export
# training
if use_cluster:
    training_set = np.zeros((46*cube_size**3, 4, subvolume_size, subvolume_size, subvolume_size), dtype=np.float32)
else:
    training_set = np.zeros((46*cube_size**3, 3, subvolume_size, subvolume_size, subvolume_size), dtype=np.float32)
training_time = np.zeros((46*cube_size**3), dtype=np.float32)

# get the coordinates of the points that will be predicted
indices = np.indices((cube_size, cube_size, cube_size)).reshape(3, -1)
print(f"indices: {indices[:,5]}")
print(indices.shape)

for i in tqdm(range(indices.shape[1]), desc="Creating training batches"):
    training_set[i*46:(i+1)*46, 0] =   msrc_arr[:, indices[0, i]:indices[0, i]+subvolume_size, indices[1, i]:indices[1, i]+subvolume_size, indices[2, i]:indices[2, i]+subvolume_size] / msrc_max
    training_set[i*46:(i+1)*46, 1] =   rho_arr[:, indices[0, i]:indices[0, i]+subvolume_size, indices[1, i]:indices[1, i]+subvolume_size, indices[2, i]:indices[2, i]+subvolume_size] / rho_max
    training_set[i*46:(i+1)*46, 2] =   mask_arr[:, indices[0, i]:indices[0, i]+subvolume_size, indices[1, i]:indices[1, i]+subvolume_size, indices[2, i]:indices[2, i]+subvolume_size] / mask_max
    if use_cluster:
        training_set[i*46:(i+1)*46, 3] =   cluster_arr[:, indices[0, i]:indices[0, i]+subvolume_size, indices[1, i]:indices[1, i]+subvolume_size, indices[2, i]:indices[2, i]+subvolume_size] / cluster_max
    training_time[i*46:(i+1)*46]   = norm_time_arr

# tell autograd not to record operations on this tensor
# idk what this means
training_set  = torch.from_numpy(training_set).requires_grad_(False)
training_time = torch.from_numpy(training_time).requires_grad_(False)

# load the model
model = cnn.CentralCNNV2(3, 1, n_pool, n_features, kernel_size, subvolume_size,
                         n_fcn_layers, fcn_div_factor, maxpool_size, maxpool_stride).to(device)
model.eval()
model.load_state_dict(torch.load(f"{modelpath}C-CNN-V2-model-{UID}.pt", map_location=device))

# prediction
length = training_set.shape[0]//46

prediction = np.zeros((46*cube_size**3), dtype=np.float32)
prediction = torch.from_numpy(prediction).requires_grad_(False)

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

# prepare the file for saving the result.
file = f'xHII_{UID}_cubesize{cube_size}_idx{exec_idx}.npy'

filepath_result = export + file
print("writing: ", filepath_result)

#save the result
with open(filepath_result, 'wb') as nf:
    np.save(nf, reshaped_prediction)

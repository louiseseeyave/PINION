# GENERATE PROPAGATION MASK

# selects the root of the project
# generate the mask on cosma, not jade
# project_root = "/cosma8/data/dp004/dc-seey1/ml_reion/modules/PINION/"
project_root = "/jmain02/home/J2AD005/jck12/lxs35-jck12/modules/PINION/"

# files location
# filepath = "/cosma8/data/dp004/dc-seey1/ml_reion/data/AI4EoR_dataset/"
filepath = "/jmain02/home/J2AD005/jck12/lxs35-jck12/data/AI4EoR_244Mpc"
memmap = True # memory mapping might need to be disabled depending on the system
show = True

import os
os.chdir(project_root)
print("working from: " + os.getcwd())

import numpy as np
from tqdm.auto import tqdm
from astropy.cosmology import WMAP3
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
import glob
import collections
import astropy.units as u
import astropy.constants as cst
from generate_kernel import generate_kernel
import tools
import scipy.ndimage as ndi

mpl.rcParams["figure.dpi"] = 100

# load the cosmology
cosmo = WMAP3

# prepare the files
# this gives you a list of redshifts (str) and file names (str)
redshifts_str, files_nsrc  = tools._get_files(filepath, 'msrc')
# redshifts_str,  files_irate = tools._get_files(filepath, 'irate')

# load the data (mass of sources and ionisation rate)
redshifts_arr, nsrc_arr   = tools.load(files_nsrc, memmap) 
# redshifts_arr, irates_arr = tools.load(files_irate, memmap) # 1/s

# irates_arr /= u.s
# irates_max = np.max(irates_arr)
# not sure what this line is for:
# log_irates_arr = np.log10(irates_max.value)/np.log10(irates_arr.value) # zero when nan or inf

# 1) Compute the mean free path and convert it to px units.
def mfp(z):
    """
    Returns the mfp for this redshift in Mpc
    """
    print(f'the mfp at z={z} is {cst.c / cosmo.H(z) * 0.1 * np.power((1+z)/4, -2.55)}')
    return cst.c / cosmo.H(z) * 0.1 * np.power((1+z)/4, -2.55)

def mfp_in_px(z):
    """
    Returns the mfp in pixel units
    
    z: float32
        Redshift
    """
    # mpc_per_px = 2.381*u.Mpc # 500Mpc/300px/0.7
    mpc_per_px = 0.6832*u.Mpc # 244Mpc/250px/0.7
    print(f'the mean free path in pixel units is {mfp(z).to(u.Mpc)/mpc_per_px}')
    return mfp(z).to(u.Mpc)/mpc_per_px

# creates the result
results = np.zeros_like(nsrc_arr, dtype=np.float32)
max_nsrc = np.max(nsrc_arr)

files   = []
filesv2 = []
for i in tqdm(range(46)):
    # select the mass of sources
    cube = nsrc_arr[i,:]

    # generate the radius in px units
    radius = mfp_in_px(redshifts_arr[i])

    # print('radius', radius)

    # generate the kernel
    kernel = generate_kernel(radius, 3)

    # print('kernel', kernel)
            
    # convolve the mass of sources volume with the kernel
    experiment = ndi.convolve(cube, kernel)

    # print('experiment', experiment)

    results[i,:] = experiment

# save
sort_idx = np.argsort([float(z) for z in redshifts_str])[::-1]

sorted_redshifts_str = [redshifts_str[idx] for idx in sort_idx]

for i, z in enumerate(sorted_redshifts_str):

    newf = f"{filepath}irate_z{z}.npy" # save the mask as the ionisation rate
    # newf = f"{filepath}mask_z{z}.npy"
    print("{}: Writing: {}".format(i, newf))
    data = results[i,:,:,:]
    print('shape of data:', data.shape, data.dtype)
    # data.flatten(order='C').astype(np.float32)
    with open(newf, 'wb') as nf:
        np.save(nf, data)

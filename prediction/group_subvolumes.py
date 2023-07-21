# Selects Unique ID from training + additionnal information added to file name 
UID = 'FID3-CELLC25-8GPU'
ADD = ''

# Selects the root of the project
project_root = "/jmain02/home/J2AD005/jck12/lxs35-jck12/modules/PINION/"

# Some other options
subcube_path  = '/jmain02/home/J2AD005/jck12/lxs35-jck12/modules/PINION/louise_subvolumes/' # import folder
fullcube_path = '/jmain02/home/J2AD005/jck12/lxs35-jck12/modules/PINION/louise_fullvolume/' # export folder
memmap = True # Memory mapping should be disabled on some systems

import os
os.chdir(project_root)
print("working from: " + os.getcwd())


import numpy as np
from tqdm.auto import tqdm
import tools
import matplotlib.pyplot as plt


# Use the information from the script
import sys
if len(sys.argv) > 4:
    print("Program does not understand extra arguments. Expected input:\npython pipeline_full.py {subcube_size} {full_cube_size} [length_of_time (Optional, Default: 46)]")
    sys.exit()
elif len(sys.argv) == 3:
    cube_size = int(sys.argv[1])
    full_cube_size = int(sys.argv[2])
    ltime = 46
    assert full_cube_size%cube_size == 0, f"{full_cube_size} isn't a multiple of {cube_size}: {full_cube_size}%{cube_size}={300%cube_size}"
elif len(sys.argv) == 4:
    cube_size = int(sys.argv[1])
    full_cube_size = int(sys.argv[2])
    ltime = int(sys.argv[3])
    assert full_cube_size%cube_size == 0, f"{full_cube_size} isn't a multiple of {cube_size}: {full_cube_size}%{cube_size}={300%cube_size}"
else:
    print("Program is missing arguments. Expected input:\npython pipeline_full.py {subcube_size} {full_cube_size}")
    sys.exit()

# Load each subvolume and add it to the full cube
full_cube = np.zeros((ltime, full_cube_size, full_cube_size, full_cube_size))

for idx in tqdm(range((full_cube_size//cube_size)**3), desc="Joining batches"):
    i, j, k = tools.coord_from_index(idx, cube_size, full_cube_size)

    cube = np.load(f"{subcube_path}xHII{ADD}_{UID}_cubesize{cube_size}_idx{idx}.npy")
    full_cube[:, cube_size*i:cube_size*(i+1), cube_size*j:cube_size*(j+1), cube_size*k:cube_size*(k+1)] = cube

# manually select the redshifts
redshifts = ['12.048', '11.791', '11.546', '11.313', '11.090', '10.877',
             '10.673', '10.478', '10.290', '10.110', '9.938', '9.771',
             '9.611', '9.457', '9.308', '9.164', '9.026', '8.892', '8.762',
             '8.636', '8.515', '8.397', '8.283', '8.172', '8.064', '7.960',
             '7.859', '7.760', '7.664', '7.570', '7.480', '7.391', '7.305',
             '7.221', '7.139', '7.059', '6.981', '6.905', '6.830', '6.757',
             '6.686', '6.617', '6.549', '6.483', '6.418', '6.354']
assert len(redshifts) == ltime, f"Your redshifts are not matching your variable ltime: {len(redshifts)} should be equal to {ltime}."

# save the full cubes
for t in range(ltime):
    file = f'xHII{ADD}_{UID}_z{redshifts[t]}.npy'

    filepath_result = fullcube_path + file
    print("writing: ", filepath_result)

    with open(filepath_result, 'wb') as nf:
        np.save(nf, full_cube[t])

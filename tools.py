"""
This submodule contains multiple useful tools that I use in the other files
"""

import numpy as np
import glob
import collections
from tqdm.auto import tqdm

def PBC(arr, *slices):
    """
    Implement the periodic boundary conditions slicing for numpy arrays.
    This function assumes that the array is only extended once.
    
    arr: np.array
        The array to slice
    *slices: list of slices
        The slices in each axis 
    """
    shape = arr.shape
    new_slices, roll_shift = [], []
    ischanged = False
    for s, size in zip(slices, shape):
        if s.stop > size:
            shift = s.stop%size
            roll_shift.append(-shift)
            ischanged = True
        elif s.start < 0:
            shift = s.start
            roll_shift.append(-shift)
            ischanged = True
        else:
            # no shift is requires in this direction
            shift = 0
            roll_shift.append(-shift)

        # append the new slice
        sl = slice(s.start - shift, s.stop - shift, s.step)
        new_slices.append(sl)    
    
    # return the sliced array with PBC
    if ischanged:
        return np.roll(arr, roll_shift, axis=range(len(shape))).__getitem__(tuple(new_slices))
    else:
        return arr.__getitem__(slices)
    

def _get_files(path, datatype, extension='npy'):

    """
    Returns a dictionary of all files to load with their associated redshift.
    PARAMETERS
    ----------
    datatype: str
        The name of the data to load, e.g. irate, xHII, ...
    extension: str (Default: 'npz')
        The file extension; must be numpy compatible.
    """

    # get list of files
    files = path+f"{datatype}_*.{extension}"
    all_files = glob.glob(path+f"{datatype}_*.{extension}")
    print(f"Got {len(all_files)} files for path: {files}")

    # extract filename
    filenames = [f.split('/')[-1] for f in all_files]

    # get the redshift information
    redshifts_str = [f.split("_z",1)[1].replace(f'.{extension}', '') for f in filenames]

    # Remove duplicates in redshift list
    redshifts_str_single = []
    [redshifts_str_single.append(item) for item in redshifts_str if item not in redshifts_str_single]

    # we store the filename with respect to their redshift and type
    results = collections.OrderedDict()
    for z in redshifts_str_single:
        results[z] = glob.glob(path+"{}_z{}*{}".format(datatype, z, extension))[0]
    
    # save the result
    return (redshifts_str_single, results)



def load(files, memmap, full_cube_size=250, num_of_redshifts=46):

    """
    Loads the data.
    PARAMETERS
    ----------
    memmap: bool
        Memory mapping, False loads the whole cube on the memory, True
        loads the "pointer" to the memory, allowing to reduce the memory
        usage at the cost of higher load time. This option might be useful
        to disable in some systems.
    full_cube_size: int (Default: 300)
        The grid size of the full volume
    num_of_redshifts: int (Default: 46)
        The number of snapshots for the simulation
    """

    redshifts, redshifts_str = [], []
    data = np.zeros((num_of_redshifts, full_cube_size, full_cube_size,
                     full_cube_size), dtype=np.float32)
    for i, (z, filename) in tqdm(enumerate(files.items()), desc="Reading data...",
                                 total=num_of_redshifts):
        # load the data.
        if memmap:
            data[i,:,:,:] = np.load(filename, mmap_mode='r').reshape((250,250,250))
            # data[i,:,:,:] = np.load(filename, mmap_mode='r').reshape((300,300,300))
        else:
            data[i,:,:,:] = np.load(filename, mmap_mode=None)

        # save only the relevant data
        redshifts.append(float(z))
        redshifts_str.append(z)

    # sort the data by redshift
    sort_idx = np.argsort(redshifts)[::-1]
    redshifts = [redshifts[idx] for idx in sort_idx]
    return (redshifts, data[sort_idx,:,:,:])

def index_from_coord(i, j, k, side_size, full_size=250):
        """
        For the subdivision of the datacube, returns the index of the subcube for given coordinates
        PARAMETERS
        ----------
        i,j,k: int, int, int
            Position in the cube for which we want the index
        side_size: int
            The size of one side of the subvolume
        full_size: int (Default 300)
            The side size of the full volume
        """
        shape = [full_size, full_size, full_size]
        nb_x, nb_y, nb_z = [int(s/side_size) for s in shape] # number of sub cubes per dimensions
        return i*nb_z*nb_y + j*nb_z + k
    
def coord_from_index(index, side_size, full_size=250):
    """
    For the subdivision of the datacube, returns the coordinates of the subcube for given index
    The conventional order for subcubes indices is:
    index = 0
    for i:
        for j:
            for k:
                print("(i,j,j)=({},{},{}) => {}".format(i,j,k,index)) 
                index += 1
    
    PARAMETERS
    ----------
    index: int
        The subvolume index from which to retrieve the coordinates
    side_size: int
        The size of one side of the subvolume
    full_size: int (Default 300)
        The side size of the full volume
    """
    shape = [full_size, full_size, full_size]
    nb_x, nb_y, nb_z = [int(s/side_size) for s in shape] # number of sub cubes per dimensions
    i = index//(nb_z*nb_y)
    j = (index-i*(nb_z*nb_y)) // nb_y
    k = index-i*(nb_z*nb_y)-j*nb_z
    return (i,j,k)

# just some tests
if __name__ == "__main__":
    i,j,k = coord_from_index(1, 10)
    print(f"{10*i}, {10*j}, {10*k}")

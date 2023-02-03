import numpy as np
import itertools

def generate_kernel(radius, dim=3):
    """
    Generate the correct kernel for the task
    radius: float
        The radius of the circle
    dim: int (Default: 3)
        number of dimensions of the kernel.
    """
    # generate a zero kernel of good size.
    min_size = int(np.ceil(2*radius))
    if min_size % 2 == 0: min_size += 1

    kernel = np.zeros([min_size for _ in range(dim)])

    circle_centre = np.array([min_size//2 for _ in range(dim)]) + .5

    # iterate over each pixels of the plot and do the following:
    # 1) compute the distance between the circle centre and all the corners of the voxels
    # 2) one can compute a ratio between the circle distance and distance to min/max
    # 3) from this, we have an output between 0 and 1 that approximates the ratio of volumes (very approximate)
    for coords in itertools.product(*[range(min_size) for _ in range(dim)]): 
        coords = np.array(coords)

        # 1) dist to corners 
        voxel_centre = coords + .5

        # 2) ratio
        max_dist = np.sqrt(dim) # voxels have 1px sides.
        val = -(np.sqrt(np.sum((voxel_centre - circle_centre)**2))-radius)/max_dist + 1/2

        # 3) apply 
        kernel[tuple(coords)] = val

    # clip to only have values between 0 and 1     
    return np.clip(kernel, 0, 1)

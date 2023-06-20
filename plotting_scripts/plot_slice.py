import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# --------------------------------------------------------------------

# plot input variables

# qs = ['irate', 'mask', 'msrc', 'overd', 'xHII', 'cluster10',
#       'cluster20', 'cluster50', 'cluster75', 'cluster100',
#       'cluster125', 'cluster150', 'coarsecluster10', 'coarsecluster20',
#       'coarsecluster50']

# def plot(q):

#     fname = f'{q}_z7.859.npy'
#     data = np.load(fname)

#     box_dims = 244/0.7 # Length of the volume along each direction in Mpc.
#     dx, dy = box_dims/data.shape[1], box_dims/data.shape[2]
#     y, x = np.mgrid[slice(dy/2,box_dims,dy),
#                     slice(dx/2,box_dims,dx)]

#     fig, ax = plt.subplots(1,1)
#     plt.title(f'{q}')
#     if q=='msrc':
#         plot = ax.pcolormesh(x, y, data[0],
#                              norm=matplotlib.colors.Normalize(vmin=0, vmax=1000))
#     else:
#         plot = ax.pcolormesh(x, y, data[0])
        
#     fig.colorbar(plot, ax=ax)

#     plt.savefig(f'plots/test_{q}', dpi=300, bbox_inches='tight')


# for q in qs:
#     plot(q)


# --------------------------------------------------------------------

# plot output variables

t_path = '/jmain02/home/J2AD005/jck12/lxs35-jck12/data/AI4EoR_244Mpc'
p_path = '/jmain02/home/J2AD005/jck12/lxs35-jck12/modules/PINION/louise_fullvolume'

def compare_output(UID, z, truth_path=t_path, pred_path=p_path):

    truth_fname = f'{t_path}/xHII_z{z}.npy'
    truth_data = np.load(truth_fname)
    print(f'shape of truth: {truth_data.shape}')

    pred_fname = f'{p_path}/xHII_{UID}_z{z}.npy'
    pred_data = np.load(pred_fname)
    print(f'shape of pred: {pred_data.shape}')

    box_dims = 244/0.7 # Length of the volume along each direction in Mpc.
    # dx, dy = box_dims/truth_data.shape[1], box_dims/truth_data.shape[2]
    # y, x = np.mgrid[slice(dy/2,box_dims,dy),
    #                 slice(dx/2,box_dims,dx)]

    fig = plt.figure(figsize=(11,3.7))
    fig.suptitle(f"UID: {UID}")

    dx, dy = box_dims/pred_data.shape[1], box_dims/pred_data.shape[2]
    y, x = np.mgrid[slice(dy/2,box_dims,dy),
                    slice(dx/2,box_dims,dx)]
    axs0 = fig.add_subplot(131)
    pred = axs0.pcolormesh(x, y, pred_data[0])
    axs0.set_title("Predicted")
    axs0_divider = make_axes_locatable(axs0)
    cax0 = axs0_divider.append_axes("right", size="4%", pad=0)
    cb0 = fig.colorbar(pred, cax=cax0)

    dx, dy = box_dims/truth_data.shape[1], box_dims/truth_data.shape[2]
    y, x = np.mgrid[slice(dy/2,box_dims,dy),
                    slice(dx/2,box_dims,dx)]
    axs1 = fig.add_subplot(132)
    truth = axs1.pcolormesh(x, y, truth_data[0])
    axs1.set_title("Truth")
    axs1_divider = make_axes_locatable(axs1)
    cax1 = axs1_divider.append_axes("right", size="4%", pad=0)
    cb1 = fig.colorbar(truth, cax=cax1)

    axs2 = fig.add_subplot(133)
    residual = axs2.pcolormesh(x, y, truth_data[0]-pred_data[0],
                                 cmap='bwr')
    axs2.set_title("Residual")
    axs2_divider = make_axes_locatable(axs2)
    cax2 = axs2_divider.append_axes("right", size="4%", pad=0)
    cb2 = fig.colorbar(residual, cax=cax2)

    fig.tight_layout()
    
    plt.savefig(f'plots/compare_output/compare_output_{UID}_{z}.png',
                dpi=300, bbox_inches='tight')
    print(f'saved plot at plots/compare_output/compare_output_{UID}_{z}.png')
    plt.close()

# zs = ['7.059']
zs = ['6.830', '7.059', '7.480']
# UIDs = ['C100NM']
# UIDs = ['244FID', 'CEL025', 'CEL050', 'CEL075', 'CEL100']
UIDs = ['C100NM', 'MASKxx', 'CE100x']

for UID in UIDs:
    for z in zs:
        compare_output(UID, z)

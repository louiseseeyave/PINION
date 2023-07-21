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

    print('generating slice plot...')

    # load true ionisation fraction
    truth_fname = f'{t_path}/xHII_z{z}.npy'
    truth_data = np.load(truth_fname)
    print(f'shape of truth: {truth_data.shape}')
    # load predicted ionisation fraction
    pred_fname = f'{p_path}/xHII_{UID}_z{z}.npy'
    pred_data = np.load(pred_fname)
    print(f'shape of pred: {pred_data.shape}')

    box_dims = 244/0.7

    # begin plot
    fig = plt.figure(figsize=(11,3.7))
    fig.suptitle(f"UID: {UID}, z={z}")

    # get grid cell coordinates for predictions
    dx, dy = box_dims/pred_data.shape[1], box_dims/pred_data.shape[2]
    y, x = np.mgrid[slice(dy/2,box_dims,dy),
                    slice(dx/2,box_dims,dx)]

    # index of slice to be plotted
    ind = 100
    
    # plot predicted ionisation fraction for a slice
    axs0 = fig.add_subplot(131)
    pred = axs0.pcolormesh(x, y, pred_data[ind],
                           norm=matplotlib.colors.Normalize(vmin=0, vmax=1))
    axs0.grid(color='whitesmoke')
    axs0.set_title("Predicted")
    axs0_divider = make_axes_locatable(axs0)
    cax0 = axs0_divider.append_axes("right", size="4%", pad=0)
    cb0 = fig.colorbar(pred, cax=cax0)

    # get grid cell coordinates for truth
    dx, dy = box_dims/truth_data.shape[1], box_dims/truth_data.shape[2]
    y, x = np.mgrid[slice(dy/2,box_dims,dy),
                    slice(dx/2,box_dims,dx)]

    # plot true ionisation fraction for a slice
    axs1 = fig.add_subplot(132)
    truth = axs1.pcolormesh(x, y, truth_data[ind],
                            norm=matplotlib.colors.Normalize(vmin=0, vmax=1))
    axs1.grid(color='whitesmoke')
    axs1.set_title("Truth")
    axs1_divider = make_axes_locatable(axs1)
    cax1 = axs1_divider.append_axes("right", size="4%", pad=0)
    cb1 = fig.colorbar(truth, cax=cax1)

    # plot residuals for the slice
    axs2 = fig.add_subplot(133)
    diff = pred_data[ind]-truth_data[ind]
    residual = axs2.pcolormesh(x, y, diff, cmap='bwr',
                               norm=matplotlib.colors.Normalize(vmin=-1, vmax=1))
    axs2.grid(color='whitesmoke')
    axs2.set_title("Residual")
    axs2_divider = make_axes_locatable(axs2)
    cax2 = axs2_divider.append_axes("right", size="4%", pad=0)
    cb2 = fig.colorbar(residual, cax=cax2)

    fig.tight_layout()

    # save plot
    plt.savefig(f'../plots/compare_output/compare_output_{UID}_{z}.png',
                dpi=300, bbox_inches='tight')
    print(f'saved plot at ../plots/compare_output/compare_output_{UID}_{z}.png')
    plt.close()

    print('generating histogram of residuals...')

    # plot histogram of residuals

    fig, ax = plt.subplots(1,1)
    plt.title(f'UID:{UID}')
    
    z1='6.354'
    truth_fname = f'{t_path}/xHII_z{z1}.npy'
    truth_data1 = np.load(truth_fname)
    pred_fname = f'{p_path}/xHII_{UID}_z{z1}.npy'
    pred_data1 = np.load(pred_fname)
    residuals1 = pred_data1 - truth_data1
    residuals1 = residuals1.flatten()

    z2='8.064'
    truth_fname = f'{t_path}/xHII_z{z2}.npy'
    truth_data2 = np.load(truth_fname)
    pred_fname = f'{p_path}/xHII_{UID}_z{z2}.npy'
    pred_data2 = np.load(pred_fname)
    residuals2 = pred_data2 - truth_data2
    residuals2 = residuals2.flatten()
    
    z3='10.110'
    truth_fname = f'{t_path}/xHII_z{z3}.npy'
    truth_data3 = np.load(truth_fname)
    pred_fname = f'{p_path}/xHII_{UID}_z{z3}.npy'
    pred_data3 = np.load(pred_fname)
    residuals3 = pred_data3 - truth_data3
    residuals3 = residuals3.flatten()

    ax.grid(color='whitesmoke')
    ax.set_axisbelow('True')
    ax.hist([residuals1, residuals2, residuals3], bins=20, #edgecolor='white'
             label=[f'z={z1}', f'z={z2}', f'z={z3}'])
    ax.legend()
    plt.xlabel('residual (pred-truth)')
    plt.ylabel('N')
    plt.yscale('log')
    plt.savefig(f'../plots/compare_output/residuals_hist_{UID}.png',
                dpi=300, bbox_inches='tight')
    print(f'saved plot at ../plots/compare_output/residuals_hist_{UID}.png')
    plt.close()

    
# --------------------------------------------------------------------
    
    
# zs = ['7.059']
zs = ['6.830', '7.059', '7.480']
# UIDs = ['C100NM']
# UIDs = ['244FID', 'CEL025', 'CEL050', 'CEL075', 'CEL100']
UIDs = ['FID3', '244FID', 'MASK-ONLY-8GPU']
# UIDs = ['MASKxx']

for UID in UIDs:
    for z in zs:
        compare_output(UID, z)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tools
from scipy import stats

# Load data

# filepath = '/jmain02/home/J2AD005/jck12/lxs35-jck12/data/AI4EoR_244Mpc/'
# redshifts_str, files_irate  = tools._get_files(filepath, 'irate')
# redshifts_str, files_mask  = tools._get_files(filepath, 'mask')
# redshifts_str, files_cluster25  = tools._get_files(filepath, 'cellcluster25')
# redshifts_str, files_cluster50  = tools._get_files(filepath, 'cellcluster50')
# redshifts_str, files_cluster75  = tools._get_files(filepath, 'cellcluster75')
# redshifts_str, files_cluster100  = tools._get_files(filepath, 'cellcluster100')

# redshifts_arr, irates_arr      = tools.load(files_irate, memmap) # 1/s
# redshifts_arr, mask_arr        = tools.load(files_mask, memmap) # unitless
# redshifts_arr, cluster25_arr     = tools.load(files_cluster25, memmap) # unitless
# redshifts_arr, cluster50_arr     = tools.load(files_cluster50, memmap) # unitless
# redshifts_arr, cluster75_arr     = tools.load(files_cluster75, memmap) # unitless
# redshifts_arr, cluster100_arr     = tools.load(files_cluster100, memmap) # unitless

filepath = '/jmain02/home/J2AD005/jck12/lxs35-jck12/data/AI4EoR_244Mpc'

def plot_corr(z, xdata, ydata, filepath=filepath):
    
    fname = f'{filepath}/{xdata}_z{z}.npy'
    xx = np.load(fname)
    xx = xx.flatten()

    fname = f'{filepath}/{ydata}_z{z}.npy'
    yy = np.load(fname)
    yy = yy.flatten()

    norm = matplotlib.colors.LogNorm()

    fig, ax = plt.subplots()
    plt.title(f'z={z}')
    plot = ax.hexbin(xx, yy, mincnt=1, linewidth=0., norm=norm)
    cb = fig.colorbar(plot)
    corr = stats.pearsonr(xx, yy)[0]
    ax.text(0.1, 0.9, f'corr: {corr:.2f}', transform=ax.transAxes)
    ax.set_xlabel(xdata)
    ax.set_ylabel(ydata)
    ax.grid(color='whitesmoke')
    ax.set_axisbelow('True')
    plt.savefig(f'../plots/corr/{xdata}_{ydata}_{z}.png', dpi=300,
                bbox_inches='tight')
    plt.close()


zs = ['6.830', '7.059', '7.480']
qs = ['cellcluster25', 'cellcluster50', 'cellcluster75', 'cellcluster100',
      'irate', 'mask', 'msrc', 'overd']

pairs = []
for q in qs:
    pairs.append(('xHII', q))

for (x,y) in pairs:
    for z in zs:
        plot_corr(z, x, y, filepath=filepath)

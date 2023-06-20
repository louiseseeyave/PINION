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

def plot_corr(z, filename, filepath=filepath):
    
    fname = f'{filepath}/irate_z{z}.npy'
    irate = np.load(fname)
    irate = irate.flatten()

    fname = f'{filepath}/mask_z{z}.npy'
    mask = np.load(fname)
    mask = mask.flatten()

    fname = f'{filepath}/{filename}_z{z}.npy'
    data = np.load(fname)
    data = data.flatten()

    norm = matplotlib.colors.LogNorm()

    fig, ax = plt.subplots()
    plot = ax.hexbin(data, mask, mincnt=1, linewidth=0., norm=norm)
    cb = fig.colorbar(plot)
    corr = stats.pearsonr(data, mask)[0]
    ax.text(0.1, 0.9, f'corr: {corr:.2f}', transform=ax.transAxes)
    ax.set_xlabel(filename)
    ax.set_ylabel('mask')
    plt.savefig(f'plots/compare_output/corr_{filename}_mask_{z}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    plot = ax.hexbin(data, irate, mincnt=1, linewidth=0., norm=norm)
    cb = fig.colorbar(plot)
    corr = stats.pearsonr(data, irate)[0]
    ax.set_xlabel(filename)
    ax.set_ylabel('irate')
    ax.text(0.1, 0.9, f'corr: {corr:.2f}', transform=ax.transAxes)
    plt.savefig(f'plots/compare_output/corr_{filename}_irate_{z}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    plot = ax.hexbin(mask, irate, mincnt=1, linewidth=0., norm=norm)
    cb = fig.colorbar(plot)
    corr = stats.pearsonr(mask, irate)[0]
    ax.set_xlabel('mask')
    ax.set_ylabel('irate')
    ax.text(0.1, 0.9, f'corr: {corr:.2f}', transform=ax.transAxes)
    plt.savefig(f'plots/compare_output/corr_mask_irate_{z}.png',
                dpi=300, bbox_inches='tight')
    plt.close()


zs = ['6.830', '7.059', '7.480']
filenames = ['cellcluster25', 'cellcluster50', 'cellcluster75', 'cellcluster100']

for filename in filenames:
    for z in zs:
        plot_corr(z, filename, filepath=filepath)

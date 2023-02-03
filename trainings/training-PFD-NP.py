# Physics and Full Data & No Physics training

# This script produces the training with the physics and full data & no physics settings.
# The first setting trains is the full package training.
# All data snapshots are used for the data training and the ODE constraints is enabled.
# For the second setting, the ODE constraint is disable.
# For this, set the boolean "enable_pinn" to False.

#choose to enable pinn or not
enable_pinn = False

# Selects the root of the project
project_root = "/cosma8/data/dp004/dc-seey1/ml_reion/modules/PINION/"

# files location
filepath = '/cosma8/data/dp004/dc-seey1/ml_reion/data/AI4EoR_dataset/' # macos
savepath = '/cosma8/data/dp004/dc-seey1/ml_reion/modules/PINION/louise_models/'
memmap = True

# training settings
from calendar import different_locale
from dataclasses import dataclass

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
    score: str = 'mse' # other choice: r2
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

# Choice of configuration
myconfig = Config(kernel_size=3, n_pool=3, subvolume_size=7, n_features=64, score='r2', maxpool_stride=1, nb_train=4000, nb_test=500, batch_size=4600//5*4, fcn_div_factor=4, n_fcn_layers=5, show=True)

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

# --------------------------------------------------------------------

import os
os.chdir(project_root)
print("working from: " + os.getcwd())

# just testing something
#import sys
#print('sys.path:', sys.path)
#sys.path.remove('/cosma/home/dp004/dc-seey1/.local/lib/python3.9/site-packages')

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
import tools21cm as t2c
import tools

mpl.rcParams["figure.dpi"] = 100

# load the cosmology
cosmo = WMAP3

# prepare for cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------------------------------------------------

# Loading the data

# prepare the files
redshifts_str, files_irate  = tools._get_files(filepath, 'irate')
redshifts_str, files_xHII   = tools._get_files(filepath, 'xHII')
redshifts_str, files_rho   = tools._get_files(filepath, 'overd')
redshifts_str, files_nsrc   = tools._get_files(filepath, 'msrc')
redshifts_str, files_mask  = tools._get_files(filepath, 'mask') 


# load the data
redshifts_arr, irates_arr      = tools.load(files_irate, memmap) # 1/s
redshifts_arr, xHII_arr        = tools.load(files_xHII, memmap) # unitless
redshifts_arr, overdensity_arr = tools.load(files_rho, memmap) # unitless
redshifts_arr, nsrc_arr        = tools.load(files_nsrc, memmap) # unitless
redshifts_arr, mask_arr        = tools.load(files_mask, memmap) # unitless

# apply units
irates_arr /= u.s
xHII_arr *= (u.m/u.m)
overdensity_arr *= (u.m/u.m)
redshifts_arr *= (u.m/u.m)
nsrc_arr *= (u.m/u.m)
mask_arr *= (u.m/u.m)

# Now we can convert all the data into their correct form.
Om0 = cosmo.Om0
alpha = 2.59e-13 * (u.cm**3/u.s) # 2.59e-13 cm^3/s
rhoc0 = cosmo.critical_density0 # g/cm3
mu = 1.32 * (u.m/u.m) # unitless
mp = 1.67e-24 * u.g #1.67e-27 * u.kg # kg

# we want to get the n_H information, which is the number density of hydrogen.
#rho = rhoc0.value * (overdensity + 1)
mu_He = 0.074 * (u.m/u.m)
nh_bar = (1-mu_He) * (cosmo.Ob0 * rhoc0)/(mu * mp)
nh_bar.to(1/u.cm**3)
nh_arr = nh_bar * (1 + overdensity_arr)

# get the density from overdensity
rho_arr = rhoc0 * (1 + overdensity_arr)

del overdensity_arr

# load the cosmology and convert redshift to time
#time = [cosmo.lookback_time(np.inf).value - cosmo.lookback_time(z).value for z in redshifts]
time_arr = np.asarray([cosmo.age(z).to(u.s).value for z in redshifts_arr], dtype=np.float32) * u.s
time_max = np.max(time_arr)
norm_time_arr = time_arr / time_max

#irates_max = np.max(irates_arr)
nsrc_max = np.max(nsrc_arr)
rho_max = np.max(rho_arr)
mask_max = np.max(mask_arr)

irates_max = irates_arr.max()
#log_irates_arr = np.log10(irates_max.value)/np.log10(irates_arr.value) # zero when nan or inf

# --------------------------------------------------------------------

# Preparing the training set

# choose the random subvolumes we'll consider.
rand_train_i, rand_train_j, rand_train_k = np.random.randint(0, 300-subvolume_size, size=(3, nb_train))
rand_test_i, rand_test_j, rand_test_k = np.random.randint(0, 300-subvolume_size, size=(3, nb_test))

# training
training_set = np.zeros((46*nb_train, 3, subvolume_size, subvolume_size, subvolume_size), dtype=np.float32)
training_truth = np.zeros((46*nb_train, 1), dtype=np.float32)
training_time = np.zeros((46*nb_train), dtype=np.float32)
training_nh = np.zeros((46*nb_train), dtype=np.float32)
training_irates = np.zeros((46*nb_train), dtype=np.float32)

for batch in tqdm(range(nb_train), desc="Creating training batches"):
    training_set[batch*46:(batch+1)*46, 0]   =   nsrc_arr[:, rand_train_i[batch]:rand_train_i[batch]+subvolume_size, rand_train_j[batch]:rand_train_j[batch]+subvolume_size, rand_train_k[batch]:rand_train_k[batch]+subvolume_size] / nsrc_max
    training_set[batch*46:(batch+1)*46, 1]   =    rho_arr[:, rand_train_i[batch]:rand_train_i[batch]+subvolume_size, rand_train_j[batch]:rand_train_j[batch]+subvolume_size, rand_train_k[batch]:rand_train_k[batch]+subvolume_size] / rho_max
    training_set[batch*46:(batch+1)*46, 2]   =   mask_arr[:, rand_train_i[batch]:rand_train_i[batch]+subvolume_size, rand_train_j[batch]:rand_train_j[batch]+subvolume_size, rand_train_k[batch]:rand_train_k[batch]+subvolume_size] / mask_max
    training_truth[batch*46:(batch+1)*46,0]  =   xHII_arr[:, (2*rand_train_i[batch]+subvolume_size)//2, (2*rand_train_j[batch]+subvolume_size)//2, (2*rand_train_k[batch]+subvolume_size)//2]
    training_nh[batch*46:(batch+1)*46]       =     nh_arr[:, (2*rand_train_i[batch]+subvolume_size)//2, (2*rand_train_j[batch]+subvolume_size)//2, (2*rand_train_k[batch]+subvolume_size)//2]
    training_irates[batch*46:(batch+1)*46]   = irates_arr[:, (2*rand_train_i[batch]+subvolume_size)//2, (2*rand_train_j[batch]+subvolume_size)//2, (2*rand_train_k[batch]+subvolume_size)//2]
    training_time[batch*46:(batch+1)*46]     = norm_time_arr


# testing
testing_set = np.zeros((46*nb_train, 3, subvolume_size, subvolume_size, subvolume_size), dtype=np.float32)
testing_truth = np.zeros((46*nb_train, 1), dtype=np.float32)
testing_time = np.zeros((46*nb_train), dtype=np.float32)
testing_nh = np.zeros((46*nb_train), dtype=np.float32)
testing_irates = np.zeros((46*nb_train), dtype=np.float32)

for batch in tqdm(range(nb_test), desc="Creating training batches"):
    testing_set[batch*46:(batch+1)*46, 0]   =   nsrc_arr[:, rand_test_i[batch]:rand_test_i[batch]+subvolume_size, rand_test_j[batch]:rand_test_j[batch]+subvolume_size, rand_test_k[batch]:rand_test_k[batch]+subvolume_size] / nsrc_max
    testing_set[batch*46:(batch+1)*46, 1]   =    rho_arr[:, rand_test_i[batch]:rand_test_i[batch]+subvolume_size, rand_test_j[batch]:rand_test_j[batch]+subvolume_size, rand_test_k[batch]:rand_test_k[batch]+subvolume_size] / rho_max
    testing_set[batch*46:(batch+1)*46, 2]   =   mask_arr[:, rand_test_i[batch]:rand_test_i[batch]+subvolume_size, rand_test_j[batch]:rand_test_j[batch]+subvolume_size, rand_test_k[batch]:rand_test_k[batch]+subvolume_size] / mask_max
    testing_truth[batch*46:(batch+1)*46, 0] =   xHII_arr[:, (2*rand_test_i[batch]+subvolume_size)//2, (2*rand_test_j[batch]+subvolume_size)//2, (2*rand_test_k[batch]+subvolume_size)//2]
    testing_nh[batch*46:(batch+1)*46]       =     nh_arr[:, (2*rand_test_i[batch]+subvolume_size)//2, (2*rand_test_j[batch]+subvolume_size)//2, (2*rand_test_k[batch]+subvolume_size)//2]
    testing_irates[batch*46:(batch+1)*46]   = irates_arr[:, (2*rand_test_i[batch]+subvolume_size)//2, (2*rand_test_j[batch]+subvolume_size)//2, (2*rand_test_k[batch]+subvolume_size)//2]
    testing_time[batch*46:(batch+1)*46]     = norm_time_arr


# plotting
plot_set = np.zeros((plot_size**2, 3, subvolume_size, subvolume_size, subvolume_size), dtype=np.float32)
plot_truth = np.zeros((plot_size**2, 1), dtype=np.float32)

# pick the centre of the cube
centre = [s//2 for s in xHII_arr.shape][1:]
time_plot = 32

for j in tqdm(range(centre[1] - plot_size//2, centre[1] + plot_size//2, 1), desc="Iterating px for plot"):
    for k in range(centre[2] - plot_size//2, centre[2] + plot_size//2, 1):
        batch = (j-centre[1]+plot_size//2)*plot_size + k-centre[2]+plot_size//2
        plot_set[batch, 0] = nsrc_arr[time_plot, centre[0]:centre[0]+subvolume_size, j:j+subvolume_size, k:k+subvolume_size] / nsrc_max
        plot_set[batch, 1] = rho_arr[time_plot, centre[0]:centre[0]+subvolume_size, j:j+subvolume_size, k:k+subvolume_size] / rho_max
        plot_set[batch, 2] = mask_arr[time_plot, centre[0]:centre[0]+subvolume_size, j:j+subvolume_size, k:k+subvolume_size] / mask_max
        plot_truth[batch, 0] = xHII_arr[time_plot, (2*centre[0]+subvolume_size)//2, (2*j+subvolume_size)//2, (2*k+subvolume_size)//2]

plot_time = np.repeat(norm_time_arr[time_plot], plot_size**2)

# convert everything to pytorch tensor
training_set    = torch.from_numpy(training_set).requires_grad_(True)
training_truth  = torch.from_numpy(training_truth)
training_irates = torch.from_numpy(training_irates)
training_nh     = torch.from_numpy(training_nh)
training_time   = torch.from_numpy(training_time)

testing_set     = torch.from_numpy(testing_set)
testing_truth   = torch.from_numpy(testing_truth)
testing_irates  = torch.from_numpy(testing_irates)
testing_nh      = torch.from_numpy(testing_nh)
testing_time    = torch.from_numpy(testing_time)

plot_set        = torch.from_numpy(plot_set)
plot_truth      = torch.from_numpy(plot_truth)
plot_time       = torch.from_numpy(plot_time)

# --------------------------------------------------------------------

def plot_comparative(file, epoch, predicted, truth, r2, show=False):
    fig = plt.figure(figsize=(10, 3))
    plt.suptitle(f"Epoch: {epoch}")
    plt.subplot(131)
    pos = plt.imshow(predicted, origin='lower', norm=mpl.colors.Normalize(vmin=0, vmax=1), interpolation='none')
    plt.title("Predicted")
    fig.colorbar(pos)
    
    plt.subplot(132)
    pos = plt.imshow(truth, origin='lower', norm=mpl.colors.Normalize(vmin=0, vmax=1), interpolation='none')
    plt.title("Truth")
    fig.colorbar(pos)
    
    plt.subplot(133)
    pos = plt.imshow(predicted - truth, origin='lower', cmap='bwr', norm=mpl.colors.Normalize(vmin=-1, vmax=1), interpolation='none')
    plt.title("Diff: $R^2$={:.2e}".format(1-r2))
    fig.colorbar(pos)
    
    
    plt.tight_layout()
    if show:
        plt.show()
    else:
        fig.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
    plt.close()

def plot_input_data(nsrc, rho, irates):
    fig = plt.figure(figsize=(10, 3))
    plt.subplot(131)
    pos = plt.imshow(nsrc, origin='lower', interpolation='none', cmap='bw')
    plt.title("nsrc")
    fig.colorbar(pos)
    
    plt.subplot(132)
    pos = plt.imshow(rho, origin='lower', interpolation='none', cmap='PiYG')
    plt.title("rho")
    fig.colorbar(pos)
    
    plt.subplot(133)
    pos = plt.imshow(irates, origin='lower', interpolation='none', cmap='Oranges')
    plt.title("approx irates")
    fig.colorbar(pos)
              
    plt.show()
    
def plot_statistics(file, epoch, prediction, truth, show=False):    
    # compute box size
    shape = prediction.shape
    fullboxsize = 500/0.743 # in Mpc
    resolution = fullboxsize/300 # in Mpc
    boxsize = shape[0] * resolution

    # compute power spectrum
    ps1t, ks1t = t2c.power_spectrum_1d(truth,      kbins=15, box_dims=boxsize)
    ps1p, ks1p = t2c.power_spectrum_1d(prediction, kbins=15, box_dims=boxsize)
    
    # compute bubble size
    r_mfp1t, dn_mfp1t = t2c.mfp(truth>0.5, boxsize=boxsize, iterations=1000000)
    r_mfp1p, dn_mfp1p = t2c.mfp(prediction>0.5, boxsize=boxsize, iterations=1000000)

    # plot
    fig = plt.figure(figsize = (10,5))
    plt.suptitle(f"Morphology study of 2D slice for epoch {epoch}")
        
    nan_mask_t = np.isnan(ks1t*ps1t) == False
    ft = ps1t[nan_mask_t]*ks1t[nan_mask_t]**3/2/np.pi**2
    xt = ks1t[nan_mask_t]
    tott = np.trapz(ft, xt)
    avt = np.trapz(xt * ft / tott, xt)

    print(f"Truth av: {avt}")
    print(xt, ft)
    
    nan_mask_p = np.isnan(ks1p*ps1p) == False
    fp = ps1p[nan_mask_p]*ks1t[nan_mask_p]**3/2/np.pi**2
    xp = ks1p[nan_mask_p]
    totp = np.trapz(fp, xp)
    avp = np.trapz(xp * fp / totp, xp)

    plt.subplot(121)
    plt.title('Spherically averaged power spectrum')
    plt.plot(xt, ft, '-',  color='C0', label="Truth: xHII={:.3f}".format(np.mean(truth)))
    plt.plot(xp, fp, '--', color='C0', label="Predi: xHII={:.3f}".format(np.mean(prediction)))
    ylim = [min(min(ft), min(fp)), max(max(ft), max(fp))]
    plt.plot([avt, avt], ylim, '-', color='black', label="Mean truth")
    plt.plot([avp, avp], ylim, '--', color='black', label="Mean predi")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('k (Mpc$^{-1}$)')
    plt.ylabel('P(k) k$^{3}$/$(2\pi^2)$')
    plt.grid(True, linewidth=.1)
    plt.legend()
    
    nan_mask_t = np.isnan(r_mfp1t*dn_mfp1t) == False
    ft = dn_mfp1t[nan_mask_t]
    xt = r_mfp1t[nan_mask_t]
    
    tott = np.trapz(ft, xt)
    avt = np.trapz(xt * ft / tott, xt)
    bft = np.trapz(ft[xt<=avt], xt[xt<=avt])
    aft = np.trapz(ft[xt>=avt], xt[xt>=avt])
    
    nan_mask_p = np.isnan(r_mfp1p*dn_mfp1p) == False
    fp = dn_mfp1p[nan_mask_p]
    xp = r_mfp1p[nan_mask_p]
    
    totp = np.trapz(fp, xp)
    avp = np.trapz(xp * fp / totp, xp)
    bfp = np.trapz(fp[xp<=avp], xp[xp<=avp])
    afp = np.trapz(fp[xp>=avp], xp[xp>=avp])
    
    plt.subplot(122)
    plt.title('Bubble size: Mean free path method')
    plt.plot(r_mfp1t, dn_mfp1t, '-',  label="Truth: xHII={:.3f}".format(np.mean(truth)), color='C0')
    plt.plot(r_mfp1p, dn_mfp1p, '--', label="Predi: xHII={:.3f}".format(np.mean(prediction)), color='C0')
    ylim = [min(min(ft), min(fp)), max(max(ft), max(fp))]
    plt.plot([avt, avt], ylim, '-', color='black', label="Mean truth")
    plt.plot([avp, avp], ylim, '--', color='black', label="Mean predi")
    plt.xscale('log')
    plt.xlabel('$R$ (Mpc)')
    plt.ylabel('$R\mathrm{d}P/\mathrm{d}R$')
    plt.grid(True, linewidth=.1)
    plt.legend()

    plt.tight_layout()
    if show:
        plt.show()
    else:
        fig.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
    plt.close()



def plot_loss(file, total_loss, data_loss, pinn_loss, validation_loss, learning_rate, show=False):
    fig = plt.figure()
    ax = plt.subplot()
    ax.plot(total_loss, label="Total")
    ax.plot(data_loss, label="Data")
    ax.plot(pinn_loss, label="Physics")
    ax.plot(validation_loss, label="Validation")
    plt.legend()
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Losses")
    
    ax2 = ax.twinx()
    ax2.plot(learning_rate, '--', c='gray')
    ax2.set_ylabel("Learning rate")
    ax2.set_yscale('log')
    
    plt.tight_layout()
    if show:
        plt.show()
    else:
        fig.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
    plt.close()

def generate_random_string(length = 6):
    import random, string
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k = length))

def get_lr(optimizer):
    return [ group['lr'] for group in optimizer.param_groups ][0]

# --------------------------------------------------------------------

# Loading the model

from importlib import reload
import central_cnn as cnn
import r2score as r2s

reload(cnn)
reload(mpl)
reload(r2s)


# 1) Define the models
model = cnn.CentralCNNV2(3, 1, n_pool, n_features, kernel_size, subvolume_size, n_fcn_layers, fcn_div_factor, maxpool_size, maxpool_stride).to(device)
if enable_pinn==True:
    UID = 'PFD' + generate_random_string(6) # id to save stuff and avoid overriding plots.
else:
    UID = 'NP' + generate_random_string(6) # id to save stuff and avoid overriding plots.
print(f"Unique ID for this run: {UID}")

from torchinfo import summary
print(summary(model, [(920, 3, subvolume_size, subvolume_size, subvolume_size), (920, 1)]))
print(model)

# --------------------------------------------------------------------

# Training the network

# 2) loss and opt
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2) # it will optimize both gamma and x at the same time
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min') # default patience: 10 epochs
if score_type == 'mse':
    criterion = nn.MSELoss() #wmse.WeightedMSELoss().to(device)
elif score_type == 'r2':
    criterion = r2s.InvertedR2Score()
else:
    assert False, f"The score type that you chose doesn't exist: {score_type}"

    
    
total_losses, data_losses, validation_losses, pinn_losses = [], [], [], []
learning_rates = []

best_loss = 1e15

files_loss, files_slice, files_morph = [], [], []

# we do a first evaluation round to "normalize" the losses
model.eval()
with torch.no_grad():
    train_pinn_loss, train_data_loss,  = 0,0
    for batch in tqdm(range((46*nb_train)//batch_size), position=1, leave=False, disable=False):
        # free the optimizer (otherwise it will accumulate)
        optimizer.zero_grad()
        
        # predict the result
        input_train_x = training_set[batch_size*batch:batch_size*(batch+1)].to(device)
        input_train_t = training_time[batch_size*batch:batch_size*(batch+1)].view(-1, 1).to(device)
        truth_train   = training_truth[batch_size*batch:batch_size*(batch+1),0].to(device)
        truth_irates  = training_irates[batch_size*batch:batch_size*(batch+1)].to(device)
        truth_nh      = training_nh[batch_size*batch:batch_size*(batch+1)].to(device)

        prediction = model(input_train_x, input_train_t).view(-1)#.view(truth_train.shape)
        
        # get the loss
        train_data_loss_batch = criterion(prediction, truth_train)
        
        if enable_pinn:
            # physics with finite difference
            time_batch = input_train_t.reshape(-1, 46)*time_max.value
            dt = time_batch[:,2:] - time_batch[:,:-2]
            xh = prediction.reshape(-1, 46)
            xi = xh[:,:-2]
            xip1 = xh[:,1:-1]
            xip2 = xh[:,2:]
            gamma = truth_irates.view(-1, 46)
            gammai = gamma[:,:-2]
            gammaip1 = gamma[:,1:-1]
            gammaip2 = gamma[:,2:]
            
            xh_truth = truth_train.reshape(-1,46)
            xip2_truth = xh_truth[:,2:]
            
            nh = truth_nh.view(-1, 46)
            nhi   = nh[:,:-2]
            nhip1 = nh[:,1:-1]
            nhip2 = nh[:,2:]
            
            Di = alpha.value*nhi
            Dip1 = alpha.value*nhip1
            Dip2 = alpha.value*nhip2
            
            k1 = (1-xi)*gammai - Di*xi**2
            k2 = (1-xi-dt/2*k1)*gammaip1 - Dip1*(xi+dt/2*k1)**2
            k3 = (1-xi-dt/2*k2)*gammaip1 - Dip1*(xi+dt/2*k2)**2
            k4 = (1-xi-dt*k3  )*gammaip2 - Dip2*(xi+dt*k3  )**2
        
            xip2_pred = xi + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
            xdiff = (torch.clip(torch.abs(xip2_pred), 0, 1) - xi)/dt
            physics = xdiff - ( (1-xi)*gammai - Di*xi**2 )
            #physics = xip2_truth - torch.clip(torch.abs(xip2_pred),0,1)
        
            #compute the loss of the physics, i.e. compute the mean square error
            # Note: We force the first value to be 1.
            PINNLoss = torch.mean(physics**2) * pinn_multiplication_factor
            #PINNLoss = criterion(xi, xip2) * pinn_multiplication_factor
        
        # training loss
        loss_batch = train_data_loss_batch #+ PINNLoss
                
        # save the losses
        train_data_loss += train_data_loss_batch.item()
        if enable_pinn:
            train_pinn_loss += PINNLoss.item()
        else:
            train_pinn_loss += 0#PINNLoss.item()
        
    # save the initial losses
    init_train_data_loss = train_data_loss
    init_train_pinn_loss = train_pinn_loss
    
    
    test_data_loss = 0

    for batch in tqdm(range((46*nb_test)//batch_size), position=1, leave=False, disable=False):
        # predict the result
        input_test_x = testing_set[batch_size*batch:batch_size*(batch+1)].to(device)
        input_test_t = testing_time[batch_size*batch:batch_size*(batch+1)].view(-1, 1).to(device)
        truth_test   = testing_truth[batch_size*batch:batch_size*(batch+1),0].to(device)
        truth_irates  = testing_irates[batch_size*batch:batch_size*(batch+1)].to(device)
        truth_nh      = testing_nh[batch_size*batch:batch_size*(batch+1)].to(device)

        prediction = model(input_test_x, input_test_t).view(-1)#.view(truth_train.shape)

        # get the loss
        # Note: We force the first value to be one
        test_data_loss_batch = criterion(prediction, truth_test) 

        # save the losses
        test_data_loss += train_data_loss_batch.item()

    init_test_data_loss = test_data_loss
    

for epoch in tqdm(range(nb_epoch), desc="Iterating epoch", position=0):
    model.train()
    train_total_loss, train_pinn_loss, train_data_loss = 0,0,0
    
    for batch in tqdm(range((46*nb_train)//batch_size), position=1, leave=False, disable=False):
        # free the optimizer (otherwise it will accumulate)
        optimizer.zero_grad()
        
        # predict the result
        input_train_x = training_set[batch_size*batch:batch_size*(batch+1)].to(device)
        input_train_t = training_time[batch_size*batch:batch_size*(batch+1)].view(-1, 1).to(device)
        truth_train   = training_truth[batch_size*batch:batch_size*(batch+1),0].to(device)
        truth_irates  = training_irates[batch_size*batch:batch_size*(batch+1)].to(device)
        truth_nh      = training_nh[batch_size*batch:batch_size*(batch+1)].to(device)

        prediction = model(input_train_x, input_train_t).view(-1)#.view(truth_train.shape)
        
        # get the loss
        # Note: We force the first value to be one
        train_data_loss_batch = criterion(prediction, truth_train) / init_train_data_loss
        
        # physics with finite difference
        if enable_pinn:
            time_batch = input_train_t.reshape(-1, 46)*time_max.value
            dt = time_batch[:,2:] - time_batch[:,:-2]
            xh = prediction.reshape(-1, 46)
            xi = xh[:,:-2]
            xip1 = xh[:,1:-1]
            xip2 = xh[:,2:]
            gamma = truth_irates.view(-1, 46)
            gammai = gamma[:,:-2]
            gammaip1 = gamma[:,1:-1]
            gammaip2 = gamma[:,2:]
                        
            nh = truth_nh.view(-1, 46)
            nhi   = nh[:,:-2]
            nhip1 = nh[:,1:-1]
            nhip2 = nh[:,2:]
            
            Di = alpha.value*nhi
            Dip1 = alpha.value*nhip1
            Dip2 = alpha.value*nhip2
            
            k1 = (1-xi)*gammai - Di*xi**2
            k2 = (1-xi-dt/2*k1)*gammaip1 - Dip1*(xi+dt/2*k1)**2
            k3 = (1-xi-dt/2*k2)*gammaip1 - Dip1*(xi+dt/2*k2)**2
            k4 = (1-xi-dt*k3  )*gammaip2 - Dip2*(xi+dt*k3  )**2
        
            xip2_pred = xi + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
            
            xdiff = (torch.clip(torch.abs(xip2_pred), 0, 1) - xi)/dt
            physics = xdiff - ( (1-xi)*gammai - Di*xi**2 )
            

            #compute the loss of the physics, i.e. compute the mean square error
            # Note: We force the first value to be 1.
            PINNLoss = torch.mean(physics**2) / init_train_pinn_loss * pinn_multiplication_factor
        

        
        # training loss
        loss_batch = train_data_loss_batch
        if enable_pinn:
            loss_batch += PINNLoss
        
        # backward
        loss_batch.backward()
        optimizer.step()
        
        # save the losses
        train_total_loss += loss_batch.item()
        train_data_loss += train_data_loss_batch.item()
        if enable_pinn:
            train_pinn_loss += PINNLoss.item()
        else:
            train_pinn_loss += 0
        
    total_losses.append(train_total_loss)
    data_losses.append(train_data_loss)
    pinn_losses.append(train_pinn_loss)     
    
    
    # adapt learning rate.
    learning_rates.append(get_lr(optimizer))
    scheduler.step(train_total_loss)
    
    model.eval()
    with torch.no_grad():
        test_data_loss = 0

        for batch in tqdm(range((46*nb_test)//batch_size), position=1, leave=False, disable=False):
            # predict the result
            input_test_x = testing_set[batch_size*batch:batch_size*(batch+1)].to(device)
            input_test_t = testing_time[batch_size*batch:batch_size*(batch+1)].view(-1, 1).to(device)
            truth_test   = testing_truth[batch_size*batch:batch_size*(batch+1),0].to(device)
            truth_irates  = testing_irates[batch_size*batch:batch_size*(batch+1)].to(device)
            truth_nh      = testing_nh[batch_size*batch:batch_size*(batch+1)].to(device)

            prediction = model(input_test_x, input_test_t).view(-1)

            # get the loss
            # Note: We force the first value to be one
            test_data_loss_batch = criterion(prediction, truth_test) / init_test_data_loss

            # save the losses
            test_data_loss += train_data_loss_batch.item()
        
        validation_losses.append(test_data_loss)

        # save the model if the loss is smaller
        if best_loss > test_data_loss:
            print(f"Best new loss: {test_data_loss}")
            best_loss = test_data_loss
            torch.save(model.state_dict(), f"{savepath}C-CNN-V2-model-{UID}.pt")

        if (epoch+1) % 10 == 0:
            plot_input_x = plot_set.to(device)
            plot_input_t  = plot_time.to(device).view(-1, 1)

            prediction = model(plot_input_x, plot_input_t).view(-1)

            prediction = prediction.reshape((plot_size, plot_size)).cpu()
            truth = plot_truth.reshape((plot_size, plot_size)).cpu()

            r2score = criterion(prediction, truth)

            prediction = prediction.numpy()
            truth = truth.numpy()

            shape = plot_input_x.shape
            centre = [s//2 for s in shape]
            plot_nsrc   = plot_input_x[:,0, centre[2], centre[3], centre[4]].reshape((plot_size, plot_size)).cpu().numpy()
            plot_rho    = plot_input_x[:,1, centre[2], centre[3], centre[4]].reshape((plot_size, plot_size)).cpu().numpy()
            plot_irates = plot_input_x[:,2, centre[2], centre[3], centre[4]].reshape((plot_size, plot_size)).cpu().numpy()

            
            file = "plots/C-CNN-V2-{}_slice_{:08d}.png".format(UID, epoch+1)
            files_slice.append(file)
            plot_comparative(file, epoch+1, prediction, truth, r2score, show=show)
            file = "plots/C-CNN-V2-{}_loss_{:08d}.png".format(UID, epoch+1)
            files_loss.append(file)
            plot_loss(file, total_losses, data_losses, pinn_losses, validation_losses, learning_rates, show=show)
            file = "plots/C-CNN-V2-{}_morphology_{:08d}.png".format(UID, epoch+1)
            files_morph.append(file)
            

print(UID)

# save: the losses and the best epoch 
np.savez(f'loss/C-CNN-V2-{UID}.npz', train_total=total_losses, train_data=data_losses, train_pinn=pinn_losses, losses_validation=validation_losses, learning_rates=learning_rates)

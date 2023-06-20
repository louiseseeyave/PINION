import numpy as np
import matplotlib.pyplot as plt
import tools21cm as t2c

def compare_training_loss(UIDs, string='', cmap=None):

    """
    Plots the loss as a function of epoch.
    """

    fig = plt.figure()
    ax = plt.subplot()

    if cmap!=None:
        cmap = plt.get_cmap(cmap)

    for i, UID in enumerate(UIDs):
        losses = np.load(f'../loss/C-CNN-V2-{UID}.npz')
        if cmap==None:
            ax.plot(losses['train_total'], label=UID)
        else:
            col = cmap(i/(len(UIDs)-1))
            ax.plot(losses['train_total'], label=UID, c=col)
        # ax.plot(data_loss, label="Data")
        # ax.plot(pinn_loss, label="Physics")
        # ax.plot(validation_loss, label="Validation")

    plt.legend()
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Losses")
    
    
    fig.savefig(f'plots/compare_training_loss{string}', bbox_inches='tight',  dpi=300)

    
def compare_statistics(UIDs, z, string='', cmap=None):

    # def plot_statistics(file, epoch, prediction, truth, show=False):    

    # compute box size
    boxsize = 244/0.7 # in Mpc
    
    # plot
    fig = plt.figure(figsize = (10,5))
    plt.suptitle("Morphology study of 3D box")

    if cmap!=None:
        cmap = plt.get_cmap(cmap)

    # ----------------------------------------------------------------
    # plot power spectrum
    plt.subplot(121)
    plt.title('Spherically averaged power spectrum')

    # predictions
    for UID in UIDs:
        fname = f'../louise_fullvolume/xHII_{UID}_z{z}.npy'
        d = np.load(fname)
        # d_slice = d[d.shape[0]//2]
        # compute power spectrum
        ps1p, ks1p = t2c.power_spectrum_1d(d, kbins=15,
                                           box_dims=boxsize)
        nan_mask_p = np.isnan(ks1p*ps1p) == False
        fp = ps1p[nan_mask_p]*ks1p[nan_mask_p]**3/2/np.pi**2
        xp = ks1p[nan_mask_p]
        totp = np.trapz(fp, xp)
        avp = np.trapz(xp * fp / totp, xp)
        if cmap==None:
            plt.plot(xp, fp, '--',
                     label=f"{UID}: xHII={np.mean(d):.3f}")
        else:
            col = cmap(i/len(UIDs))
            plt.plot(xp, fp, '--', c=col,
                     label=f"{UID}: xHII={np.mean(d):.3f}")

    # truth
    fname = f'../../../data/AI4EoR_244Mpc/xHII_z{z}.npy'
    d = np.load(fname)
    ps1t, ks1t = t2c.power_spectrum_1d(d, kbins=15, box_dims=boxsize)
    nan_mask_t = np.isnan(ks1t*ps1t) == False
    ft = ps1t[nan_mask_t]*ks1t[nan_mask_t]**3/2/np.pi**2
    xt = ks1t[nan_mask_t]
    tott = np.trapz(ft, xt)
    avt = np.trapz(xt * ft / tott, xt)
    # print(f"Truth av: {avt}")
    # print(xt, ft)
    if cmap==None:
        plt.plot(xt, ft, '-', 
             label="Truth: xHII={:.3f}".format(np.mean(d)))
    else:
        plt.plot(xt, ft, '-', c=cmap(1),
                 label="Truth: xHII={:.3f}".format(np.mean(d)))

    # plot the mean power spectrum - add another time
    # ylim = [min(min(ft), min(fp)), max(max(ft), max(fp))]
    # plt.plot([avt, avt], ylim, '-', color='black', label="Mean truth")
    # plt.plot([avp, avp], ylim, '--', color='black', label="Mean predi")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('k (Mpc$^{-1}$)')
    plt.ylabel('P(k) k$^{3}$/$(2\pi^2)$')
    plt.grid(True, linewidth=.1)
    plt.legend()
    
    # the following code is to obtain ylim
    # nan_mask_t = np.isnan(r_mfp1t*dn_mfp1t) == False
    # ft = dn_mfp1t[nan_mask_t]
    # xt = r_mfp1t[nan_mask_t]
    
    # tott = np.trapz(ft, xt)
    # avt = np.trapz(xt * ft / tott, xt)
    # bft = np.trapz(ft[xt<=avt], xt[xt<=avt])
    # aft = np.trapz(ft[xt>=avt], xt[xt>=avt])
    
    # nan_mask_p = np.isnan(r_mfp1p*dn_mfp1p) == False
    # fp = dn_mfp1p[nan_mask_p]
    # xp = r_mfp1p[nan_mask_p]
    
    # totp = np.trapz(fp, xp)
    # avp = np.trapz(xp * fp / totp, xp)
    # bfp = np.trapz(fp[xp<=avp], xp[xp<=avp])
    # afp = np.trapz(fp[xp>=avp], xp[xp>=avp])

    # ----------------------------------------------------------------
    # plot bubble size

    plt.subplot(122)
    plt.title('Bubble size: Mean free path method')

    # predictions
    for UID in UIDs:
        fname = f'../louise_fullvolume/xHII_{UID}_z{z}.npy'
        d = np.load(fname)
        # d_slice = d[d.shape[0]//2]
        # compute bubble size
        r_mfp1p, dn_mfp1p = t2c.mfp(d, boxsize=boxsize,
                                    iterations=1000000)
        if cmap==None:
            plt.plot(r_mfp1p, dn_mfp1p, '--',
                     label=f"{UID}: xHII={np.mean(d):.3f}")
        else:
            col = cmap(i/len(UIDs))
            plt.plot(r_mfp1p, dn_mfp1p, '--', c=col,
                     label=f"{UID}: xHII={np.mean(d):.3f}")

    # truth
    fname = f'../../../data/AI4EoR_244Mpc/xHII_z{z}.npy'
    d = np.load(fname)
    r_mfp1t, dn_mfp1t = t2c.mfp(d, boxsize=boxsize, iterations=1000000)
    if cmap==None:
        plt.plot(r_mfp1t, dn_mfp1t, '-',
                 label="Truth: xHII={:.3f}".format(np.mean(d)))
    else:
        plt.plot(r_mfp1t, dn_mfp1t, '-', c=cmap(1),
                 label="Truth: xHII={:.3f}".format(np.mean(d)))
    # ylim = [min(min(ft), min(fp)), max(max(ft), max(fp))]
    # plt.plot([avt, avt], ylim, '-', color='black', label="Mean truth")
    # plt.plot([avp, avp], ylim, '--', color='black', label="Mean predi")
    plt.xscale('log')
    plt.xlabel('$R$ (Mpc)')
    plt.ylabel('$R\mathrm{d}P/\mathrm{d}R$')
    plt.grid(True, linewidth=.1)
    plt.legend()

    plt.tight_layout()
    fsave = f'plots/compare_statistics_{z}{string}.png'
    fig.savefig(fsave, bbox_inches='tight', pad_inches=0.1, dpi=300,
                facecolor="white")
    plt.close()


# ====================================================================
    
string = '_160623'

UIDs = ['244FID', 'CEL100', 'C100NM', 'MASKxx', 'CE100x']
compare_training_loss(UIDs, string)

# UIDs = ['244FID','HCL150', 'CEL025', 'CEL050', 'CEL075', 'CEL100']
zs = ['6.830', '7.059', '7.480']
for z in zs:
    compare_statistics(UIDs, z, string)

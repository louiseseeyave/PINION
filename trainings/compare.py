import numpy as np
import matplotlib.pyplot as plt

def compare_training_loss(UIDs):

    """
    Plots the loss as a function of epoch.
    """

    fig = plt.figure()
    ax = plt.subplot()

    for UID in UIDs:
        losses = np.load(f'../loss/C-CNN-V2-{UID}.npz')
        ax.plot(losses['train_total'], label=UID)
        # ax.plot(data_loss, label="Data")
        # ax.plot(pinn_loss, label="Physics")
        # ax.plot(validation_loss, label="Validation")

    plt.legend()
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Losses")
    
    
    fig.savefig('plots/compare_training_loss', bbox_inches='tight',  dpi=300)


UIDs = ['244P3D', '244C50', '244C20', '244C10']

compare_training_loss(UIDs)

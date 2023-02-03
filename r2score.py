import torch
import torch.nn as nn
import numpy as np

class R2Score(nn.Module):
    """
    R2 score following wikipedia's definition: https://en.wikipedia.org/wiki/Coefficient_of_determination 
    Note that it's "inverted" because 0 is the best result and 1 the worst.
    The true definition is R^2 = 1 - SSres/SStot
    But we use: R^2 = SSres/SStot
    
    USES NUMPY ARRAYS
    """
    def __init__(self):
        super().__init__()
        self.__name__ = 'R2Score'

    def forward(self, prediction: np.array, targets: np.array):
        mean = 1/len(targets) * np.sum(targets)

        # residual sum of squares
        SSres = np.sum((targets - prediction)**2)

        # total sum of squares
        SStot = np.sum((targets - mean)**2)

        return SSres / SStot

class InvertedR2Score(nn.Module):
    """
    R2 score following wikipedia's definition: https://en.wikipedia.org/wiki/Coefficient_of_determination 
    Note that it's "inverted" because 0 is the best result and 1 the worst.
    The true definition is R^2 = 1 - SSres/SStot
    But we use: R^2 = SSres/SStot
    
    USES PYTORCH TENSORS
    """
    def __init__(self):
        super().__init__()
        self.__name__ = 'InvertedR2Score'

    def forward(self, prediction: torch.Tensor, targets: torch.Tensor):
        mean = 1/len(targets) * torch.sum(targets)

        # residual sum of squares
        SSres = torch.sum((targets - prediction)**2)

        # total sum of squares
        SStot = torch.sum((targets - mean)**2)

        return SSres / SStot

# some tests
if __name__ == "__main__":
    r2 = InvertedR2Score()
    array_size = 10
    x_t = torch.linspace(0, 10, array_size)
    y_t = x_t**2 + x_t + 5

    x_p = torch.linspace(0, 10, array_size)
    y_p = x_t**2 + x_t + 5 + 5*(torch.rand((array_size))-.5)
    
    print(r2(y_t, y_p))

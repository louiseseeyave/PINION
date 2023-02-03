import torch
import torch.nn as nn

class CentralCNNV2(nn.Module):
    def __init__(self, n_input_channel, n_output_val, n_pool, n_features, kernel_size, data_input_size, n_fcn_layers, fcn_div_factor=2, maxpool_size=2, maxpool_stride=2):
        """
        Initializes the CentralCNN.
        The structure of the network is the following:
        - First start with a subvolume of nsrc, rho and the propagating mask
        - convolve it until some point
        - flatten the result
        - Append time in the flatten result
        - Use this as the input for a FCN
        - Output x at the end
        n_input_channel: int
            Number of input channels
        n_output_val: int
            Number of output values
        n_pool: int
            Number of Conv+pool blocks (including the initial one)
        n_features: int
            Number of features in the mean time
        kernel_size: int
            The kernel size (for now, only int is available, but maybe later (int, int, int) will be as well)
        data_input_size: int
            The size of the input data (so the size of side only)
        n_fcn_layers: int
            Number of fully connected layers (including the initial one)
        fcn_div_factor: int (Default 2)
            Provides the division faction between input and output of each fcn layer.
        maxpool_size: int (Default 2)
            Size of maxpooling kernel. 
        maxpool_stride: int (Default 2)
            Stride of maxpooling.
        """
        # initialize all the parameters
        super().__init__()
        self.n_in = n_input_channel
        self.n_out = n_output_val
        self.n_pool = n_pool
        self.kernel_size = kernel_size
        self.nf = n_features
        self.data_size = data_input_size
        self.n_fcn = n_fcn_layers
        self.fcn_div_factor = fcn_div_factor
        self.maxpool_size = maxpool_size
        self.maxpool_stride = maxpool_stride

        # start with a layer that goes from n_inputs to n_features
        self.first_conv = nn.Sequential(
            nn.Conv3d(n_input_channel, n_features, kernel_size),
            nn.PReLU(),
            nn.BatchNorm3d(n_features),
            nn.Dropout3d(p=0.1)
        )

        # then continue to apply convolution to reduce the size of the data
        most_conv = []
        nf = n_features
        if n_pool > 1:
            for i in range(n_pool-1):
                most_conv.append(
                    nn.Sequential(
                        nn.Conv3d(nf, 2*nf, kernel_size),
                        nn.PReLU(),
                        nn.BatchNorm3d(2*nf),
                        nn.Dropout3d(p=0.1)
                    )
                )
                nf *= 2
            self.most_conv = nn.Sequential(*most_conv)

        # compute the side length
        # Formula: (side_length - kernel_size + 2*padding)/stride + 1
        # Source: https://stackoverflow.com/a/54423584
        # works for both conv and maxpool
        side_length = data_input_size
        for _ in range(n_pool):
            side_length = (side_length - kernel_size) + 1
        self.fcn_input_size = side_length**3 * nf + 1
        assert side_length > 0, "Your network or your subvolume is too small!"
        print(f"The FCN has {self.fcn_input_size} inputs")

        # define the FCN
        fcn = []
        input_size = self.fcn_input_size

        for _ in range(n_fcn_layers-1):
            layer = nn.Sequential(
                nn.Linear(input_size, input_size // fcn_div_factor),
                nn.PReLU(),
                nn.BatchNorm1d(input_size // fcn_div_factor),
                nn.Dropout(p=0.1)
                )
            input_size = input_size // fcn_div_factor
            fcn.append(layer)
        
        # The final FCN layer
        final_layer = nn.Sequential(
            nn.Linear(input_size, n_output_val),
            nn.Sigmoid()
        )
        fcn.append(final_layer)

        # wrap up everything
        self.fcn = nn.Sequential(*fcn)
    
    def forward(self, x, t):
        """
        The input must have the following form: (x, t) where x is the cubic subvolume and t is the time for this cube
        """
        # The first convolutional layers
        x = self.first_conv(x)

        # Then all of the others
        if self.n_pool > 1:
            x = self.most_conv(x)
        
        # Flatten and append time to result
        x = x.view(x.shape[0], -1)
        x = torch.cat([x, t], dim=1)

        # the FCN layers
        return self.fcn(x)


# some tests
if __name__ == "__main__":
    print("\n\n\n%---------- CUBE CENTRAL CNN V2 ----------%")
    kernel_size = 3
    n_pool: int = 3
    subvolume_size = 9
    maxpool_size = 2
    maxpool_stride = 1
    n_input_channel = 3
    n_output_val = 1
    fcn_div_factor = 4
    n_fcn_layers = 6
    n_features = 64

    model = CentralCNNV2(n_input_channel, n_output_val, n_pool, n_features, kernel_size, subvolume_size, n_fcn_layers, fcn_div_factor, maxpool_size, maxpool_stride)

    n_samples = 30
    n_channels = 3
    data = torch.rand((n_samples, n_channels, subvolume_size, subvolume_size, subvolume_size))
    time = torch.Tensor([0.3]*n_samples).view(n_samples, -1)

    from torchinfo import summary
    summary(model, [data.shape, time.shape])

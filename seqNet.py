import torch
import torch.nn as nn
import numpy as np

# DEFAULT SEQNET: python3 /Users/williamphan/Desktop/seqNet/main.py --mode train --pooling seqnet --dataset nordland-sw --seqL 10 --w 5 --outDims 4096 --expName "w5" --nocuda

# SEQNET-MIX: python3 /Users/williamphan/Desktop/seqNet/main.py --mode train --pooling seqnet_mix --dataset nordland-sw --seqL 10 --w 5 --outDims 4096 --expName "w5" --nocuda

# parser.add_argument('--pooling', type=str, default='seqnet', help='type of pooling to use', choices=[ 'seqnet', 'smooth', 'delta', 'single','single+seqmatch', 's1+seqmatch'])
# parser.add_argument('--seqL', type=int, default=5, help='Sequence Length')
# parser.add_argument('--w', type=int, default=3, help='filter size for seqNet')

"""
ORIGINAL SEQNET
"""


class seqNet(nn.Module):
    def __init__(self, inDims, outDims, seqL, w=5):

        super(seqNet, self).__init__()
        self.inDims = inDims
        self.outDims = outDims
        self.w = w 
        self.conv = nn.Conv1d(inDims, outDims, kernel_size=self.w)

    def forward(self, x):
        print(f"X Shape: {x.shape}")
        # Input X Shape: torch.Size([24, 10, 4096]) - [batch size, input channels, dimensions]
        
        if len(x.shape) < 3:
            x = x.unsqueeze(1) # convert [B,C] to [B,1,C]

        x = x.permute(0,2,1) # from [B,T,C] to [B,C,T]
        seqFt = self.conv(x)
        seqFt = torch.mean(seqFt,-1) # Average pooling over the temporal dimension
        print("Sequence Feature Shape",seqFt.shape)
        # Sequence Feature Shape torch.Size([24, 4096]) - [batch size, output feature dimensions]

        return seqFt
    
"""
The Delta class highlights how feature values change over a sequence. It does this by using a special set of weights: 

- earlier parts of the sequence get negative weights
- later parts get positive weights

It calculates how features change over time within a sequence. — whether they tend to increase, decrease, or stay relatively constant over the sequence.
"""
    
class Delta(nn.Module):
    def __init__(self, inDims, seqL):
        super(Delta, self).__init__()
        self.inDims = inDims  # Number of input dimensions
        # Create a weighting vector to compute the delta (change) across the sequence
        self.weight = (np.ones(seqL, np.float32)) / (seqL / 2.0)
        self.weight[:seqL // 2] *= -1  # Negative weights for the first half, positive for the second
        self.weight = nn.Parameter(torch.from_numpy(self.weight), requires_grad=False)  # Convert to a tensor and set as a non-trainable parameter

    def forward(self, x):
        # Rearrange dimensions: [B,T,C] to [B,C,T] to align with weight vector for matrix multiplication
        x = x.permute(0, 2, 1)
        # Apply the weighting vector to compute the delta across the sequence for each feature
        delta = torch.matmul(x, self.weight)

        return delta

"""
SEQNET-MIX
"""

# MLP Layer
class FeatureMixer(nn.Module):
    def __init__(self, channels, num_blocks):
        super(FeatureMixer, self).__init__()
        self.mlp_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(channels, channels),  # W_1
                nn.ReLU(),                      # ReLU activation
                nn.Linear(channels, channels)   # W_2
            ) for _ in range(num_blocks)
        ])
        
        # Weight and bias initialization
        for block in self.mlp_blocks:
            for layer in block:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0.0, std=0.02)  # Assuming normal distribution as a stand-in for trunc_normal_
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x):
        B, C, T = x.shape
        x = x.view(B * T, C)  # Reshape to [B*T, C]
        for mlp in self.mlp_blocks:
            x_res = mlp(x)
            x = x + x_res  # Apply skip connection
        x = x.view(B, C, T)  # Reshape back to original shape
        return x

class seqNet_mix(nn.Module):
    def __init__(self, inDims, outDims, seqL, w=5, num_mixer_blocks=2):
        super(seqNet_mix, self).__init__()
        self.inDims = inDims
        self.outDims = outDims
        self.w = w
        self.conv = nn.Conv1d(inDims, outDims, kernel_size=self.w)
        self.feature_mixer = FeatureMixer(outDims, num_mixer_blocks)

    def forward(self, x):
        if len(x.shape) < 3:
            x = x.unsqueeze(1)  # convert [B,C] to [B,1,C]
        x = x.permute(0, 2, 1)  # from [B,T,C] to [B,C,T]
        seqFt = self.conv(x)
        seqFt = self.feature_mixer(seqFt)  # Pass through FeatureMixer
        seqFt = torch.mean(seqFt, -1)
        print("Sequence Feature Shape",seqFt.shape)
        return seqFt
    

    # loss: .28, .33, .28 around



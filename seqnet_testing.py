import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# DEFAULT SEQNET: python3 /Users/williamphan/Desktop/seqNet/main.py --mode train --pooling seqnet --dataset nordland-sw --seqL 10 --w 5 --outDims 4096 --expName "w5" --nocuda

# SEQNET-MIX: python3 main.py --mode train --pooling seqnet_mix --dataset nordland-sw --seqL 10 --w 5 --outDims 4096 --expName "w5" --nocuda
# SEQNET-MIX: python3 /Users/williamphan/Desktop/developer/projects/seqNet/train.py --mode train --pooling seqnet_mix --dataset nordland-sw --seqL 10 --w 5 --outDims 4096 --expName "w5" --nocuda

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
        # print(f"X Shape: {x.shape}")
        # Input X Shape: torch.Size([24, 10, 4096]) - [batch size, sequence length, dimensions]
        
        if len(x.shape) < 3:
            x = x.unsqueeze(1) # convert [B,C] to [B,1,C]

        x = x.permute(0,2,1) # from [B,T,C] to [B,C,T]
        seqFt = self.conv(x)
        seqFt = torch.mean(seqFt,-1) # Average pooling over the temporal dimension
        # print("Sequence Feature Shape",seqFt.shape)
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
# class FeatureMixer(nn.Module):
#     def __init__(self, channels, num_blocks):
#         super(FeatureMixer, self).__init__()
#         self.mlp_blocks = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(channels, channels),  # W_1
#                 nn.ReLU(),                      # ReLU activation
#                 nn.Linear(channels, channels)   # W_2
#             ) for _ in range(num_blocks)
#         ])
        
#         # Weight and bias initialization
#         for block in self.mlp_blocks:
#             for layer in block:
#                 if isinstance(layer, nn.Linear):
#                     nn.init.normal_(layer.weight, mean=0.0, std=0.02)  # Assuming normal distribution as a stand-in for trunc_normal_
#                     if layer.bias is not None:
#                         nn.init.zeros_(layer.bias)

#     def forward(self, x):
#         # batch size, channels, seq length
#         B, C, T = x.shape
#         x = x.view(B * T, C)  # Reshape to [B*T, C]
#         for mlp in self.mlp_blocks:
#             x_res = mlp(x)
#             x = x + x_res  # Apply skip connection
#         x = x.view(B, C, T)  # Reshape back to original shape
#         return x

class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim), # applied across the input dimension, normalizes values so that each feature has a mean close to zero and a variance around one
            nn.Linear(in_dim, int(in_dim * mlp_ratio)), # Linear transformation to adjust dimensionality of features, increasing them by a factor of 'mlp_ratio'. Captures complex relationships between features by projecting them into a higher-dimensional space
            nn.ReLU(), # Applies the Rectified Linear Unit (ReLU) activation function to introduce non-linearity.
            nn.Linear(int(in_dim * mlp_ratio), in_dim), # map features back to the original dimensionality.
        )

        # iterate over all modules of FeatureMixerLayer
        for m in self.modules():
            # check if linear layer
            if isinstance(m, (nn.Linear)):
                # if linear, initializes its weights using a truncated normal distribution with a standard deviation of 0.02. helps prevent the gradients from vanishing or exploding, provides a good starting base
                nn.init.trunc_normal_(m.weight, std=0.02)
                # if bias term, initializes the bias of linear layers to zero
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # combines original input with transformed version produced by FeatureMixerLayer
    def forward(self, x):
        #print("Feature Mixer Input", x.shape)
        # Pass the input tensor x through the FeatureMixerLayer (self.mix)
        # and add the output of the mixer to the input tensor x
        return x + self.mix(x)


class MixVPR(nn.Module):
    def __init__(self,
                 in_channels=4096,
                 in_h=1,
                 in_w=6,
                 out_channels=4096,
                 mix_depth=6,
                 mlp_ratio=1,
                 out_rows=10,
                 ) -> None:
        super().__init__()

        self.in_h = in_h # height of input feature maps
        self.in_w = in_w # width of input feature maps
        self.in_channels = in_channels # depth of input feature maps
        
        self.out_channels = out_channels # depth wise projection dimension
        self.out_rows = out_rows # row wise projection dimesion

        self.mix_depth = mix_depth # L the number of stacked FeatureMixers
        self.mlp_ratio = mlp_ratio # ratio of the mid projection layer in the mixer block

        hw = in_h*in_w
        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio)
            for _ in range(self.mix_depth)
        ])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x):
        #print("MixVPR Input",x.shape) # [24, 4096, 6]
        x = x.flatten(2)
        #print("MixVPR Flatten 1",x.shape) # [24, 4096, 6]
        x = self.mix(x)
        #print("MixVPR Mixer 1",x.shape)
        x = x.permute(0, 2, 1)
        #print("MixVPR Permute 1",x.shape)
        x = self.channel_proj(x)
        #print("MixVPR Channel Proj",x.shape)
        x = x.permute(0, 2, 1)
        #print("MixVPR Permute 2",x.shape)
        x = self.row_proj(x)
        #print("MixVPR Row Proj",x.shape)
        # x = F.normalize(x.flatten(1), p=2, dim=-1)
        # print("Normalize",x.shape)
        return x


class seqNet_mix(nn.Module):
    # w: kernal size
    def __init__(self, inDims, outDims, seqL, w=5, num_mixer_blocks=4):
        super(seqNet_mix, self).__init__()
        self.inDims = inDims
        self.outDims = outDims
        self.w = w
        self.conv = nn.Conv1d(inDims, outDims, kernel_size=self.w)
        self.aggregator = MixVPR(out_channels=outDims, mix_depth=num_mixer_blocks)

    def forward(self, x):
        #print("SeqNet Input",x.shape) # [24, 10, 4096] - [batch size, sequence length, dimensions]
        if len(x.shape) < 3:
            x = x.unsqueeze(1)  # convert [B,C] to [B,1,C]
        x = x.permute(0, 2, 1)  # from [B,T,C] to [B,C,T]
        #print("SeqNet Permute",x.shape) # [24, 4096, 10]
        seqFt = self.conv(x)
        #print("SeqNet Conv",seqFt.shape) # [24, 4096, 6]
        seqFt = self.aggregator(seqFt)  # pass through FeatureMixer
        seqFt = torch.mean(seqFt, -1) # returns averaged tensor, getting rid of last dim
        # print("Sequence Feature Shape",seqFt.shape) [24, 4096] - returns one tensor, seq len 1
        return seqFt
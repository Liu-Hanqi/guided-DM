import torch
from guided_diffusion.axial_attention import AxialAttention

img = torch.randn(2, 128, 32, 32)

attn = AxialAttention(
    dim = 128,               # embedding dimension
    dim_index = 1,         # where is the embedding dimension
    dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
    heads = 1,             # number of heads for multi-head attention
    num_dimensions = 2,    # number of axial dimensions (images is 2, video is 3, or more)
    sum_axial_out = True   # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
)
print(attn)

test = attn(img) # (2, 3, 256, 256)
print(test.shape)
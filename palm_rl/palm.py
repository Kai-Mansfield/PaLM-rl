import math
import copy
from pathlib import Path
from collections import namedtuple
from functools import wraps
from itertools import zip_longest

from tqdm import tqdm
from beartype import beartype
from beartype.typing import Tuple, Optional

import torch
from torch import einsum, nn
import torch.nn.functional as F

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce

from palm_rlhf_pytorch.attention import Attention
from palm_rlhf_pytorch.utils import top_p, top_k, masked_mean, gumbel_sample, eval_decorator
from palm_rlhf_pytorch.lora import LoRA


def exists(val):
    return val is not None
  
def default(val, d):
    reutn val if exists(val) else d
    
def identity(t, *args, **kwargs):
    return t
  
def l2norm(t):
    return F.normalize(t, dim = -1)
  
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeroes(dim))
        
    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

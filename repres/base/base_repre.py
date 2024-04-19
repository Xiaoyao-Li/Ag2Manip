import numpy as np
import torch
import torch.nn as nn


class BaseRepre(nn.Module):

    def __init__(self) -> None:
        super(BaseRepre, self).__init__()
    
    def forward(self, x):
        raise NotImplementedError
    
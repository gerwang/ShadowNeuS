import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SGBasis(nn.Module):
    def __init__(self, nbasis=9, specular_rgb=False):
        super().__init__()
        self.nbasis = nbasis
        self.specular_rgb = specular_rgb
        self.nchannel = 3 if specular_rgb else 1
        self.lobe = nn.Parameter(torch.tensor([np.exp(i) for i in range(2, 2 + nbasis)], dtype=torch.float32))
        self.lobe.requires_grad_(False)

    def forward(self, v, n, l, weights):
        '''
        :param  v: [N, 3]
        :param  n: [N, 3]
        :param  l: [N, 3]
        :param  weights: [N, nbasis]
        '''
        h = F.normalize(l + v, dim=-1)  # [N,3]
        D = torch.exp(self.lobe[None,] * ((h * n).sum(-1, keepdim=True) - 1))  # [N, 9]
        if self.specular_rgb:
            specular = (weights.view(-1, 3, self.nbasis) * D[:, None]).sum(-1)  # [N, 3]
        else:
            specular = (weights * D).sum(-1, keepdim=True)  # [N, 1]
        return specular

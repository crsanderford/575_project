import torch
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np

from torchdiffeq import odeint

from neuralop.models import FNO2d, FNO



def step_fd(u, kappa, dt, dx):

    u_pad = F.pad(u.unsqueeze(0).unsqueeze(0), (1,1,1,1), mode='reflect').squeeze()
    lap = (
        u_pad[1:-1,2:] + u_pad[1:-1,:-2] +
        u_pad[2:,1:-1] + u_pad[:-2,1:-1] -
        4*u_pad[1:-1,1:-1]
    ) / dx**2
    return u + dt * kappa * lap

def solve_heat_neumann(u0, kappa=0.1, T=1.0, dt=0.01, dx=1.0):

    us = []
    us.append(u0.clone())

    u = u0.clone()
    n_steps = int(T / dt)
    for _ in range(n_steps):
        u = step_fd(u, kappa, dt, dx)
        us.append(u.clone())
    return us






class InverseHeatNDO(nn.Module):
    def __init__(self, modes=(16,16), hidden_channels=32, atol=1e-5, rtol=1e-5):

        super().__init__()

        self.fno = FNO2d(
            in_channels=1,
            out_channels=1,
            n_modes_width=modes[0],
            n_modes_height=modes[1],
            hidden_channels=hidden_channels,
        )

        self.atol = atol
        self.rtol = rtol

    def forward(self, uT, s_span=None):

        if s_span is None:
            s_span = torch.tensor([0.0, 1.0], device=uT.device)

        def vfield(s, u):

            return self.fno(u)

        us = odeint(vfield, uT, s_span, atol=self.atol, rtol=self.rtol)
        return us
    

class InverseHeatNDOWarped(nn.Module):
    def __init__(self, modes=(16,16), hidden_channels=32, atol=1e-5, rtol=1e-5):

        super().__init__()

        self.fno = FNO2d(
            in_channels=1,
            out_channels=1,
            n_modes_width=modes[0],
            n_modes_height=modes[1],
            hidden_channels=hidden_channels,
        )

        self.atol = atol
        self.rtol = rtol

    def forward(self, uT, s_span=None):

        if s_span is None:
            s_span = torch.tensor([0.0, 1e-6], device=uT.device)

        def vfield(s, u):

            return torch.exp(-s) * self.fno(u)

        us = odeint(vfield, uT, s_span, atol=self.atol, rtol=self.rtol)
        return us
    



    

class SimpleConvNet(nn.Module):
    def __init__(self, h_dim=16):
        super(SimpleConvNet, self).__init__()


        self.skip0 = nn.Conv2d(1,1, kernel_size=(5,5), padding='same')

        self.conv1 = nn.Conv2d(1, h_dim, kernel_size=(5,5), padding='same')
        self.conv2 = nn.Conv2d(h_dim, h_dim, kernel_size=(5,5), padding='same')

        self.skip1 = nn.Conv2d(h_dim, 1, kernel_size=(5,5), padding='same')
        self.skip2 = nn.Conv2d(h_dim, 1, kernel_size=(5,5), padding='same')

        self.out_conv = nn.Conv2d(h_dim, 1, kernel_size=(7,7), padding='same')

        
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = x


        x = self.act(self.conv1(x))
        out = out + self.skip1(x)


        x = self.act(self.conv2(x))
        out = out + self.skip2(x)


        out = out + self.out_conv(x)
        return out
    




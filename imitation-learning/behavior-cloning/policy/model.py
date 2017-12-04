import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, dim_in, dim_out, n_hidden=128):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(dim_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, dim_out)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

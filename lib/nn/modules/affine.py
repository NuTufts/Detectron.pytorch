import torch
import torch.nn as nn
import sparseconvnet as scn


class AffineChannel2d(nn.Module):
    """ A simple channel-wise affine transformation operation """
    def __init__(self, num_features):
        super(AffineChannel2d,self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.weight.data.uniform_()
        self.bias.data.zero_()

    def forward(self, x):
        return x * self.weight.view(1, self.num_features, 1, 1) + \
            self.bias.view(1, self.num_features, 1, 1)


class SparseAffineChannel2d(nn.Module):
    """ A simple channel-wise affine transformation operation """
    def __init__(self, num_features):
        super(SparseAffineChannel2d,self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.weight.data.uniform_()
        self.bias.data.zero_()

    def forward(self, x): #use torch.multiply()
        # print("input: ", x.features.shape)
        x.features = x.features * self.weight.view(1, self.num_features) + \
            self.bias.view(1, self.num_features)
        # print("output: ", x.features.shape)
        return x
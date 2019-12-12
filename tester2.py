import torch
import torch.nn as nn

import nn as mynn

import time

import lib.modeling.ResNet as ResNet
# import SparseResNet
import sparseconvnet as scn

class Model(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        # self.res1 = scn.Sequential(scn.SubmanifoldConvolution(2, 3, 64, 7, False),SparseAffineChannel2d(64),scn.ReLU(),scn.MaxPooling(2, 3, 2)
        #     )
        self.res1 = scn.Sequential(scn.Convolution(2, 3, 64, 7, 2, False),SparseAffineChannel2d(64),scn.ReLU(),scn.MaxPooling(2, 3, 2)
            )
        self.res2 = scn.Sequential(sparse_bottleneck(64, 256, 64, downsample=basic_bn_shortcut(64, 256, 1)),
            sparse_bottleneck(256, 256, 64),
            sparse_bottleneck(256, 256, 64))
        self.res3 = scn.Sequential(sparse_bottleneck(256, 512, 128, stride=2, downsample=basic_bn_shortcut(256, 512, 2)),
            sparse_bottleneck(512, 512, 128),
            sparse_bottleneck(512, 512, 128),
            sparse_bottleneck(512, 512, 128))
        self.res4 = scn.Sequential(sparse_bottleneck(512, 1024, 256, stride=2, downsample=basic_bn_shortcut(512, 1024, 2)),
            sparse_bottleneck(1024, 1024, 256),
            sparse_bottleneck(1024, 1024, 256),
            sparse_bottleneck(1024, 1024, 256),
            sparse_bottleneck(1024, 1024, 256),
            sparse_bottleneck(1024, 1024, 256))
        self.desparsify = scn.SparseToDense(2, 1024);

    def forward(self, x):
        print("x.features.shape: ", x.features.shape)
        print("x.spatial_size: ", x.spatial_size)

        out = self.res1(x)
        print()
        print("res1out .features.shape: ", out.features.shape)
        print("res1out.spatial_size: ", out.spatial_size)


        out = self.res2(out)
        print()
        print("res2out.features.shape: ", out.features.shape)
        print("res2out.spatial_size: ", out.spatial_size)

        out = self.res3(out)
        print()
        print("res3out .features.shape: ", out.features.shape)
        print("res3out.spatial_size: ", out.spatial_size)

        out = self.res4(out)
        print()
        print("sparse_out_before_desparsify.features.shape", out.features.shape)
        print("sparse_out_before_desparsify.spatial_size: ", out.spatial_size)

        import pdb; pdb.set_trace()
        out = self.desparsify(out)
        return out

class sparse_bottleneck(nn.Module):
    """ Bottleneck Residual Block """

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, group=1,
                 downsample=None):
        super(sparse_bottleneck,self).__init__()
        # In original resnet, stride=2 is on 1x1.
        # In fb.torch resnet, stride=2 is on 3x3.
        #(str1x1, str3x3) = (stride, 1) if cfg.RESNETS.STRIDE_1X1 else (1, stride)
        #self.stride = stride

        self.conv1 = scn.SubmanifoldConvolution(2,
            inplanes, innerplanes, 1, False)
        if (stride !=1):
            print('here')
            self.conv1 = scn.Convolution(2, inplanes, innerplanes, 1, stride, False)

        self.bn1 = SparseAffineChannel2d(innerplanes)

        self.conv2 = scn.SubmanifoldConvolution(2,
            innerplanes, innerplanes, 3, False)

        self.bn2 = SparseAffineChannel2d(innerplanes)

        self.conv3 = scn.SubmanifoldConvolution(2,
            innerplanes, outplanes, 1, False)

        self.bn3 = SparseAffineChannel2d(outplanes)

        self.downsample = downsample
        self.relu = scn.ReLU()

    def forward(self, x):
        t_st = time.time()
        residual = x
        out = self.conv1(x)
        #out = self.desparsify1(out)
        out = self.bn1(out)
        out = self.relu(out)

        #out = self.sparsify2(out)
        out = self.conv2(out)
        #out = self.desparsify2(out)
        out = self.bn2(out)
        out = self.relu(out)

        #out = self.sparsify3(out)
        out = self.conv3(out)
        #out = self.desparsify3(out)
        out = self.bn3(out)
        #out = self.sparsify3(out)
        # print("bottleneck transform convs: %0.3f" %(time.time() - t_st))
        if self.downsample is not None:
            residual = self.downsample(x)

        #import pdb; pdb.set_trace()
        out.features += residual.features
        out = self.relu(out)

        return out

def basic_bn_shortcut(inplanes, outplanes, stride):
    if stride == 1:
        return nn.Sequential(
            scn.SubmanifoldConvolution(2, inplanes,
                    outplanes,
                    1,
                    False),
            SparseAffineChannel2d(outplanes),
        )
    else:
        return nn.Sequential(
            scn.SubmanifoldConvolution(2, inplanes,
                    outplanes,
                    1,
                    False),
            scn.MaxPooling(2, 1, 2),
            SparseAffineChannel2d(outplanes)
        )

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

resnet = ResNet.ResNet50_conv4_body()
print("ResNet")
print(resnet)
sparse_resnet = Model()
print("Sparse ResNet")
print(sparse_resnet)



input = torch.ones(1, 3, 523, 843)
input[0,:,512,:] = 0
input[0,:,513,:] = 0
input[0,:,514,:] = 0
input[0,:,515,:] = 0
input[0,:,516,:] = 0
input[0,:,517,:] = 0
input[0,:,518,:] = 0
input[0,:,519,:] = 0
input[0,:,520,:] = 0
input[0,:,521,:] = 0
input[0,:,522,:] = 0
input[0,:,:,832] = 0
input[0,:,:,833] = 0
input[0,:,:,834] = 0
input[0,:,:,835] = 0
input[0,:,:,836] = 0
input[0,:,:,837] = 0
input[0,:,:,838] = 0
input[0,:,:,839] = 0
input[0,:,:,840] = 0
input[0,:,:,841] = 0
input[0,:,:,842] = 0

input_dense = torch.ones(1,3,512,832)
print(input.shape)
# for i in range(10):
#     # print(i)
#     input[0,:,i,i] = 1
#     input_dense[0,:,i,i] = 1

print()
#import pdb;
print("Input Shape: " ,input.shape)
print("---------------------------")
sparsifier = scn.DenseToSparse(2)
input_sparse = sparsifier(input)
print("Input Sparse Shape: " ,input_sparse.features.shape)
print("------Running ResNet---------")
dense_out = resnet.forward(input_dense)
print("------Running SparseResNet---------")

sparse_out = sparse_resnet.forward(input_sparse)
print("dense_out.shape", dense_out.shape)
print("sparse_out.shape", sparse_out.shape)
print("State Dict of Dense Resnet")
for key in resnet.state_dict().keys():
    print(key)
print()
print("State Dict of Sparse Resnet")
for key in sparse_resnet.state_dict().keys():
    print(key)
print()
# print(sparse_out[:,:,:,51])
# print(sparse_out[:,:,31,:])

def convert_state_dict(src_dict):
    """Return the correct mapping of tensor name and value

    Mapping from the names of torchvision model to our resnet conv_body and box_head.
    """
    dst_dict = {}
    for k, v in src_dict.items():
        toks = k.split('.')
        if k.startswith('layer'):
            assert len(toks[0]) == 6
            res_id = int(toks[0][5]) + 1
            name = '.'.join(['res%d' % res_id] + toks[1:])
            dst_dict[name] = v
        elif k.startswith('fc'):
            continue
        else:
            name = '.'.join(['res1'] + toks)
            dst_dict[name] = v
    return dst_dict
#
#
import os
import os.path as osp
weights_file = "/home/jmills/workdir/mask-rcnn.pytorch/data/pretrained_model/resnet50_caffe.pth"
pretrianed_state_dict = convert_state_dict(torch.load(weights_file))
print("Checkpoint keys")
for key in pretrianed_state_dict.keys():
    print(key)

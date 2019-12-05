import torch
import torch.nn as nn

import nn as mynn

import time

import ResNet
import modeling.SparseResNet as SparseResNet
import sparseconvnet as scn

resnet = ResNet.ResNet50_conv4_body()
print("ResNet")
print(resnet)
sparse_resnet = SparseResNet.ResNet50_conv4_body()
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

sparse_out = sparse_resnet.forward(input_dense)
print("dense_out.shape", dense_out.shape)
print("sparse_out.shape", sparse_out.shape)

counter_dense = 0
counter_sparse = 0
counter_ckpt = 0

print("State Dict of Dense Resnet")
for key in resnet.state_dict().keys():
    print(key)
    counter_dense = counter_dense + 1
print("DENSE KEYS: ", counter_dense)
print("State Dict of Sparse Resnet")
import pdb; pdb.set_trace()
for key in sparse_resnet.state_dict().keys():
    print(key)
    counter_sparse = counter_sparse + 1
print("SPARSE KEYS: ", counter_sparse)
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
    counter_ckpt = counter_ckpt + 1
print("CKPT KEYS: ", counter_ckpt)

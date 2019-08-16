# Test Script to run forward on sparse resnet
import argparse
import os
import sys
import pickle
import resource
import traceback
import logging
from collections import defaultdict

import numpy as np
import yaml
import torch
from torch.autograd import Variable
import torch.nn as nn
import cv2
cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader

import _init_paths  # pylint: disable=unused-import
import nn as mynn
import utils.net as net_utils
import utils.misc as misc_utils
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from datasets.roidb import combined_roidb_for_training
from roi_data.loader import RoiDataLoader, MinibatchSampler, BatchSampler, collate_minibatch
from modeling.model_builder import Generalized_RCNN
from utils.detectron_weight_helper import load_detectron_weight
from utils.logging import setup_logging
from utils.timer import Timer
from utils.training_stats import TrainingStats

#to Vis
import datasets.dummy_datasets as datasets
import numpy as np
import utils.vis as vis_utils
import cv2

import time
import random
import sparseconvnet as scn
# original 9317
coord  = np.random.randint(1,510,size=(1000,2))
values = np.random.random_sample(size=(1000,1))
img = np.zeros((1,3,512,832),dtype=np.float32)
print(img.shape)
for idx in range(values.shape[0]):
    img[0,:,coord[idx][0],coord[idx][1]] = values[idx][0]
device_sparse = 'cuda:0'
device_dense = 'cuda:0'

img_coords = torch.from_numpy(coord).type(torch.LongTensor).to(torch.device(device_sparse))
img_values = torch.from_numpy(values).type(torch.FloatTensor).to(torch.device(device_sparse))
img_torch = torch.from_numpy(img).to(torch.device(device_dense))


# RESNET = scn.Sequential(scn.InputLayer(2, (512,832), 0),
#         scn.MaxPooling(2,16,16),
#         scn.SparseResNet(2,1,[['basic',64,2,1],['basic',256,2,1],['basic',512,2,1],['basic',1024,2,1],]),
#         scn.SparseToDense(2,1024)).to(torch.device(device_sparse))
# Sparse ResNet
input = scn.InputLayer(2, (512,832), 0).to(torch.device(device_sparse))
max_pool = scn.MaxPooling(2,16,16).to(torch.device(device_sparse))
resnet = scn.SparseResNet(2,1,[['basic',64,2,1],['basic',256,2,1],['basic',512,2,1],['basic',1024,2,1]]).to(torch.device(device_sparse))
sparse2dense = scn.SparseToDense(2,1024).to(torch.device(device_sparse))


# Dense ResNet
print()
cfg_from_file("/home/jmills/workdir/sparse_mask/smask-rcnn/configs/baselines/mills_config_2.yaml")
assert_and_infer_cfg()
from modeling import ResNet
from modeling import model_builder
Dense_ResNet = model_builder.get_func("ResNet.ResNet50_conv4_body")()
Dense_ResNet.to(torch.device(device_dense))
torch.cuda.synchronize
t_st_d = time.time()
y = Dense_ResNet(img_torch)
torch.cuda.synchronize
t_dense = time.time() - t_st_d
# print(Dense_ResNet)
print("Time for Dense ResNet: %.5f" % t_dense)
print("Dense Out Shape", y.shape)
print()
print()




print("init net")
x = (img_coords,img_values)
torch.cuda.synchronize
t_st = time.time()
x = input(x)
torch.cuda.synchronize
t_inp = time.time() - t_st
print("Input time:      %.5f" % t_inp)

x = max_pool(x)
torch.cuda.synchronize
t_pool = time.time() - t_inp - t_st
print("Maxpool time:    %.5f" % t_pool)

x = resnet(x)
torch.cuda.synchronize
t_res = time.time() - t_pool - t_inp - t_st
print("Resnet time:     %.5f" % t_res)

output = sparse2dense(x)
torch.cuda.synchronize
t_dense = time.time() - t_res - t_pool-t_inp-t_st
print("Sparse to dense: %.5f" % t_dense)
# print(RESNET)
# torch.cuda.synchronize
# t_st = time.time()
# output = RESNET(x)
# torch.cuda.synchronize
print("Total Time: %.3f" % (time.time() -  t_st))

print("out shape:",output.shape)

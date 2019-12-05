from __future__ import division

import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import sparseconvnet as scn

from core.config import cfg
import nn as mynn
import utils.net as net_utils
from utils.sparseresnet_weights_helper import convert_state_dict
import time

# ---------------------------------------------------------------------------- #
# Bits for specific architectures (ResNet50, ResNet101, ...)
# ---------------------------------------------------------------------------- #

def ResNet50_conv4_body():
    return SparseResNet()


# ---------------------------------------------------------------------------- #
# Generic ResNet components
# ---------------------------------------------------------------------------- #

class SparseResNet(nn.Module):
    def __init__(self):
        super(SparseResNet,self).__init__()
        self.block_counts = (3, 4, 6)
        self.convX = len(self.block_counts) + 1
        self.res1 = scn.Sequential(OrderedDict([('conv1',scn.Convolution(2, 3, 64, 7, 2, False)), 
            ('bn1',mynn.SparseAffineChannel2d(64)),
            ('relu',scn.ReLU()),
            ('maxpool',scn.MaxPooling(2, 3, 2))
            ]))
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

        self.spatial_scale = 1 / 16
        self.dim_in = 64
        self.dim_out = 1024
        self._init_modules()


    def _init_modules(self):
        assert cfg.RESNETS.FREEZE_AT in [0, 2, 3, 4, 5]
        assert cfg.RESNETS.FREEZE_AT <= self.convX
        for i in range(1, cfg.RESNETS.FREEZE_AT + 1):
            freeze_params(getattr(self, 'res%d' % i))

        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, mynn.SparseAffineChannel2d) else None)

    def detectron_weight_mapping(self):
        if cfg.RESNETS.USE_GN:
            mapping_to_detectron = {
                'res1.conv1.weight': 'conv1_w',
                'res1.gn1.weight': 'conv1_gn_s',
                'res1.gn1.bias': 'conv1_gn_b',
            }
            orphan_in_detectron = ['pred_w', 'pred_b']
        else:
            mapping_to_detectron = {
                'res1.conv1.weight': 'conv1_w',
                'res1.bn1.weight': 'res_conv1_bn_s',
                'res1.bn1.bias': 'res_conv1_bn_b',
            }
            orphan_in_detectron = ['conv1_b', 'fc1000_w', 'fc1000_b']

        for res_id in range(2, self.convX + 1):
            stage_name = 'res%d' % res_id
            mapping, orphans = residual_stage_detectron_mapping(
                getattr(self, stage_name), stage_name,
                self.block_counts[res_id - 2], res_id)
            mapping_to_detectron.update(mapping)
            orphan_in_detectron.extend(orphans)

        return mapping_to_detectron, orphan_in_detectron

    def train(self, mode=True):
        # Override
        self.training = mode

        for i in range(cfg.RESNETS.FREEZE_AT + 1, self.convX + 1):
            getattr(self, 'res%d' % i).train(mode)

    def forward(self, x):
        t_st = time.time()
        print("Sparse Res Net start")
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

        print("Time Spent Doing Sparse ResNet: %0.3f" %(time.time() - t_st))

        out = self.desparsify(out)
        out = torch.split(out, out.shape[3] - 1, 3)[0]
        print("output shape", out.shape)
        return out

# ------------------------------------------------------------------------------
# various downsample shortcuts (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

def basic_bn_shortcut(inplanes, outplanes, stride):
    if stride == 1:
        return nn.Sequential(
            scn.SubmanifoldConvolution(2, inplanes,
                    outplanes,
                    1,
                    False),
            mynn.SparseAffineChannel2d(outplanes),
        )
    else:
        return nn.Sequential(
            scn.SubmanifoldConvolution(2, inplanes,
                    outplanes,
                    1,
                    False),
            mynn.SparseAffineChannel2d(outplanes),
            scn.MaxPooling(2, 1, 2)
        )

# ------------------------------------------------------------------------------
# various stems (may expand and may consider a new helper)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# various transformations (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

class sparse_bottleneck(nn.Module):
    """ Sparse Bottleneck Residual Block """

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

        self.bn1 = mynn.SparseAffineChannel2d(innerplanes)

        self.conv2 = scn.SubmanifoldConvolution(2,
            innerplanes, innerplanes, 3, False)

        self.bn2 = mynn.SparseAffineChannel2d(innerplanes)

        self.conv3 = scn.SubmanifoldConvolution(2,
            innerplanes, outplanes, 1, False)

        self.bn3 = mynn.SparseAffineChannel2d(outplanes)

        self.downsample = downsample
        self.relu = scn.ReLU()

    def forward(self, x):
        t_st = time.time()
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # print("bottleneck transform convs: %0.3f" %(time.time() - t_st))
        if self.downsample is not None:
            residual = self.downsample(x)

        out.features += residual.features
        out = self.relu(out)

        return out

# ---------------------------------------------------------------------------- #
# Helper functions
# ---------------------------------------------------------------------------- #

def residual_stage_detectron_mapping(module_ref, module_name, num_blocks, res_id):
    """Construct weight mapping relation for a residual stage with `num_blocks` of
    residual blocks given the stage id: `res_id`
    """
    if cfg.RESNETS.USE_GN:
        norm_suffix = '_gn'
    else:
        norm_suffix = '_bn'
    mapping_to_detectron = {}
    orphan_in_detectron = []
    for blk_id in range(num_blocks):
        detectron_prefix = 'res%d_%d' % (res_id, blk_id)
        my_prefix = '%s.%d' % (module_name, blk_id)

        # residual branch (if downsample is not None)
        if getattr(module_ref[blk_id], 'downsample'):
            dtt_bp = detectron_prefix + '_branch1'  # short for "detectron_branch_prefix"
            mapping_to_detectron[my_prefix
                                 + '.downsample.0.weight'] = dtt_bp + '_w'
            orphan_in_detectron.append(dtt_bp + '_b')
            mapping_to_detectron[my_prefix
                                 + '.downsample.1.weight'] = dtt_bp + norm_suffix + '_s'
            mapping_to_detectron[my_prefix
                                 + '.downsample.1.bias'] = dtt_bp + norm_suffix + '_b'

        # conv branch
        for i, c in zip([1, 2, 3], ['a', 'b', 'c']):
            dtt_bp = detectron_prefix + '_branch2' + c
            mapping_to_detectron[my_prefix
                                 + '.conv%d.weight' % i] = dtt_bp + '_w'
            orphan_in_detectron.append(dtt_bp + '_b')
            mapping_to_detectron[my_prefix
                                 + '.' + norm_suffix[1:] + '%d.weight' % i] = dtt_bp + norm_suffix + '_s'
            mapping_to_detectron[my_prefix
                                 + '.' + norm_suffix[1:] + '%d.bias' % i] = dtt_bp + norm_suffix + '_b'
    return mapping_to_detectron, orphan_in_detectron


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False

from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

import numpy as np

from core.config import cfg
import nn as mynn
import utils.net as net_utils
import time



class fast_rcnn_outputs(nn.Module):
    def __init__(self, dim_in, validation=False):
        super(fast_rcnn_outputs,self).__init__()
        self.cls_score = nn.Linear(dim_in, cfg.MODEL.NUM_CLASSES)
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:  # bg and fg
            self.bbox_pred = nn.Linear(dim_in, 4 * 2)
        else:
            self.bbox_pred = nn.Linear(dim_in, 4 * cfg.MODEL.NUM_CLASSES)

        self.validation=validation
        self._init_weights()

    def _init_weights(self):
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, 0)
        init.normal_(self.bbox_pred.weight, std=0.001)
        init.constant_(self.bbox_pred.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'cls_score.weight': 'cls_score_w',
            'cls_score.bias': 'cls_score_b',
            'bbox_pred.weight': 'bbox_pred_w',
            'bbox_pred.bias': 'bbox_pred_b'
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        cls_score = self.cls_score(x)
        if not self.training and not self.validation:
            cls_score = F.softmax(cls_score, dim=1)
        bbox_pred = self.bbox_pred(x)

        return cls_score, bbox_pred


def fast_rcnn_losses(cls_score, bbox_pred, label_int32, bbox_targets,
                     bbox_inside_weights, bbox_outside_weights):
    device_id = ''
    if cls_score.is_cuda:
        device_id = cls_score.get_device()
    else:
        device_id = 'cpu'
    rois_label = Variable(torch.from_numpy(label_int32.astype('int64'))).to(torch.device(device_id))

    bbox_targets = Variable(torch.from_numpy(bbox_targets)).to(torch.device(device_id))
    bbox_inside_weights = Variable(torch.from_numpy(bbox_inside_weights)).to(torch.device(device_id))
    bbox_outside_weights = Variable(torch.from_numpy(bbox_outside_weights)).to(torch.device(device_id))
    loss_bbox = net_utils.smooth_l1_loss(
        bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

    # class accuracy
    cls_preds = cls_score.max(dim=1)[1].type_as(rois_label)
    #new weights
    # for el in range(len(cls_preds)):
        # print("Pred", cls_preds[el].item())
        # print("GT: ", rois_label[el].item())
    #new
    numerator   = (( rois_label > 0 ).float() * cls_preds.eq(rois_label).float() ).float().sum(dim=0).float()
    denominator = (rois_label > 0 ).float().sum(dim=0).float()

    num_cosm = (( rois_label == 1 ).float() * cls_preds.eq(rois_label).float() ).float().sum(dim=0).float()
    num_neut = (( rois_label == 5 ).float() * cls_preds.eq(rois_label).float() ).float().sum(dim=0).float()
    num_back = (( rois_label == 0 ).float() * cls_preds.eq(rois_label).float() ).float().sum(dim=0).float()

    den_cosm = (rois_label == 1 ).float().sum(dim=0).float()
    den_neut = (rois_label == 5 ).float().sum(dim=0).float()
    den_back = (rois_label == 0 ).float().sum(dim=0).float()

    # print("My Accuracy: ", numerator/denominator.item())
    # print("Numerator: ", numerator.item())
    # print("Denominator: ", denominator.item())

    accuracy_cls = numerator/denominator
    accuracy_cosm = -1
    accuracy_neut = -1

    if den_neut != 0:
        accuracy_neut = num_neut/den_neut
    if den_cosm != 0:
        accuracy_cosm = num_cosm/den_cosm
    # accuracy_cls = cls_preds.eq(rois_label).float().mean(dim=0)
    # original
    # accuracy_cls = cls_preds.eq(rois_label).float().mean(dim=0)
    neut_weight = 0
    cosmic_weight = 0
    print("num_pred_neut: ", cls_preds.eq(5).float().sum(dim=0).float().item())
    print("num_pred_cosm: ", cls_preds.eq(1).float().sum(dim=0).float().item())
    print("num_pred_back: ", cls_preds.eq(0).float().sum(dim=0).float().item())
    print()
    print("num_correct_neut: ", num_neut.item())
    print("num_correct_cosm: ", num_cosm.item())
    print("num_correct_back: ", num_back.item())
    print()
    print("den_neut: ", den_neut.item())
    print("den_cosm: ", den_cosm.item())
    print("den_back: ", den_back.item())
    mult = 2. #upweight relative to background cases
    if (den_neut ==0) or (den_cosm ==0):
        neut_weight=1.*mult
        cosmic_weight=1.*mult
    else:
        neut_weight= float(den_cosm)/float(den_neut+den_cosm)*mult
        cosmic_weight= float(den_neut)/float(den_cosm+den_neut)*mult

    neut_weight = 3498/194*mult #this is the num of cosmics/num neutrinos
    cosmic_weight = 1*mult

    weight = np.zeros(cls_score.shape[1],np.float32)
    weight[0] = 1
    weight[1] = cosmic_weight
    weight[5] = neut_weight
     # = np.array([1,cosmic_weight,0,0,0,neut_weight,0], np.float32)
    weight = (torch.from_numpy(weight)).to(torch.device(device_id))
    loss_cls = F.cross_entropy(cls_score,rois_label,weight)



    return loss_cls, loss_bbox, accuracy_cls, accuracy_neut, accuracy_cosm


# ---------------------------------------------------------------------------- #
# Box heads
# ---------------------------------------------------------------------------- #

class roi_2mlp_head(nn.Module):
    """Add a ReLU MLP with two hidden layers."""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super(roi_2mlp_head,self).__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.dim_out = hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM

        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc1 = nn.Linear(dim_in * roi_size**2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        mynn.init.XavierFill(self.fc1.weight)
        init.constant_(self.fc1.bias, 0)
        mynn.init.XavierFill(self.fc2.weight)
        init.constant_(self.fc2.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'fc1.weight': 'fc6_w',
            'fc1.bias': 'fc6_b',
            'fc2.weight': 'fc7_w',
            'fc2.bias': 'fc7_b'
        }
        return detectron_weight_mapping, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        batch_size = x.size(0)
        x = F.relu(self.fc1(x.view(batch_size, -1)), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        return x


class roi_Xconv1fc_head(nn.Module):
    """Add a X conv + 1fc head, as a reference if not using GroupNorm"""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super(roi_Xconv1fc_head,self).__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
        module_list = []
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            module_list.extend([
                nn.Conv2d(dim_in, hidden_dim, 3, 1, 1),
                nn.ReLU(inplace=True)
            ])
            dim_in = hidden_dim
        self.convs = nn.Sequential(*module_list)

        self.dim_out = fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc = nn.Linear(dim_in * roi_size * roi_size, fc_dim)

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                mynn.init.MSRAFill(m.weight)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                init.constant_(m.bias, 0)
        self.apply(_init)

    def detectron_weight_mapping(self):
        mapping = {}
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            mapping.update({
                'convs.%d.weight' % (i*2): 'head_conv%d_w' % (i+1),
                'convs.%d.bias' % (i*2): 'head_conv%d_b' % (i+1)
            })
        mapping.update({
            'fc.weight': 'fc6_w',
            'fc.bias': 'fc6_b'
        })
        return mapping, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        batch_size = x.size(0)
        x = self.convs(x)
        x = F.relu(self.fc(x.view(batch_size, -1)), inplace=True)
        return x


class roi_Xconv1fc_gn_head(nn.Module):
    """Add a X conv + 1fc head, with GroupNorm"""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super(roi_Xconv1fc_gn_head,self).__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
        module_list = []
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            module_list.extend([
                nn.Conv2d(dim_in, hidden_dim, 3, 1, 1, bias=False),
                nn.GroupNorm(net_utils.get_group_gn(hidden_dim), hidden_dim,
                             eps=cfg.GROUP_NORM.EPSILON),
                nn.ReLU(inplace=True)
            ])
            dim_in = hidden_dim
        self.convs = nn.Sequential(*module_list)

        self.dim_out = fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc = nn.Linear(dim_in * roi_size * roi_size, fc_dim)

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                mynn.init.MSRAFill(m.weight)
            elif isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                init.constant_(m.bias, 0)
        self.apply(_init)

    def detectron_weight_mapping(self):
        mapping = {}
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            mapping.update({
                'convs.%d.weight' % (i*3): 'head_conv%d_w' % (i+1),
                'convs.%d.weight' % (i*3+1): 'head_conv%d_gn_s' % (i+1),
                'convs.%d.bias' % (i*3+1): 'head_conv%d_gn_b' % (i+1)
            })
        mapping.update({
            'fc.weight': 'fc6_w',
            'fc.bias': 'fc6_b'
        })
        return mapping, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        batch_size = x.size(0)
        x = self.convs(x)
        x = F.relu(self.fc(x.view(batch_size, -1)), inplace=True)
        return x

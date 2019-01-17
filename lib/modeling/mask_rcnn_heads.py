from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from core.config import cfg
from modeling import ResNet
import nn as mynn
import utils.net as net_utils

#to Vis
import datasets.dummy_datasets as datasets
import numpy as np
import utils.vis as vis_utils
import cv2


# ---------------------------------------------------------------------------- #
# Mask R-CNN outputs and losses
# ---------------------------------------------------------------------------- #

class mask_rcnn_outputs(nn.Module):
    """Mask R-CNN specific outputs: either mask logits or probs."""
    def __init__(self, dim_in):
        super().__init__()
        self.dim_in = dim_in

        n_classes = cfg.MODEL.NUM_CLASSES if cfg.MRCNN.CLS_SPECIFIC_MASK else 1
        if cfg.MRCNN.USE_FC_OUTPUT:
            # Predict masks with a fully connected layer
            self.classify = nn.Linear(dim_in, n_classes * cfg.MRCNN.RESOLUTION**2)
        else:
            # Predict mask using Conv
            self.classify = nn.Conv2d(dim_in, n_classes, 1, 1, 0)
            if cfg.MRCNN.UPSAMPLE_RATIO > 1:
                self.upsample = mynn.BilinearInterpolation2d(
                    n_classes, n_classes, cfg.MRCNN.UPSAMPLE_RATIO)
        self._init_weights()

    def _init_weights(self):

        if not cfg.MRCNN.USE_FC_OUTPUT and cfg.MRCNN.CLS_SPECIFIC_MASK and \
                cfg.MRCNN.CONV_INIT=='MSRAFill':
            # Use GaussianFill for class-agnostic mask prediction; fills based on
            # fan-in can be too large in this case and cause divergence
            weight_init_func = mynn.init.MSRAFill
        else:
            weight_init_func = partial(init.normal_, std=0.001)
        weight_init_func(self.classify.weight)
        init.constant_(self.classify.bias, 0)

    def detectron_weight_mapping(self):
        mapping = {
            'classify.weight': 'mask_fcn_logits_w',
            'classify.bias': 'mask_fcn_logits_b'
        }
        if hasattr(self, 'upsample'):
            mapping.update({
                'upsample.upconv.weight': None,  # don't load from or save to checkpoint
                'upsample.upconv.bias': None
            })
        orphan_in_detectron = []
        return mapping, orphan_in_detectron

    def forward(self, x):
        x = self.classify(x)
        if cfg.MRCNN.UPSAMPLE_RATIO > 1:
            x = self.upsample(x)
        if not self.training:
            x = F.sigmoid(x)
        return x


# def mask_rcnn_losses(mask_pred, rois_mask, rois_label, weight):
#     n_rois, n_classes, _, _ = mask_pred.size()
#     rois_mask_label = rois_label[weight.data.nonzero().view(-1)]
#     # select pred mask corresponding to gt label
#     if cfg.MRCNN.MEMORY_EFFICIENT_LOSS:  # About 200~300 MB less. Not really sure how.
#         mask_pred_select = Variable(
#             mask_pred.data.new(n_rois, cfg.MRCNN.RESOLUTION,
#                                cfg.MRCNN.RESOLUTION))
#         for n, l in enumerate(rois_mask_label.data):
#             mask_pred_select[n] = mask_pred[n, l]
#     else:
#         inds = rois_mask_label.data + \
#           torch.arange(0, n_rois * n_classes, n_classes).long().cuda(rois_mask_label.data.get_device())
#         mask_pred_select = mask_pred.view(-1, cfg.MRCNN.RESOLUTION,
#                                           cfg.MRCNN.RESOLUTION)[inds]
#     loss = F.binary_cross_entropy_with_logits(mask_pred_select, rois_mask)
#     return loss


def mask_rcnn_losses(masks_pred, masks_int32):
    """Mask R-CNN specific losses."""
    # print('Taking a loss')
    # print('Shape of truth:', masks_int32.shape)
    # print('Shape of pred: ', masks_pred.shape)

    n_rois, n_classes, _, _ = masks_pred.size()
    device_id = masks_pred.get_device()
    masks_gt = Variable(torch.from_numpy(masks_int32.astype('float32'))).cuda(device_id)
    el1 = float(np.amax(masks_int32))
    el2 = float(torch.max(masks_gt))

    weight = (masks_gt > -1).float()
    # if el1 != el2:
        # print (el1,el2)

    #vis code
    if cfg.TRAIN.MAKE_IMAGES and np.amax(masks_int32) > 0:
        resolution= cfg.MRCNN.RESOLUTION
        print('Have Maxes')
        for roi in range(n_rois):
            numpy_arr = masks_gt.cpu()[roi].numpy()
            ind =0
            for clas in range(7):
                if weight[roi][resolution*resolution*clas+5].item() ==1:
                    ind = clas
            print('Array Copied')
            if np.amax(numpy_arr) != 1:
                print('continuing')
                continue
            for i in range(ind,ind+1):
                im_numpy = np.zeros((resolution,resolution,3))
                im_numpy2 =np.zeros((resolution,resolution,3))
                for x in range(resolution):
                    for y in range(resolution):
                        im_numpy[x,y,:] = masks_gt[roi][i*resolution*resolution+x*resolution+y].item()
                        im_numpy2[x,y,:] = masks_pred[roi][i][x][y].item()


                print('Array Filled')

                boxes = np.array([[50,50,60,60,.99],[1,1,5,5,.99]])
                # im_numpy = np.swapaxes(im_numpy,2,1)
                # im_numpy = np.swapaxes(im_numpy,2,0)
                # im_numpy[im_numpy>0] = 100
                # im_numpy[im_numpy<=0] =0

                # print('LENGTH:', len(im_numpy),len(im_numpy[0]))
                if np.amax(im_numpy) !=0:
                    vis_utils.vis_one_image(
                        im_numpy,
                        str(roi)+'_'+str(i)+'gt_image',
                        'hmmm/',
                        boxes,
                        None,
                        None,
                        dataset=datasets.get_particle_dataset(),
                        box_alpha=0.3,
                        show_class=True,
                        thresh=0.7,
                        kp_thresh=2,
                        plain_img=True
                    )
                    vis_utils.vis_one_image(
                        im_numpy2,
                        str(roi)+'_'+str(i)+'pred_image',
                        'hmmm/',
                        boxes,
                        None,
                        None,
                        dataset=datasets.get_particle_dataset(),
                        box_alpha=0.3,
                        show_class=True,
                        thresh=0.7,
                        kp_thresh=2,
                        plain_img=True
                    )







      # masks_int32 {1, 0, -1}, -1 means ignore
    total_for_avg = weight.sum()

    num_on  = torch.sum(masks_gt>0).item()
    num_off = torch.sum(masks_gt==0).item()
    num_inv = torch.sum(masks_gt==-1).item()
    total_num = num_on+num_off
    dim1, dim2 = masks_gt.shape
    # print('on: ', num_on, " off: ", num_off, " inv: ",num_inv)
    # print(dim1*dim2 , ", ", num_on+num_off+num_inv)
    if num_off==0:
        num_off=1
    if num_on ==0:
        num_on=1

    weight = weight  * ( (1-masks_gt)*total_num/num_off + (masks_gt)*total_num/num_on )


    # print()
    # # print('masks_gt is type: ', type(masks_gt))
    # print('masks_pred shape is: ', masks_pred.shape)
    # print('max', torch.max(masks_pred).item())
    # print('min', torch.min(masks_pred).item())
    # print()

    # print()
    # # print('masks_gt is type: ', type(masks_gt))
    # print('masks_pred.view(n_rois, -1) shape is: ', masks_pred.view(n_rois, -1).shape)
    # print('max', torch.max(masks_pred.view(n_rois, -1)).item())
    # print('min', torch.min(masks_pred.view(n_rois, -1)).item())
    # print()

    # print()
    # print('masks_gt is type: ', type(masks_gt))
    # print('masks_gt shape is: ', masks_gt.shape)
    # print('max', torch.max(masks_gt).item())
    # print('min', torch.min(masks_gt).item())
    # print()
    #
    # print()
    # print('weight is type: ', type(weight))
    # print('weight shape is: ', weight.shape)
    # print('max', torch.max(weight).item())
    # print('min', torch.min(weight).item())
    # print()

    # print('Pred Shape: ', masks_pred.view(n_rois, -1).shape, "   Truth Shape: ", masks_gt.shape)
    # print()
    # print()
    # print()
    loss = F.binary_cross_entropy_with_logits(
        masks_pred.view(n_rois, -1), masks_gt, weight, size_average=False)
    # print()
    # print('loss is type: ', type(loss))
    # print('loss shape is: ', loss.shape)
    # print()
    loss /= total_for_avg
    return loss * cfg.MRCNN.WEIGHT_LOSS_MASK


# ---------------------------------------------------------------------------- #
# Mask heads
# ---------------------------------------------------------------------------- #

def mask_rcnn_fcn_head_v1up4convs(dim_in, roi_xform_func, spatial_scale):
    """v1up design: 4 * (conv 3x3), convT 2x2."""
    return mask_rcnn_fcn_head_v1upXconvs(
        dim_in, roi_xform_func, spatial_scale, 4
    )


def mask_rcnn_fcn_head_v1up4convs_gn(dim_in, roi_xform_func, spatial_scale):
    """v1up design: 4 * (conv 3x3), convT 2x2, with GroupNorm"""
    return mask_rcnn_fcn_head_v1upXconvs_gn(
        dim_in, roi_xform_func, spatial_scale, 4
    )


def mask_rcnn_fcn_head_v1up(dim_in, roi_xform_func, spatial_scale):
    """v1up design: 2 * (conv 3x3), convT 2x2."""
    return mask_rcnn_fcn_head_v1upXconvs(
        dim_in, roi_xform_func, spatial_scale, 2
    )


class mask_rcnn_fcn_head_v1upXconvs(nn.Module):
    """v1upXconvs design: X * (conv 3x3), convT 2x2."""
    def __init__(self, dim_in, roi_xform_func, spatial_scale, num_convs):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.num_convs = num_convs

        dilation = cfg.MRCNN.DILATION
        dim_inner = cfg.MRCNN.DIM_REDUCED
        self.dim_out = dim_inner

        module_list = []
        for i in range(num_convs):
            module_list.extend([
                nn.Conv2d(dim_in, dim_inner, 3, 1, padding=1*dilation, dilation=dilation),
                nn.ReLU(inplace=True)
            ])
            dim_in = dim_inner
        self.conv_fcn = nn.Sequential(*module_list)

        # upsample layer
        self.upconv = nn.ConvTranspose2d(dim_inner, dim_inner, 2, 2, 0)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if cfg.MRCNN.CONV_INIT == 'GaussianFill':
                init.normal_(m.weight, std=0.001)
            elif cfg.MRCNN.CONV_INIT == 'MSRAFill':
                mynn.init.MSRAFill(m.weight)
            else:
                raise ValueError
            init.constant_(m.bias, 0)

    def detectron_weight_mapping(self):
        mapping_to_detectron = {}
        for i in range(self.num_convs):
            mapping_to_detectron.update({
                'conv_fcn.%d.weight' % (2*i): '_[mask]_fcn%d_w' % (i+1),
                'conv_fcn.%d.bias' % (2*i): '_[mask]_fcn%d_b' % (i+1)
            })
        mapping_to_detectron.update({
            'upconv.weight': 'conv5_mask_w',
            'upconv.bias': 'conv5_mask_b'
        })

        return mapping_to_detectron, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='mask_rois',
            method=cfg.MRCNN.ROI_XFORM_METHOD,
            resolution=cfg.MRCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.MRCNN.ROI_XFORM_SAMPLING_RATIO
        )
        x = self.conv_fcn(x)
        return F.relu(self.upconv(x), inplace=True)


class mask_rcnn_fcn_head_v1upXconvs_gn(nn.Module):
    """v1upXconvs design: X * (conv 3x3), convT 2x2, with GroupNorm"""
    def __init__(self, dim_in, roi_xform_func, spatial_scale, num_convs):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.num_convs = num_convs

        dilation = cfg.MRCNN.DILATION
        dim_inner = cfg.MRCNN.DIM_REDUCED
        self.dim_out = dim_inner

        module_list = []
        for i in range(num_convs):
            module_list.extend([
                nn.Conv2d(dim_in, dim_inner, 3, 1, padding=1*dilation, dilation=dilation, bias=False),
                nn.GroupNorm(net_utils.get_group_gn(dim_inner), dim_inner, eps=cfg.GROUP_NORM.EPSILON),
                nn.ReLU(inplace=True)
            ])
            dim_in = dim_inner
        self.conv_fcn = nn.Sequential(*module_list)

        # upsample layer
        self.upconv = nn.ConvTranspose2d(dim_inner, dim_inner, 2, 2, 0)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if cfg.MRCNN.CONV_INIT == 'GaussianFill':
                init.normal_(m.weight, std=0.001)
            elif cfg.MRCNN.CONV_INIT == 'MSRAFill':
                mynn.init.MSRAFill(m.weight)
            else:
                raise ValueError
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def detectron_weight_mapping(self):
        mapping_to_detectron = {}
        for i in range(self.num_convs):
            mapping_to_detectron.update({
                'conv_fcn.%d.weight' % (3*i): '_mask_fcn%d_w' % (i+1),
                'conv_fcn.%d.weight' % (3*i+1): '_mask_fcn%d_gn_s' % (i+1),
                'conv_fcn.%d.bias' % (3*i+1): '_mask_fcn%d_gn_b' % (i+1)
            })
        mapping_to_detectron.update({
            'upconv.weight': 'conv5_mask_w',
            'upconv.bias': 'conv5_mask_b'
        })

        return mapping_to_detectron, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='mask_rois',
            method=cfg.MRCNN.ROI_XFORM_METHOD,
            resolution=cfg.MRCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.MRCNN.ROI_XFORM_SAMPLING_RATIO
        )
        x = self.conv_fcn(x)
        return F.relu(self.upconv(x), inplace=True)


class mask_rcnn_fcn_head_v0upshare(nn.Module):
    """Use a ResNet "conv5" / "stage5" head for mask prediction. Weights and
    computation are shared with the conv5 box head. Computation can only be
    shared during training, since inference is cascaded.

    v0upshare design: conv5, convT 2x2.
    """
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.dim_out = cfg.MRCNN.DIM_REDUCED
        self.SHARE_RES5 = True
        assert cfg.MODEL.SHARE_RES5

        self.res5 = None  # will be assigned later
        dim_conv5 = 2048
        self.upconv5 = nn.ConvTranspose2d(dim_conv5, self.dim_out, 2, 2, 0)

        self._init_weights()

    def _init_weights(self):
        if cfg.MRCNN.CONV_INIT == 'GaussianFill':
            init.normal_(self.upconv5.weight, std=0.001)
        elif cfg.MRCNN.CONV_INIT == 'MSRAFill':
            mynn.init.MSRAFill(self.upconv5.weight)
        init.constant_(self.upconv5.bias, 0)

    def share_res5_module(self, res5_target):
        """ Share res5 block with box head on training """
        self.res5 = res5_target

    def detectron_weight_mapping(self):
        detectron_weight_mapping, orphan_in_detectron = \
          ResNet.residual_stage_detectron_mapping(self.res5, 'res5', 3, 5)
        # Assign None for res5 modules, do not load from or save to checkpoint
        for k in detectron_weight_mapping:
            detectron_weight_mapping[k] = None

        detectron_weight_mapping.update({
            'upconv5.weight': 'conv5_mask_w',
            'upconv5.bias': 'conv5_mask_b'
        })
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x, rpn_ret, roi_has_mask_int32=None):
        # print('Then I am here!')
        if self.training:
            # On training, we share the res5 computation with bbox head, so it's necessary to
            # sample 'useful' batches from the input x (res5_2_sum). 'Useful' means that the
            # batch (roi) has corresponding mask groundtruth, namely having positive values in
            # roi_has_mask_int32.
            inds = np.nonzero(roi_has_mask_int32 > 0)[0]
            inds = Variable(torch.from_numpy(inds)).cuda(x.get_device())
            # print("feat, before upconv",x.shape)
            x = x[inds]
            # print("feat, upconv: ", x.shape)
        else:

            # On testing, the computation is not shared with bbox head. This time input `x`
            # is the output features from the backbone network
            x = self.roi_xform(
                x, rpn_ret,
                blob_rois='mask_rois',
                method=cfg.MRCNN.ROI_XFORM_METHOD,
                resolution=cfg.MRCNN.ROI_XFORM_RESOLUTION,
                spatial_scale=self.spatial_scale,
                sampling_ratio=cfg.MRCNN.ROI_XFORM_SAMPLING_RATIO
            )
            x = self.res5(x)
        x = self.upconv5(x)

        x = F.relu(x, inplace=True)
        return x


class mask_rcnn_fcn_head_v0up(nn.Module):
    """v0up design: conv5, deconv 2x2 (no weight sharing with the box head)."""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.dim_out = cfg.MRCNN.DIM_REDUCED

        self.res5, dim_out = ResNet_roi_conv5_head_for_masks(dim_in)
        self.upconv5 = nn.ConvTranspose2d(dim_out, self.dim_out, 2, 2, 0)

        # Freeze all bn (affine) layers in resnet!!!
        self.res5.apply(
            lambda m: ResNet.freeze_params(m)
            if isinstance(m, mynn.AffineChannel2d) else None)
        self._init_weights()

    def _init_weights(self):
        if cfg.MRCNN.CONV_INIT == 'GaussianFill':
            init.normal_(self.upconv5.weight, std=0.001)
        elif cfg.MRCNN.CONV_INIT == 'MSRAFill':
            mynn.init.MSRAFill(self.upconv5.weight)
        init.constant_(self.upconv5.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping, orphan_in_detectron = \
          ResNet.residual_stage_detectron_mapping(self.res5, 'res5', 3, 5)
        detectron_weight_mapping.update({
            'upconv5.weight': 'conv5_mask_w',
            'upconv5.bias': 'conv5_mask_b'
        })
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='mask_rois',
            method=cfg.MRCNN.ROI_XFORM_METHOD,
            resolution=cfg.MRCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.MRCNN.ROI_XFORM_SAMPLING_RATIO
        )
        x = self.res5(x)
        # print(x.size()) e.g. (128, 2048, 7, 7)
        x = self.upconv5(x)
        x = F.relu(x, inplace=True)
        return x


def ResNet_roi_conv5_head_for_masks(dim_in):
    """ResNet "conv5" / "stage5" head for predicting masks."""
    dilation = cfg.MRCNN.DILATION
    stride_init = cfg.MRCNN.ROI_XFORM_RESOLUTION // 7  # by default: 2
    module, dim_out = ResNet.add_stage(dim_in, 2048, 512, 3, dilation, stride_init)
    return module, dim_out

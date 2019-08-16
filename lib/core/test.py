# Written by Roy Tseng
#
# Based on:
# --------------------------------------------------------
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import numpy as np
import pycocotools.mask as mask_util

from torch.autograd import Variable
import torch

from core.config import cfg
from utils.timer import Timer
import utils.boxes as box_utils
import utils.blob as blob_utils
import utils.fpn as fpn_utils
import utils.image as image_utils
import utils.keypoints as keypoint_utils

#to Vis
import datasets.dummy_datasets as datasets
import utils.vis as vis_utils
import cv2
from scipy import sparse

#to purity efficiency
from datasets.larcvdataset import IoU


def im_detect_all(model, im, box_proposals=None, timers=None, use_polygon=True):
    """Process the outputs of model for testing
    Args:
      model: the network module
      im_data: Pytorch variable. Input batch to the model.
      im_info: Pytorch variable. Input batch to the model.
      gt_boxes: Pytorch variable. Input batch to the model.
      num_boxes: Pytorch variable. Input batch to the model.
      args: arguments from command line.
      timer: record the cost of time for different steps
      use_polygon: should the output segmentation masks be sparse matrices or encoded polygons
    The rest of inputs are of type pytorch Variables and either input to or output from the model.
    """
    if timers is None:
        timers = defaultdict(Timer)

    timers['im_detect_bbox'].tic()
    if cfg.TEST.BBOX_AUG.ENABLED:
        scores, boxes, im_scale, blob_conv = im_detect_bbox_aug(
            model, im, box_proposals)
    else:
        scores, boxes, im_scale, blob_conv = im_detect_bbox(
            model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, box_proposals)
    timers['im_detect_bbox'].toc()


    # score and boxes are from the whole image after score thresholding and nms
    # (they are not separated by class) (numpy.ndarray)
    # cls_boxes boxes and scores are separated by class and in the format used
    # for evaluating results
    timers['misc_bbox'].tic()
    scores, boxes, cls_boxes = box_results_with_nms_and_limit(scores, boxes)

    timers['misc_bbox'].toc()

    if cfg.MODEL.MASK_ON and boxes.shape[0] > 0:
        timers['im_detect_mask'].tic()
        if cfg.TEST.MASK_AUG.ENABLED:
            masks = im_detect_mask_aug(model, im, boxes, im_scale, blob_conv)
        else:
            masks = im_detect_mask(model, im_scale, boxes, blob_conv)
        timers['im_detect_mask'].toc()

        timers['misc_mask'].tic()
        # print('Type boxes: ', type(boxes))
        # print('Shape boxes: ', boxes.shape)
        # print('boxes: ', boxes)
        # print('Type cls_boxes: ', type(cls_boxes))
        # print('len cls_boxes: ', len(cls_boxes))
        # print('len cls_boxes[1]: ', len(cls_boxes[1]))
        # print('type cls_boxes[1]: ', type(cls_boxes[1]))


        #
        # for index in range(len(cls_boxes)):
        #     for index2 in range(len(cls_boxes[index])):
        #         print(cls_boxes[index][index2])
        if use_polygon:
            cls_segms = segm_results(cls_boxes, masks, boxes, im.shape[0], im.shape[1])
        else:
            cls_segms, round_boxes = segm_results(cls_boxes, masks, boxes, im.shape[0], im.shape[1], polygon=use_polygon)
        timers['misc_mask'].toc()
    else:
        cls_segms = None

    if cfg.MODEL.KEYPOINTS_ON and boxes.shape[0] > 0:
        timers['im_detect_keypoints'].tic()
        if cfg.TEST.KPS_AUG.ENABLED:
            heatmaps = im_detect_keypoints_aug(model, im, boxes, im_scale, blob_conv)
        else:
            heatmaps = im_detect_keypoints(model, im_scale, boxes, blob_conv)
        timers['im_detect_keypoints'].toc()

        timers['misc_keypoints'].tic()
        cls_keyps = keypoint_results(cls_boxes, heatmaps, boxes)
        timers['misc_keypoints'].toc()
    else:
        cls_keyps = None
    if use_polygon:
        return cls_boxes, cls_segms, cls_keyps
    else:
        return cls_boxes, cls_segms, cls_keyps, round_boxes

def im_conv_body_only(model, im, target_scale, target_max_size):
    inputs, im_scale = _get_blobs(im, None, target_scale, target_max_size)

    if cfg.PYTORCH_VERSION_LESS_THAN_040:
        inputs['data'] = Variable(torch.from_numpy(inputs['data']), volatile=True).cuda()
    else:
        inputs['data'] = torch.from_numpy(inputs['data']).cuda()
    inputs.pop('im_info')

    blob_conv = model.module.convbody_net(**inputs)

    return blob_conv, im_scale


def im_detect_bbox(model, im, target_scale, target_max_size, boxes=None):
    """Prepare the bbox for testing"""
    inputs, im_scale = _get_blobs(im, boxes, target_scale, target_max_size)
    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(inputs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(
            hashes, return_index=True, return_inverse=True
        )
        inputs['rois'] = inputs['rois'][index, :]
        boxes = boxes[index, :]

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS and not cfg.MODEL.FASTER_RCNN:
        _add_multilevel_rois_for_test(inputs, 'rois')

    if cfg.PYTORCH_VERSION_LESS_THAN_040:
        inputs['data'] = [Variable(torch.from_numpy(inputs['data']), volatile=True)]
        inputs['im_info'] = [Variable(torch.from_numpy(inputs['im_info']), volatile=True)]
    else:
        inputs['data'] = [torch.from_numpy(inputs['data'])]
        inputs['im_info'] = [torch.from_numpy(inputs['im_info'])]

    # torch.set_printoptions(profile="full")
    # torch.set_printoptions(precision=2)
    # print('start')
    # print(inputs['data'][0][0,0,0:510,25:30])
    # print('stop')

    # print()
    # print("Before Return Dict")
    # print('------------------')
    # print()
    return_dict = model(**inputs)

    if cfg.MODEL.FASTER_RCNN:
        rois = return_dict['rois'].data.cpu().numpy()

        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scale

    # cls prob (activations after softmax)
    scores = return_dict['cls_score'].data.cpu().numpy().squeeze()
    # In case there is 1 proposal
    scores = scores.reshape([-1, scores.shape[-1]])

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = return_dict['bbox_pred'].data.cpu().numpy().squeeze()
        # In case there is 1 proposal
        box_deltas = box_deltas.reshape([-1, box_deltas.shape[-1]])
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            # Remove predictions for bg class (compat with MSRA code)
            box_deltas = box_deltas[:, -4:]
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # (legacy) Optionally normalize targets by a precomputed mean and stdev
            box_deltas = box_deltas.view(-1, 4) * cfg.TRAIN.BBOX_NORMALIZE_STDS \
                         + cfg.TRAIN.BBOX_NORMALIZE_MEANS
        pred_boxes = box_utils.bbox_transform(boxes, box_deltas, cfg.MODEL.BBOX_REG_WEIGHTS)
        pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, im.shape)
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            pred_boxes = np.tile(pred_boxes, (1, scores.shape[1]))
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, pred_boxes, im_scale, return_dict['blob_conv']


def im_detect_bbox_aug(model, im, box_proposals=None):
    """Performs bbox detection with test-time augmentations.
    Function signature is the same as for im_detect_bbox.
    """
    assert not cfg.TEST.BBOX_AUG.SCALE_SIZE_DEP, \
        'Size dependent scaling not implemented'
    assert not cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION' or \
        cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION', \
        'Coord heuristic must be union whenever score heuristic is union'
    assert not cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION' or \
        cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION', \
        'Score heuristic must be union whenever coord heuristic is union'
    assert not cfg.MODEL.FASTER_RCNN or \
        cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION', \
        'Union heuristic must be used to combine Faster RCNN predictions'

    # Collect detections computed under different transformations
    scores_ts = []
    boxes_ts = []

    def add_preds_t(scores_t, boxes_t):
        scores_ts.append(scores_t)
        boxes_ts.append(boxes_t)

    # Perform detection on the horizontally flipped image
    if cfg.TEST.BBOX_AUG.H_FLIP:
        scores_hf, boxes_hf, _ = im_detect_bbox_hflip(
            model,
            im,
            cfg.TEST.SCALE,
            cfg.TEST.MAX_SIZE,
            box_proposals=box_proposals
        )
        add_preds_t(scores_hf, boxes_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.BBOX_AUG.SCALES:
        max_size = cfg.TEST.BBOX_AUG.MAX_SIZE
        scores_scl, boxes_scl = im_detect_bbox_scale(
            model, im, scale, max_size, box_proposals
        )
        add_preds_t(scores_scl, boxes_scl)

        if cfg.TEST.BBOX_AUG.SCALE_H_FLIP:
            scores_scl_hf, boxes_scl_hf = im_detect_bbox_scale(
                model, im, scale, max_size, box_proposals, hflip=True
            )
            add_preds_t(scores_scl_hf, boxes_scl_hf)

    # Perform detection at different aspect ratios
    for aspect_ratio in cfg.TEST.BBOX_AUG.ASPECT_RATIOS:
        scores_ar, boxes_ar = im_detect_bbox_aspect_ratio(
            model, im, aspect_ratio, box_proposals
        )
        add_preds_t(scores_ar, boxes_ar)

        if cfg.TEST.BBOX_AUG.ASPECT_RATIO_H_FLIP:
            scores_ar_hf, boxes_ar_hf = im_detect_bbox_aspect_ratio(
                model, im, aspect_ratio, box_proposals, hflip=True
            )
            add_preds_t(scores_ar_hf, boxes_ar_hf)

    # Compute detections for the original image (identity transform) last to
    # ensure that the Caffe2 workspace is populated with blobs corresponding
    # to the original image on return (postcondition of im_detect_bbox)
    scores_i, boxes_i, im_scale_i, blob_conv_i = im_detect_bbox(
        model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes=box_proposals
    )
    add_preds_t(scores_i, boxes_i)

    # Combine the predicted scores
    if cfg.TEST.BBOX_AUG.SCORE_HEUR == 'ID':
        scores_c = scores_i
    elif cfg.TEST.BBOX_AUG.SCORE_HEUR == 'AVG':
        scores_c = np.mean(scores_ts, axis=0)
    elif cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION':
        scores_c = np.vstack(scores_ts)
    else:
        raise NotImplementedError(
            'Score heur {} not supported'.format(cfg.TEST.BBOX_AUG.SCORE_HEUR)
        )

    # Combine the predicted boxes
    if cfg.TEST.BBOX_AUG.COORD_HEUR == 'ID':
        boxes_c = boxes_i
    elif cfg.TEST.BBOX_AUG.COORD_HEUR == 'AVG':
        boxes_c = np.mean(boxes_ts, axis=0)
    elif cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION':
        boxes_c = np.vstack(boxes_ts)
    else:
        raise NotImplementedError(
            'Coord heur {} not supported'.format(cfg.TEST.BBOX_AUG.COORD_HEUR)
        )

    return scores_c, boxes_c, im_scale_i, blob_conv_i


def im_detect_bbox_hflip(
        model, im, target_scale, target_max_size, box_proposals=None):
    """Performs bbox detection on the horizontally flipped image.
    Function signature is the same as for im_detect_bbox.
    """
    # Compute predictions on the flipped image
    im_hf = im[:, ::-1, :]
    im_width = im.shape[1]

    if not cfg.MODEL.FASTER_RCNN:
        box_proposals_hf = box_utils.flip_boxes(box_proposals, im_width)
    else:
        box_proposals_hf = None

    scores_hf, boxes_hf, im_scale, _ = im_detect_bbox(
        model, im_hf, target_scale, target_max_size, boxes=box_proposals_hf
    )

    # Invert the detections computed on the flipped image
    boxes_inv = box_utils.flip_boxes(boxes_hf, im_width)

    return scores_hf, boxes_inv, im_scale


def im_detect_bbox_scale(
        model, im, target_scale, target_max_size, box_proposals=None, hflip=False):
    """Computes bbox detections at the given scale.
    Returns predictions in the original image space.
    """
    if hflip:
        scores_scl, boxes_scl, _ = im_detect_bbox_hflip(
            model, im, target_scale, target_max_size, box_proposals=box_proposals
        )
    else:
        scores_scl, boxes_scl, _, _ = im_detect_bbox(
            model, im, target_scale, target_max_size, boxes=box_proposals
        )
    return scores_scl, boxes_scl


def im_detect_bbox_aspect_ratio(
        model, im, aspect_ratio, box_proposals=None, hflip=False):
    """Computes bbox detections at the given width-relative aspect ratio.
    Returns predictions in the original image space.
    """
    # Compute predictions on the transformed image
    im_ar = image_utils.aspect_ratio_rel(im, aspect_ratio)

    if not cfg.MODEL.FASTER_RCNN:
        box_proposals_ar = box_utils.aspect_ratio(box_proposals, aspect_ratio)
    else:
        box_proposals_ar = None

    if hflip:
        scores_ar, boxes_ar, _ = im_detect_bbox_hflip(
            model,
            im_ar,
            cfg.TEST.SCALE,
            cfg.TEST.MAX_SIZE,
            box_proposals=box_proposals_ar
        )
    else:
        scores_ar, boxes_ar, _, _ = im_detect_bbox(
            model,
            im_ar,
            cfg.TEST.SCALE,
            cfg.TEST.MAX_SIZE,
            boxes=box_proposals_ar
        )

    # Invert the detected boxes
    boxes_inv = box_utils.aspect_ratio(boxes_ar, 1.0 / aspect_ratio)

    return scores_ar, boxes_inv


def im_detect_mask(model, im_scale, boxes, blob_conv):
    """Infer instance segmentation masks. This function must be called after
    im_detect_bbox as it assumes that the Caffe2 workspace is already populated
    with the necessary blobs.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scale (list): image blob scales as returned by im_detect_bbox
        boxes (ndarray): R x 4 array of bounding box detections (e.g., as
            returned by im_detect_bbox)
        blob_conv (Variable): base features from the backbone network.

    Returns:
        pred_masks (ndarray): R x K x M x M array of class specific soft masks
            output by the network (must be processed by segm_results to convert
            into hard masks in the original image coordinate space)
    """
    M = cfg.MRCNN.RESOLUTION
    if boxes.shape[0] == 0:
        pred_masks = np.zeros((0, M, M), np.float32)
        return pred_masks

    inputs = {'mask_rois': _get_rois_blob(boxes, im_scale)}

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois_for_test(inputs, 'mask_rois')

    pred_masks = model.module.mask_net(blob_conv, inputs)
    pred_masks = pred_masks.data.cpu().numpy().squeeze()

    if cfg.MRCNN.CLS_SPECIFIC_MASK:
        pred_masks = pred_masks.reshape([-1, cfg.MODEL.NUM_CLASSES, M, M])
    else:
        pred_masks = pred_masks.reshape([-1, 1, M, M])

    # n_rois,_,__,___ = pred_masks.shape
    # print('pred_masks_shape', pred_masks.shape)
    # print('Max', np.amax(pred_masks))
    # for roi in range(n_rois):
    #     print('Array Copied')
    #     if np.amax(pred_masks) == 0:
    #         print('continuing')
    #         continue
    #     for i in range(0,2):
    #         im_numpy2 =np.zeros((448,448,3))
    #         for x in range(448):
    #             for y in range(448):
    #                 im_numpy2[x,y,:] = pred_masks[roi][i][x][y]*250
    #                 # if pred_masks[roi][i][x][y] != 0 and pred_masks[roi][i][x][y] != 1 :
    #                 #     print(pred_masks[roi][i][x][y])
    #
    #             # im_numpy2[im_numpy2>0.1] = 100
    #             # im_numpy2[im_numpy2<=0.1] =0
    #         print('Array Filled')
    #
    #         boxes = np.array([[50,50,60,60,.99],[1,1,5,5,.99]])
    #
    #
    #         vis_utils.vis_one_image(
    #             im_numpy2,
    #             str(roi)+'_'+str(i)+'deploy_pred_image_ad250_',
    #             'hmmm/',
    #             boxes,
    #             None,
    #             None,
    #             dataset=datasets.get_particle_dataset(),
    #             box_alpha=0.3,
    #             show_class=True,
    #             thresh=0.7,
    #             kp_thresh=2,
    #             plain_img=True
    #         )
    # print('no sir!')
    return pred_masks


def im_detect_mask_aug(model, im, boxes, im_scale, blob_conv):
    """Performs mask detection with test-time augmentations.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im (ndarray): BGR image to test
        boxes (ndarray): R x 4 array of bounding boxes
        im_scale (list): image blob scales as returned by im_detect_bbox
        blob_conv (Tensor): base features from the backbone network.

    Returns:
        masks (ndarray): R x K x M x M array of class specific soft masks
    """
    assert not cfg.TEST.MASK_AUG.SCALE_SIZE_DEP, \
        'Size dependent scaling not implemented'

    # Collect masks computed under different transformations
    masks_ts = []

    # Compute masks for the original image (identity transform)
    masks_i = im_detect_mask(model, im_scale, boxes, blob_conv)
    masks_ts.append(masks_i)

    # Perform mask detection on the horizontally flipped image
    if cfg.TEST.MASK_AUG.H_FLIP:
        masks_hf = im_detect_mask_hflip(
            model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes
        )
        masks_ts.append(masks_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.MASK_AUG.SCALES:
        max_size = cfg.TEST.MASK_AUG.MAX_SIZE
        masks_scl = im_detect_mask_scale(model, im, scale, max_size, boxes)
        masks_ts.append(masks_scl)

        if cfg.TEST.MASK_AUG.SCALE_H_FLIP:
            masks_scl_hf = im_detect_mask_scale(
                model, im, scale, max_size, boxes, hflip=True
            )
            masks_ts.append(masks_scl_hf)

    # Compute masks at different aspect ratios
    for aspect_ratio in cfg.TEST.MASK_AUG.ASPECT_RATIOS:
        masks_ar = im_detect_mask_aspect_ratio(model, im, aspect_ratio, boxes)
        masks_ts.append(masks_ar)

        if cfg.TEST.MASK_AUG.ASPECT_RATIO_H_FLIP:
            masks_ar_hf = im_detect_mask_aspect_ratio(
                model, im, aspect_ratio, boxes, hflip=True
            )
            masks_ts.append(masks_ar_hf)

    # Combine the predicted soft masks
    if cfg.TEST.MASK_AUG.HEUR == 'SOFT_AVG':
        masks_c = np.mean(masks_ts, axis=0)
    elif cfg.TEST.MASK_AUG.HEUR == 'SOFT_MAX':
        masks_c = np.amax(masks_ts, axis=0)
    elif cfg.TEST.MASK_AUG.HEUR == 'LOGIT_AVG':

        def logit(y):
            return -1.0 * np.log((1.0 - y) / np.maximum(y, 1e-20))

        logit_masks = [logit(y) for y in masks_ts]
        logit_masks = np.mean(logit_masks, axis=0)
        masks_c = 1.0 / (1.0 + np.exp(-logit_masks))
    else:
        raise NotImplementedError(
            'Heuristic {} not supported'.format(cfg.TEST.MASK_AUG.HEUR)
        )

    return masks_c


def im_detect_mask_hflip(model, im, target_scale, target_max_size, boxes):
    """Performs mask detection on the horizontally flipped image.
    Function signature is the same as for im_detect_mask_aug.
    """
    # Compute the masks for the flipped image
    im_hf = im[:, ::-1, :]
    boxes_hf = box_utils.flip_boxes(boxes, im.shape[1])

    blob_conv, im_scale = im_conv_body_only(model, im_hf, target_scale, target_max_size)
    masks_hf = im_detect_mask(model, im_scale, boxes_hf, blob_conv)

    # Invert the predicted soft masks
    masks_inv = masks_hf[:, :, :, ::-1]

    return masks_inv


def im_detect_mask_scale(
        model, im, target_scale, target_max_size, boxes, hflip=False):
    """Computes masks at the given scale."""
    if hflip:
        masks_scl = im_detect_mask_hflip(
            model, im, target_scale, target_max_size, boxes
        )
    else:
        blob_conv, im_scale = im_conv_body_only(model, im, target_scale, target_max_size)
        masks_scl = im_detect_mask(model, im_scale, boxes, blob_conv)
    return masks_scl


def im_detect_mask_aspect_ratio(model, im, aspect_ratio, boxes, hflip=False):
    """Computes mask detections at the given width-relative aspect ratio."""

    # Perform mask detection on the transformed image
    im_ar = image_utils.aspect_ratio_rel(im, aspect_ratio)
    boxes_ar = box_utils.aspect_ratio(boxes, aspect_ratio)

    if hflip:
        masks_ar = im_detect_mask_hflip(
            model, im_ar, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes_ar
        )
    else:
        blob_conv, im_scale = im_conv_body_only(
            model, im_ar, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE
        )
        masks_ar = im_detect_mask(model, im_scale, boxes_ar, blob_conv)

    return masks_ar


def im_detect_keypoints(model, im_scale, boxes, blob_conv):
    """Infer instance keypoint poses. This function must be called after
    im_detect_bbox as it assumes that the Caffe2 workspace is already populated
    with the necessary blobs.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scale (list): image blob scales as returned by im_detect_bbox
        boxes (ndarray): R x 4 array of bounding box detections (e.g., as
            returned by im_detect_bbox)

    Returns:
        pred_heatmaps (ndarray): R x J x M x M array of keypoint location
            logits (softmax inputs) for each of the J keypoint types output
            by the network (must be processed by keypoint_results to convert
            into point predictions in the original image coordinate space)
    """
    M = cfg.KRCNN.HEATMAP_SIZE
    if boxes.shape[0] == 0:
        pred_heatmaps = np.zeros((0, cfg.KRCNN.NUM_KEYPOINTS, M, M), np.float32)
        return pred_heatmaps

    inputs = {'keypoint_rois': _get_rois_blob(boxes, im_scale)}

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois_for_test(inputs, 'keypoint_rois')

    pred_heatmaps = model.module.keypoint_net(blob_conv, inputs)
    pred_heatmaps = pred_heatmaps.data.cpu().numpy().squeeze()

    # In case of 1
    if pred_heatmaps.ndim == 3:
        pred_heatmaps = np.expand_dims(pred_heatmaps, axis=0)

    return pred_heatmaps


def im_detect_keypoints_aug(model, im, boxes, im_scale, blob_conv):
    """Computes keypoint predictions with test-time augmentations.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im (ndarray): BGR image to test
        boxes (ndarray): R x 4 array of bounding boxes
        im_scale (list): image blob scales as returned by im_detect_bbox
        blob_conv (Tensor): base features from the backbone network.

    Returns:
        heatmaps (ndarray): R x J x M x M array of keypoint location logits
    """
    # Collect heatmaps predicted under different transformations
    heatmaps_ts = []
    # Tag predictions computed under downscaling and upscaling transformations
    ds_ts = []
    us_ts = []

    def add_heatmaps_t(heatmaps_t, ds_t=False, us_t=False):
        heatmaps_ts.append(heatmaps_t)
        ds_ts.append(ds_t)
        us_ts.append(us_t)

    # Compute the heatmaps for the original image (identity transform)
    heatmaps_i = im_detect_keypoints(model, im_scale, boxes, blob_conv)
    add_heatmaps_t(heatmaps_i)

    # Perform keypoints detection on the horizontally flipped image
    if cfg.TEST.KPS_AUG.H_FLIP:
        heatmaps_hf = im_detect_keypoints_hflip(
            model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes
        )
        add_heatmaps_t(heatmaps_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.KPS_AUG.SCALES:
        ds_scl = scale < cfg.TEST.SCALE
        us_scl = scale > cfg.TEST.SCALE
        heatmaps_scl = im_detect_keypoints_scale(
            model, im, scale, cfg.TEST.KPS_AUG.MAX_SIZE, boxes
        )
        add_heatmaps_t(heatmaps_scl, ds_scl, us_scl)

        if cfg.TEST.KPS_AUG.SCALE_H_FLIP:
            heatmaps_scl_hf = im_detect_keypoints_scale(
                model, im, scale, cfg.TEST.KPS_AUG.MAX_SIZE, boxes, hflip=True
            )
            add_heatmaps_t(heatmaps_scl_hf, ds_scl, us_scl)

    # Compute keypoints at different aspect ratios
    for aspect_ratio in cfg.TEST.KPS_AUG.ASPECT_RATIOS:
        heatmaps_ar = im_detect_keypoints_aspect_ratio(
            model, im, aspect_ratio, boxes
        )
        add_heatmaps_t(heatmaps_ar)

        if cfg.TEST.KPS_AUG.ASPECT_RATIO_H_FLIP:
            heatmaps_ar_hf = im_detect_keypoints_aspect_ratio(
                model, im, aspect_ratio, boxes, hflip=True
            )
            add_heatmaps_t(heatmaps_ar_hf)

    # Select the heuristic function for combining the heatmaps
    if cfg.TEST.KPS_AUG.HEUR == 'HM_AVG':
        np_f = np.mean
    elif cfg.TEST.KPS_AUG.HEUR == 'HM_MAX':
        np_f = np.amax
    else:
        raise NotImplementedError(
            'Heuristic {} not supported'.format(cfg.TEST.KPS_AUG.HEUR)
        )

    def heur_f(hms_ts):
        return np_f(hms_ts, axis=0)

    # Combine the heatmaps
    if cfg.TEST.KPS_AUG.SCALE_SIZE_DEP:
        heatmaps_c = combine_heatmaps_size_dep(
            heatmaps_ts, ds_ts, us_ts, boxes, heur_f
        )
    else:
        heatmaps_c = heur_f(heatmaps_ts)

    return heatmaps_c


def im_detect_keypoints_hflip(model, im, target_scale, target_max_size, boxes):
    """Computes keypoint predictions on the horizontally flipped image.
    Function signature is the same as for im_detect_keypoints_aug.
    """
    # Compute keypoints for the flipped image
    im_hf = im[:, ::-1, :]
    boxes_hf = box_utils.flip_boxes(boxes, im.shape[1])

    blob_conv, im_scale = im_conv_body_only(model, im_hf, target_scale, target_max_size)
    heatmaps_hf = im_detect_keypoints(model, im_scale, boxes_hf, blob_conv)

    # Invert the predicted keypoints
    heatmaps_inv = keypoint_utils.flip_heatmaps(heatmaps_hf)

    return heatmaps_inv


def im_detect_keypoints_scale(
    model, im, target_scale, target_max_size, boxes, hflip=False):
    """Computes keypoint predictions at the given scale."""
    if hflip:
        heatmaps_scl = im_detect_keypoints_hflip(
            model, im, target_scale, target_max_size, boxes
        )
    else:
        blob_conv, im_scale = im_conv_body_only(model, im, target_scale, target_max_size)
        heatmaps_scl = im_detect_keypoints(model, im_scale, boxes, blob_conv)
    return heatmaps_scl


def im_detect_keypoints_aspect_ratio(
    model, im, aspect_ratio, boxes, hflip=False):
    """Detects keypoints at the given width-relative aspect ratio."""

    # Perform keypoint detectionon the transformed image
    im_ar = image_utils.aspect_ratio_rel(im, aspect_ratio)
    boxes_ar = box_utils.aspect_ratio(boxes, aspect_ratio)

    if hflip:
        heatmaps_ar = im_detect_keypoints_hflip(
            model, im_ar, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes_ar
        )
    else:
        blob_conv, im_scale = im_conv_body_only(
            model, im_ar, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE
        )
        heatmaps_ar = im_detect_keypoints(model, im_scale, boxes_ar, blob_conv)

    return heatmaps_ar


def combine_heatmaps_size_dep(hms_ts, ds_ts, us_ts, boxes, heur_f):
    """Combines heatmaps while taking object sizes into account."""
    assert len(hms_ts) == len(ds_ts) and len(ds_ts) == len(us_ts), \
        'All sets of hms must be tagged with downscaling and upscaling flags'

    # Classify objects into small+medium and large based on their box areas
    areas = box_utils.boxes_area(boxes)
    sm_objs = areas < cfg.TEST.KPS_AUG.AREA_TH
    l_objs = areas >= cfg.TEST.KPS_AUG.AREA_TH

    # Combine heatmaps computed under different transformations for each object
    hms_c = np.zeros_like(hms_ts[0])

    for i in range(hms_c.shape[0]):
        hms_to_combine = []
        for hms_t, ds_t, us_t in zip(hms_ts, ds_ts, us_ts):
            # Discard downscaling predictions for small and medium objects
            if sm_objs[i] and ds_t:
                continue
            # Discard upscaling predictions for large objects
            if l_objs[i] and us_t:
                continue
            hms_to_combine.append(hms_t[i])
        hms_c[i] = heur_f(hms_to_combine)

    return hms_c


def box_results_with_nms_and_limit(scores, boxes):  # NOTE: support single-batch
    """Returns bounding-box detection results by thresholding on scores and
    applying non-maximum suppression (NMS).

    `boxes` has shape (#detections, 4 * #classes), where each row represents
    a list of predicted bounding boxes for each of the object classes in the
    dataset (including the background class). The detections in each row
    originate from the same object proposal.

    `scores` has shape (#detection, #classes), where each row represents a list
    of object detection confidence scores for each of the object classes in the
    dataset (including the background class). `scores[i, j]`` corresponds to the
    box at `boxes[i, j * 4:(j + 1) * 4]`.
    """
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_boxes = [[] for _ in range(num_classes)]
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > cfg.TEST.SCORE_THRESH)[0]
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 4:(j + 1) * 4]
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(np.float32, copy=False)
        if cfg.TEST.SOFT_NMS.ENABLED:
            nms_dets, _ = box_utils.soft_nms(
                dets_j,
                sigma=cfg.TEST.SOFT_NMS.SIGMA,
                overlap_thresh=cfg.TEST.NMS,
                score_thresh=0.0001,
                method=cfg.TEST.SOFT_NMS.METHOD
            )
        else:
            # print()
            # print(j)
            # print("About to use TEST.NMS")
            # print('Num Proposals First', (dets_j.shape))
            keep = box_utils.nms(dets_j, cfg.TEST.NMS)



            nms_dets = dets_j[keep, :]
            # print()
            # print('Num Proposals After', (nms_dets.shape))
            # print()
            # print('dets_j type', type(dets_j))
            # print('dets_j shape', (dets_j.shape))
            # print(dets_j)
            # print()
            # print('nms_dets type', type(nms_dets))
            # print('nms_dets shape', (nms_dets.shape))
            # print(nms_dets)

        # Refine the post-NMS boxes using bounding-box voting
        if cfg.TEST.BBOX_VOTE.ENABLED:
            nms_dets = box_utils.box_voting(
                nms_dets,
                dets_j,
                cfg.TEST.BBOX_VOTE.VOTE_TH,
                scoring_method=cfg.TEST.BBOX_VOTE.SCORING_METHOD
            )
        cls_boxes[j] = nms_dets

    # Limit to max_per_image detections **over all classes**
    if cfg.TEST.DETECTIONS_PER_IM > 0:
        image_scores = np.hstack(
            [cls_boxes[j][:, -1] for j in range(1, num_classes)]
        )
        if len(image_scores) > cfg.TEST.DETECTIONS_PER_IM:
            image_thresh = np.sort(image_scores)[-cfg.TEST.DETECTIONS_PER_IM]
            for j in range(1, num_classes):
                keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]

    im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
    return scores, boxes, cls_boxes


def segm_results(cls_boxes, masks, ref_boxes, im_h, im_w, polygon=True):
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_segms = [[] for _ in range(num_classes)]
    round_boxes =[[] for _ in range(num_classes)]
    mask_ind = 0


    # To work around an issue with cv2.resize (it seems to automatically pad
    # with repeated border values), we manually zero-pad the masks by 1 pixel
    # prior to resizing back to the original image resolution. This prevents
    # "top hat" artifacts. We therefore need to expand the reference boxes by an
    # appropriate factor.
    M = cfg.MRCNN.RESOLUTION
    scale = (M + 2.0) / M
    ref_boxes = box_utils.expand_boxes(ref_boxes, scale)
    ref_boxes = ref_boxes.astype(np.int32)
    padded_mask = np.zeros((M + 2, M + 2), dtype=np.float32)
    # skip j = 0, because it's the background class
    for j in range(1, num_classes):
        segms = []
        rboxes= []
        for _ in range(cls_boxes[j].shape[0]):
            if cfg.MRCNN.CLS_SPECIFIC_MASK:
                padded_mask[1:-1, 1:-1] = masks[mask_ind, j, :, :]
            else:
                padded_mask[1:-1, 1:-1] = masks[mask_ind, 0, :, :]

            ref_box = ref_boxes[mask_ind, :]
            w = (ref_box[2] - ref_box[0] + 1)
            h = (ref_box[3] - ref_box[1] + 1)
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)

            mask = cv2.resize(padded_mask, (w, h))
            mask = np.array(mask > cfg.MRCNN.THRESH_BINARIZE, dtype=np.uint8)
            im_mask = np.zeros((im_h, im_w), dtype=np.uint8)

            x_0 = max(ref_box[0], 0)
            x_1 = min(ref_box[2] + 1, im_w)
            y_0 = max(ref_box[1], 0)
            y_1 = min(ref_box[3] + 1, im_h)



            # Get RLE encoding used by the COCO evaluation API
            if polygon:
                im_mask[y_0:y_1, x_0:x_1] = mask[
                    (y_0 - ref_box[1]):(y_1 - ref_box[1]), (x_0 - ref_box[0]):(x_1 - ref_box[0])]
                rle = mask_util.encode(np.array(im_mask[:, :, np.newaxis], order='F'))[0]
                # For dumping to json, need to decode the byte string.
                # https://github.com/cocodataset/cocoapi/issues/70
                rle['counts'] = rle['counts'].decode('ascii')
                segms.append(rle)
            else:
                sparse_mask = sparse.csr_matrix(
                    mask[
                        (y_0 - ref_box[1]):(y_1 - ref_box[1]),
                        (x_0 - ref_box[0]):(x_1 - ref_box[0])
                        ]
                        )
                segms.append(sparse_mask)
                rboxes.append(ref_box)
            mask_ind += 1

        cls_segms[j] = segms
        round_boxes[j] = rboxes
    assert mask_ind == masks.shape[0]
    if polygon:
        return cls_segms
    else:
        return cls_segms, round_boxes

def keypoint_results(cls_boxes, pred_heatmaps, ref_boxes):
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_keyps = [[] for _ in range(num_classes)]
    person_idx = keypoint_utils.get_person_class_index()
    xy_preds = keypoint_utils.heatmaps_to_keypoints(pred_heatmaps, ref_boxes)

    # NMS OKS
    if cfg.KRCNN.NMS_OKS:
        keep = keypoint_utils.nms_oks(xy_preds, ref_boxes, 0.3)
        xy_preds = xy_preds[keep, :, :]
        ref_boxes = ref_boxes[keep, :]
        pred_heatmaps = pred_heatmaps[keep, :, :, :]
        cls_boxes[person_idx] = cls_boxes[person_idx][keep, :]

    kps = [xy_preds[i] for i in range(xy_preds.shape[0])]
    cls_keyps[person_idx] = kps
    return cls_keyps


def _get_rois_blob(im_rois, im_scale):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid with columns
            [level, x1, y1, x2, y2]
    """
    rois, levels = _project_im_rois(im_rois, im_scale)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)


def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (ndarray): image pyramid levels used by each projected RoI
    """
    rois = im_rois.astype(np.float, copy=False) * scales
    levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)
    return rois, levels


def _add_multilevel_rois_for_test(blobs, name):
    """Distributes a set of RoIs across FPN pyramid levels by creating new level
    specific RoI blobs.

    Arguments:
        blobs (dict): dictionary of blobs
        name (str): a key in 'blobs' identifying the source RoI blob

    Returns:
        [by ref] blobs (dict): new keys named by `name + 'fpn' + level`
            are added to dict each with a value that's an R_level x 5 ndarray of
            RoIs (see _get_rois_blob for format)
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL
    lvls = fpn_utils.map_rois_to_fpn_levels(blobs[name][:, 1:5], lvl_min, lvl_max)
    fpn_utils.add_multilevel_roi_blobs(
        blobs, name, blobs[name], lvls, lvl_min, lvl_max
    )


def _get_blobs(im, rois, target_scale, target_max_size):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], im_scale, blobs['im_info'] = \
        blob_utils.get_image_blob(im, target_scale, target_max_size)
    if rois is not None:
        blobs['rois'] = _get_rois_blob(rois, im_scale)
    return blobs, im_scale

def efficiency_calculation_union(gt_box, gt_mask, pred_boxes, pred_masks, adc_image):
    ### Take in one ground truth box and its mask, then loop through all
    ### the predicted boxes adding up their coverages to find what % of GT is
    ### covered by their union
    ### also record the chosen's IoU with another GT mask
    percent_covered = 0

    #Get Union of pred_boxes
    union_pred = np.zeros(gt_mask.shape)
    for pred_index in range(pred_masks.shape[2]):
        pred_box = pred_boxes[pred_index,:]
        if not do_boxes_overlap(pred_box[0:4], gt_box[0:4]):
            #save some compute time, dont sum masks if boxes not overlapping
            continue
        union_pred = union_pred + pred_masks[:,:,pred_index]
    #make union a binary:
    union_pred[union_pred > 0]  = 1
    union_pred[union_pred <= 0] = 0


    gt_mask_box = gt_mask[int(np.floor(gt_box[1])):int(np.ceil(gt_box[3]+1)),int(np.floor(gt_box[0])):int(np.ceil(gt_box[2]+1))]
    union_pred_box = union_pred[int(np.floor(gt_box[1])):int(np.ceil(gt_box[3]+1)),int(np.floor(gt_box[0])):int(np.ceil(gt_box[2]+1))]
    adc_image_box = adc_image[int(np.floor(gt_box[1])):int(np.ceil(gt_box[3]+1)),int(np.floor(gt_box[0])):int(np.ceil(gt_box[2]+1))]

    pixels_covered   = np.sum(np.multiply(np.multiply(union_pred_box,gt_mask_box),adc_image_box))
    total_gt_pixels  = np.sum(np.multiply(gt_mask_box,adc_image_box))
    if total_gt_pixels == 0:
        #must be a dead region or not above adc threshold, let's not count this
        return -1

    percent_covered = float(pixels_covered)/float(total_gt_pixels)

    return percent_covered

def efficiency_calculation_single(gt_box, gt_mask, pred_boxes, pred_masks, adc_image):
    ### Take in one ground truth box and its mask, then loop through all
    ### the predicted boxes to find who has the most efficient coverage
    ### also record the chosen's IoU with another GT mask
    percent_covered = 0

    #Get Union of pred_boxes
    actual_pred = np.zeros(gt_mask.shape)
    for pred_index in range(pred_masks.shape[2]):
        tmp_percent_covered = 0.0
        pred_box = pred_boxes[pred_index,:]
        if not do_boxes_overlap(pred_box[0:4], gt_box[0:4]):
            #save some compute time, dont sum masks if boxes not overlapping
            continue
        actual_pred = actual_pred + pred_masks[:,:,pred_index]
        #make union a binary:
        actual_pred[actual_pred > 0]  = 1
        actual_pred[actual_pred <= 0] = 0


        gt_mask_box = gt_mask[int(np.floor(gt_box[1])):int(np.ceil(gt_box[3]+1)),int(np.floor(gt_box[0])):int(np.ceil(gt_box[2]+1))]
        actual_pred_box = actual_pred[int(np.floor(gt_box[1])):int(np.ceil(gt_box[3]+1)),int(np.floor(gt_box[0])):int(np.ceil(gt_box[2]+1))]
        adc_image_box = adc_image[int(np.floor(gt_box[1])):int(np.ceil(gt_box[3]+1)),int(np.floor(gt_box[0])):int(np.ceil(gt_box[2]+1))]

        pixels_covered   = np.sum(np.multiply(np.multiply(actual_pred_box,gt_mask_box),adc_image_box))
        total_gt_pixels  = np.sum(np.multiply(gt_mask_box,adc_image_box))
        if total_gt_pixels == 0:
            #must be a dead region or not above adc threshold, let's not count this
            return -1

        tmp_percent_covered = float(pixels_covered)/float(total_gt_pixels)
        if (tmp_percent_covered > percent_covered):
            # print(percent_covered, " Replaced with ", tmp_percent_covered)
            # print()
            percent_covered = tmp_percent_covered
        actual_pred.fill(0)
    return percent_covered

def best_iou(index, box_list):
    ### Take in a list of boxes of shape (N, 5) where the 5 represents
    ### X1,Y1,X2,Y2, score/class and a target index. Loop through all other
    ### indices in box_list to find the greatest IoU with the target index box
    assert index >= 0 and index < box_list.shape[0]
    best_iou =0
    target_box = box_list[index,:]

    for compare_index in range(box_list.shape[0]):
        if compare_index == index:
            continue
        this_iou = IoU(target_box[0:4], box_list[compare_index,0:4])
        if this_iou >= best_iou:
            best_iou = this_iou
    return best_iou


def purity_calculation(pred_box, pred_mask, gt_boxes, gt_masks, adc_image):
    ### Take in one prediction box and its mask, then loop through all
    ### the ground truth boxes figuring out which has the highest % coverage
    ### then also record the 2 highest IoU to ground truth to see how easily
    ### the prediction was to match to a GT
    ### it also uses an adc image to cut away pixels below some threshold in
    ### both the numerator and denominator
    percent_covered = 0

    # print()
    # print("pred_box.shape" , pred_box.shape)
    # print(pred_box)
    # print("pred_mask.shape" , pred_mask.shape)
    # print("gt_boxes.shape" , gt_boxes.shape)
    # print("gt_masks.shape" , gt_masks.shape)
    index_best_coverage = -1
    for gt_index in range(gt_boxes.shape[0]):
        gt_box = gt_boxes[gt_index,:]
        if not do_boxes_overlap(gt_box[0:4], pred_box[0:4]):
            #Boxes do not overlap, skip comparison of this GT to our pred
            # print("boxes not overlapping")
            continue
        # box_ints = [int(np.floor(pred_box[1])),int(np.ceil(pred_box[3]+1)),int(np.floor(pred_box[0])),int(np.ceil(pred_box[2]+1))]

        gt_mask = gt_masks[:,:,gt_index]
        gt_mask_box = gt_mask[int(np.floor(pred_box[1])):int(np.ceil(pred_box[3]+1)),int(np.floor(pred_box[0])):int(np.ceil(pred_box[2]+1))]
        pred_mask_box = pred_mask[int(np.floor(pred_box[1])):int(np.ceil(pred_box[3]+1)),int(np.floor(pred_box[0])):int(np.ceil(pred_box[2]+1))]
        adc_image_box = adc_image[int(np.floor(pred_box[1])):int(np.ceil(pred_box[3]+1)),int(np.floor(pred_box[0])):int(np.ceil(pred_box[2]+1))]

        overlap_count = np.sum(np.multiply(np.multiply(pred_mask_box,gt_mask_box),adc_image_box))
        pred_count = np.sum(np.multiply(pred_mask_box,adc_image_box))
        if pred_count ==0:
            this_percent=0
        else:
            this_percent = float(overlap_count)/float(pred_count)
        if this_percent > percent_covered:
            percent_covered = this_percent
            index_best_coverage = gt_index
        # print()
        # print("this_percent", this_percent)
        # print("numerator:" ,float(overlap_count))
        # print("denominator:", float(pred_count))
        # print()
    best_iou =0
    next_best_iou = 0
    index_best_iou =-1
    for gt_index in range(gt_boxes.shape[0]):
        this_iou = IoU(pred_box[0:4], gt_boxes[gt_index,0:4])
        if this_iou >= best_iou:
            next_best_iou = best_iou
            best_iou = this_iou
            index_best_iou =gt_index
        elif this_iou > next_best_iou:
            next_best_iou = this_iou

    if index_best_coverage == -1:
        same = -1
    elif (index_best_coverage ==index_best_iou):
        same = 1
    else:
        same = 0
    return percent_covered, best_iou, next_best_iou, same


def do_boxes_overlap(box1, box2):
    ### Take in two boxes in (x1,y1,x2,y2) format, and return whether they
    ### overlap

    #Check if box 1 is all right of box 2
    if box1[0] > box2[2]:
        return False
    #Check if box 1 all above box 2
    elif box1[1] > box2[3]+1:
        return False
    #Check if box 1 is all left of box 2
    elif box1[2]+1 < box2[0]:
        return False
    #Check if box 1 is all below box 2
    elif box1[3]+1 < box2[1]:
        return False
    #Eliminate the impossible, what remains is what is. They overlap
    else:
        return True

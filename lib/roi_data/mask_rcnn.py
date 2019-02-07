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
"""Construct minibatches for Mask R-CNN training. Handles the minibatch blobs
that are specific to Mask R-CNN. Other blobs that are generic to RPN or
Fast/er R-CNN are handled by their respecitive roi_data modules.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np

from core.config import cfg
import utils.blob as blob_utils
import utils.boxes as box_utils
import utils.segms as segm_utils

#larcvdataset original imports
import os,time
import ROOT
from larcv import larcv
import numpy as np
from torch.utils.data import Dataset
#new imports:
import cv2
import torchvision


def add_mask_rcnn_blobs(blobs, sampled_boxes, roidb, im_scale, batch_idx):
    """Add Mask R-CNN specific blobs to the input blob dictionary."""
    # Prepare the mask targets by associating one gt mask to each training roi
    # that has a fg (non-bg) class label.
    print("roidb['id']      :     ", roidb['id'])
    M = cfg.MRCNN.RESOLUTION
    polys_gt_inds = np.where((roidb['gt_classes'] > 0) &
                             (roidb['is_crowd'] == 0))[0]

    # input mask instead of polygon
    clustermask_cluster_crop_chain = roidb['chain_cluster']
    clustermask_cluster_crop_chain.GetEntry(roidb['id'])
    entry_clustermaskcluster_crop_data = clustermask_cluster_crop_chain.clustermask_masks_branch
    clustermaskcluster_crop_array = entry_clustermaskcluster_crop_data.as_vector()
    cluster_masks = clustermaskcluster_crop_array[roidb['plane']]
    masks_orig_size = []
    boxes_from_polys = np.empty((0,4))
    for i in polys_gt_inds:

        bbox = larcv.as_ndarray_bbox(cluster_masks[int(i)])
        # print(boxes_from_polys.shape)
        # print(np.array([bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]]).shape)
        boxes_from_polys = np.append(boxes_from_polys, np.array([[bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]]]), 0)
        # print(i)
        # print(bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3])
        masks_orig_size.append(larcv.as_ndarray_mask(cluster_masks[int(i)]))




    # polys_gt = [roidb['segms'][i] for i in polys_gt_inds]
    # for i in range(len(polys_gt)):
    #     # print("i is:", i)
    #     poly = polys_gt[i]
    #     if len(poly) ==0:
    #         print()
    #         print('Cheated, and made my own box')
    #         print()
    #         poly = [[0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0, 0.0]]
    #         polys_gt[i] = poly
    # print('Type Boxes: ', type(boxes_from_polys))
    # print('Shape Boxes: ', boxes_from_polys.shape)
    # print('Boxes: ', boxes_from_polys)
    # boxes_from_polys = segm_utils.polys_to_boxes(polys_gt)
    # print('Type Boxes: ', type(boxes_from_polys))
    #
    # print('Shape Boxes: ', boxes_from_polys.shape)
    # print('Boxes: ', boxes_from_polys)


    fg_inds = np.where(blobs['labels_int32'] > 0)[0]
    roi_has_mask = blobs['labels_int32'].copy()
    roi_has_mask[roi_has_mask > 0] = 1

    if fg_inds.shape[0] > 0:
        # Class labels for the foreground rois
        mask_class_labels = blobs['labels_int32'][fg_inds]
        # print('mask_class_labels', mask_class_labels)
        masks = blob_utils.zeros((fg_inds.shape[0], M**2), int32=True)
        # print('masks type', type(masks), masks.shape)
        # print('masks max:', masks.max())
        # print('masks min:', masks.min())
        # print()


        # Find overlap between all foreground rois and the bounding boxes
        # enclosing each segmentation
        rois_fg = sampled_boxes[fg_inds]
        overlaps_bbfg_bbpolys = box_utils.bbox_overlaps(
            rois_fg.astype(np.float32, copy=False),
            boxes_from_polys.astype(np.float32, copy=False))
        # Map from each fg rois to the index of the mask with highest overlap
        # (measured by bbox overlap)
        fg_polys_inds = np.argmax(overlaps_bbfg_bbpolys, axis=1)

        # original = np.array([[1,2,3], [4,5,6], [7,8,9]])
        # print('original[0][2]', original[0][2])
        # box_orig = np.array([7,7,9,9])
        # box_adjust = np.array([2,2,6,6])
        # resized = resize_mask_to_set_dim(original, box_adjust, box_orig, 10)
        #
        #
        # for y in reversed(range(resized.shape[1])):
        #     for x in range(resized.shape[0]):
        #         print(int(resized[x][y]),end=' ')
        #     print()
        # print()

        # add fg targets
        for i in range(rois_fg.shape[0]):
            fg_polys_ind = fg_polys_inds[i]
            # poly_gt = polys_gt[fg_polys_ind]
            mask_gt_orig_size = masks_orig_size[fg_polys_ind]
            box_gt = boxes_from_polys[fg_polys_ind]


            roi_fg = rois_fg[i]
            # Rasterize the portion of the polygon mask within the given fg roi
            # to an M x M binary image
            # print(fg_polys_ind)
            # print('roi_fg', roi_fg)
            # mask = segm_utils.polys_to_mask_wrt_box(poly_gt, roi_fg, M)
            mask = resize_mask_to_set_dim(mask_gt_orig_size, roi_fg, box_gt, M)

            mask = np.array(mask > 0, dtype=np.int32)  # Ensure it's binary
            masks[i, :] = np.reshape(mask, M**2)
    else:  # If there are no fg masks (it does happen)

        # The network cannot handle empty blobs, so we must provide a mask
        # We simply take the first bg roi, given it an all -1's mask (ignore
        # label), and label it with class zero (bg).
        bg_inds = np.where(blobs['labels_int32'] == 0)[0]
        # rois_fg is actually one background roi, but that's ok because ...
        rois_fg = sampled_boxes[bg_inds[0]].reshape((1, -1))
        # We give it an -1's blob (ignore label)
        masks = -blob_utils.ones((1, M**2), int32=True)
        # We label it with class = 0 (background)
        mask_class_labels = blob_utils.zeros((1, ))
        # Mark that the first roi has a mask
        roi_has_mask[0] = 1
    # print('Before Expansion')
    # for mask in range(len(masks)):
    #     if mask >5:
    #         break
    #     if np.amax(masks[mask]) > 0:
    #         for x in range(14):
    #             for y in range(14):
    #                 print(masks[mask][x*14+y], end=' ')
    #             print()
    #         print()
    if cfg.MRCNN.CLS_SPECIFIC_MASK:
        masks = _expand_to_class_specific_mask_targets(masks,
                                                       mask_class_labels)

    # Scale rois_fg and format as (batch_idx, x1, y1, x2, y2)
    rois_fg *= im_scale
    repeated_batch_idx = batch_idx * blob_utils.ones((rois_fg.shape[0], 1))
    rois_fg = np.hstack((repeated_batch_idx, rois_fg))

    # Update blobs dict with Mask R-CNN blobs
    blobs['mask_rois'] = rois_fg
    blobs['roi_has_mask_int32'] = roi_has_mask

    # print('masks type', type(masks), masks.shape)

    # for mask in range(len(masks)):
    #     if mask >5:
    #         break
    #     if np.amax(masks[mask]) > 0:
    #         print('MAX:     ',np.amax(masks[mask]))
    #         for m in range(7):
    #             print('Class:' , m)
    #             for x in range(14):
    #                 for y in range(14):
    #                     print(masks[mask][m*14*14+x*14+y], end=' ')
    #                 print()
    #             print()
    #
    # print()

    blobs['masks_int32'] = masks


def _expand_to_class_specific_mask_targets(masks, mask_class_labels):
    """Expand masks from shape (#masks, M ** 2) to (#masks, #classes * M ** 2)
    to encode class specific mask targets.
    """
    assert masks.shape[0] == mask_class_labels.shape[0]
    M = cfg.MRCNN.RESOLUTION

    # Target values of -1 are "don't care" / ignore labels
    mask_targets = -blob_utils.ones(
        (masks.shape[0], cfg.MODEL.NUM_CLASSES * M**2), int32=True)

    for i in range(masks.shape[0]):
        cls = int(mask_class_labels[i])
        start = M**2 * cls
        end = start + M**2
        # Ignore background instance
        # (only happens when there is no fg samples in an image)
        if cls > 0:
            mask_targets[i, start:end] = masks[i, :]

    return mask_targets

def resize_mask_to_set_dim(mask_gt_orig_size, roi_fg, box_gt, M):
    """Take in original binary gt mask at original size in gt bbox
    Then take roi_fg (the predicted bbox) and crop the gt mask into it
    Finally output a square matrix pooled or upsampled version to
    dimensions MxM
    """
    #plus one to include the
    pred_w = int(roi_fg[2]-roi_fg[0] +1)
    pred_h = int(roi_fg[3]-roi_fg[1] +1)
    # print("mask_gt_orig_size shape")
    # print(mask_gt_orig_size.shape)
    # print("orig bbox dim")
    # print(box_gt[2]-box_gt[0], box_gt[3]-box_gt[1])
    # print("pred bbox dim")
    # print(pred_w, pred_h)
    # print('Desired Square Dim:')
    # print(M)
    mask_cropped = np.zeros((pred_h,pred_w,1), dtype=np.uint8)

    # print()
    # for y in reversed(range(mask_gt_orig_size.shape[1])):
    #     for x in range(mask_gt_orig_size.shape[0]):
    #         print(int(mask_gt_orig_size[x][y]),end=' ')
    #     print()
    # for y in reversed(range(mask_cropped.shape[1])):
    #     for x in range(mask_cropped.shape[0]):
    #         print(int(mask_cropped[x][y]),end=' ')
    #     print()
    #Find x indices to copy
    if box_gt[0] >= roi_fg[0]:
        start_copy_x = int(box_gt[0] - roi_fg[0])
    else:
        start_copy_x = 0
    if box_gt[2] >= roi_fg[2]:
        end_copy_x = pred_w
    else:
        end_copy_x = int(box_gt[2] - roi_fg[0] +1)

    #Find y indices to copy
    if box_gt[1] >= roi_fg[1]:
        start_copy_y = int(box_gt[1] - roi_fg[1])
    else:
        start_copy_y = 0
    if box_gt[3] >= roi_fg[3]:
        end_copy_y = pred_h
    else:
        end_copy_y = int(box_gt[3] - roi_fg[1] +1)

    for x in range(start_copy_x, end_copy_x):
        for y in range(start_copy_y, end_copy_y):
            mask_cropped[y][x][0] = np.uint8(mask_gt_orig_size[ y - int(box_gt[1] - roi_fg[1]) ][ x - int(box_gt[0] - roi_fg[0])])
    #now we need to figure out how to resize this to a constant MxM
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(mode='L'),
        torchvision.transforms.Resize((M,M), interpolation=2)
        ])

    pil_image =transform(mask_cropped)
    mask_resized = np.array(pil_image)
    # print('Shape', (mask_resized.shape))



    return mask_resized

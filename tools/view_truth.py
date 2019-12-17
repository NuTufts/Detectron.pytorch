from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import os
import sys
import pprint
import subprocess
from collections import defaultdict
from six.moves import xrange

# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable

import _init_paths
import nn as mynn
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test import im_detect_all
from modeling.model_builder import Generalized_RCNN
import datasets.dummy_datasets as datasets
import utils.misc as misc_utils
import utils.net as net_utils
import utils.vis as vis_utils
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer

import ROOT
from larcv import larcv


# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn truth')

    parser.add_argument(
        '--image_file',
        help='directory to load images for demo')
    parser.add_argument(
        '--images', nargs='+',
        help='images to infer. Must not use with --image_file')
    parser.add_argument(
        '--output_dir',
        help='directory to save demo results',
        default="view_truth")
    parser.add_argument(
        '--merge_pdfs', type=distutils.util.strtobool, default=True)
    parser.add_argument(
        '--num_images',
        help='Perform Infer on num_images or total images in file, whichever is less',
        default=10, type=int)

    parser.add_argument(
        '--start_image',
        help='Number of image to start on from root file',
        default=0, type=int)
    parser.add_argument(
        '--plane',
        help='Number of image to start on from root file',
        default=2, type=int)
    args = parser.parse_args()

    return args


def main():
    """main function"""

    args = parse_args()
    print('Called with args:')
    print(args)

    assert args.image_file or args.images
    assert bool(args.image_file) ^ bool(args.images)

    image2d_adc_crop_chain = ROOT.TChain("image2d_adc_tree")
    clustermask_cluster_crop_chain = ROOT.TChain("clustermask_masks_tree")

    _files = [args.image_file]
    for _file in _files: image2d_adc_crop_chain.AddFile(_file)
    for _file in _files: clustermask_cluster_crop_chain.AddFile(_file)
    print(image2d_adc_crop_chain.GetEntries())
    assert image2d_adc_crop_chain.GetEntries() == clustermask_cluster_crop_chain.GetEntries()
    assert image2d_adc_crop_chain.GetEntries() >= args.num_images+args.start_image

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    dataset = datasets.get_particle_dataset()

    for i in range(args.start_image, args.num_images):
        print('img', i)
        #first get image
        image2d_adc_crop_chain.GetEntry(i)
        entry_image2dadc_crop_data = image2d_adc_crop_chain.image2d_adc_branch
        image2dadc_crop_array = entry_image2dadc_crop_data.as_vector()
        im_2d =  np.transpose(larcv.as_ndarray(image2dadc_crop_array[cfg.PLANE]))
        im = np.moveaxis(np.array([np.copy(im_2d),np.copy(im_2d),np.copy(im_2d)]),0,2)

        #now get bounding boxes, masks, classes
        clustermask_cluster_crop_chain.GetEntry(i)
        entry_clustermaskcluster_crop_data = clustermask_cluster_crop_chain.clustermask_masks_branch
        clustermask_array = entry_clustermaskcluster_crop_data.as_vector()
        # boxes = np.ones((clustermask_array[args.plane].size(),5))
        boxes = []
        for ____ in range(7):
            boxes.append([])
        for idx, mask in enumerate(clustermask_array[args.plane]):
            mask_box_arr = larcv.as_ndarray_bbox(mask)
            if (int(mask_box_arr[4]) > 6):
                continue
            boxes[int(mask_box_arr[4])].append(np.array([mask_box_arr[0],
                                    mask_box_arr[1],
                                    mask_box_arr[0]+mask_box_arr[2],
                                    mask_box_arr[1]+mask_box_arr[3],
                                    mask_box_arr[4]]))



        assert im is not None

        timers = defaultdict(Timer)


        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            "gt_boxes_"+str(i),
            args.output_dir,
            boxes,
            None, #segms
            None, #keypoints
            dataset=dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.4,
            kp_thresh=2,
            no_adc=False,
            entry=i,
        )
        for idx, mask in enumerate(clustermask_array[args.plane]):
            # mask_bin_arr = (np.transpose(larcv.as_ndarray_mask(mask)))
            mask_bin_arr = ((larcv.as_ndarray_mask(mask)))
            mask_box_arr = larcv.as_ndarray_bbox(mask)
            if (int(mask_box_arr[4]) > 6):
                continue
            x_start = int(mask_box_arr[0])
            y_start = int(mask_box_arr[1])
            for x_idx in range(mask_bin_arr.shape[0]):
                for y_idx in range(mask_bin_arr.shape[1]):
                    if mask_bin_arr[x_idx][y_idx] == 1:
                        im[y_start+y_idx][x_start+x_idx][:] = 200

        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            "gt_boxes_"+str(i)+"_masks",
            args.output_dir,
            boxes,
            None, #segms
            None, #keypoints
            dataset=dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.4,
            kp_thresh=2,
            no_adc=False,
            entry=i,
        )

    if args.merge_pdfs:
        merge_out_path = '{}/gt_results.pdf'.format(args.output_dir)
        if os.path.exists(merge_out_path):
            os.remove(merge_out_path)
        command = "pdfunite {}/*.pdf {}".format(args.output_dir,
                                                merge_out_path)
        subprocess.call(command, shell=True)


if __name__ == '__main__':
    main()

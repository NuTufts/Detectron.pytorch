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

#for LArCVDataset
import os,time
import ROOT
from larcv import larcv
import numpy as np
from torch.utils.data import Dataset
#new imports:
import cv2

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn results')
    parser.add_argument(
        '--dataset', required=True,
        help='training dataset')

    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file',
        default=[], nargs='+')

    parser.add_argument(
        '--no_cuda', dest='cuda', help='whether use CUDA', action='store_false')

    parser.add_argument('--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--image_dir',
        help='directory to load images for demo')
    parser.add_argument(
        '--images', nargs='+',
        help='images to infer. Must not use with --image_dir')
    parser.add_argument(
        '--output_dir',
        help='directory to save demo results',
        default="infer_outputs")
    parser.add_argument(
        '--merge_pdfs', type=distutils.util.strtobool, default=True)

    args = parser.parse_args()

    return args


def main():
    """main function"""

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    args = parse_args()
    print('Called with args:')
    print(args)

    assert args.image_dir or args.images
    assert bool(args.image_dir) ^ bool(args.images)

    if args.dataset.startswith("coco"):
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)
    elif args.dataset.startswith("keypoints_coco"):
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == "particle":
        dataset = datasets.get_particle_dataset()
        cfg.TRAIN.DATASETS = ('particle_physics_train')
        cfg.MODEL.NUM_CLASSES = 6
        # 0=Muon (cosmic), 1=Neutron, 2=Proton, 3=Electron, 4=neutrino, 5=Other
    else:
        raise ValueError('Unexpected dataset name: {}'.format(args.dataset))

    print('load cfg from file: {}'.format(args.cfg_file))
    cfg_from_file(args.cfg_file)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
    assert_and_infer_cfg()

    maskRCNN = Generalized_RCNN()

    if args.cuda:
        maskRCNN.cuda()

    if args.load_ckpt:
        load_name = args.load_ckpt
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(maskRCNN, checkpoint['model'])

    if args.load_detectron:
        print("loading detectron weights %s" % args.load_detectron)
        load_detectron_weight(maskRCNN, args.load_detectron)

    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                 minibatch=True, device_ids=[0])  # only support single GPU

    maskRCNN.eval()

    if args.image_dir:
        file_list = os.listdir(args.image_dir)
        for idx in range(len(file_list)):
            file_list[idx]=args.image_dir+file_list[idx]
    else:
        file_list = args.images

    #collect files
    image2d_adc_crop_chain = ROOT.TChain("image2d_adc_tree")
    for file in file_list: image2d_adc_crop_chain.AddFile(file)


    num_images = image2d_adc_crop_chain.GetEntries()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    for i in xrange(20):
        print('img', i)
        image2d_adc_crop_chain.GetEntry(i)
        entry_image2dadc_crop_data = image2d_adc_crop_chain.image2d_adc_branch
        image2dadc_crop_array = entry_image2dadc_crop_data.as_vector()
        im_2d = larcv.as_ndarray(image2dadc_crop_array[2])
        height, width = im_2d.shape
        im = np.zeros ((height,width,3), np.int8)
        # print('height: ',roidb[i]['height'] , "     dim1: ",len(im_2d))
        # print('width: ',roidb[i]['width'] , "     dim2: ",len(im_2d[0]))

        for dim1 in range(len(im_2d)):
            for dim2 in range(len(im_2d[0])):
                if im_2d[dim1][dim2] > 250:
                    value = 250
                elif im_2d[dim1][dim2] < 0:
                    value = 0
                else:
                    value = im_2d[dim1][dim2]
                im[dim1][dim2][0] = value
                im[dim1][dim2][1] = value
                im[dim1][dim2][2] = value


        # im = cv2.imread(file_list[i])
        assert im is not None

        timers = defaultdict(Timer)

        cls_boxes, cls_segms, cls_keyps = im_detect_all(maskRCNN, im, timers=timers)

        im_name, _ = os.path.splitext(os.path.basename(file_list[0]))
        im_name = im_name+str(i)
        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            args.output_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2
        )

    if args.merge_pdfs and num_images > 1:
        merge_out_path = '{}/results.pdf'.format(args.output_dir)
        if os.path.exists(merge_out_path):
            os.remove(merge_out_path)
        command = "pdfunite {}/*.pdf {}".format(args.output_dir,
                                                merge_out_path)
        subprocess.call(command, shell=True)


if __name__ == '__main__':
    main()

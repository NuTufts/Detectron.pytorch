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
import time

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

    # parser.add_argument( #Deprecated Arg, use the device in the cfg, or use --set
    #     '--no_cuda', dest='cuda', help='whether use CUDA', action='store_false')

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

    parser.add_argument(
        '--no_vis', dest='vis', help='whether to vis', action='store_false')

    parser.add_argument(
        '--num_images',
        help='Perform Infer on num_images or total images in file, whichever is less',
        default=10, type=int)

    parser.add_argument(
        '--start_image',
        help='Number of image to start on from root file',
        default=0, type=int)

    parser.add_argument(
        '--num_loops',
        help='Number of times to run the network on a single image',
        default=2, type=int)

    args = parser.parse_args()


    return args


def main():
    """main function"""



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
        cfg.MODEL.NUM_CLASSES = 7
        # 0=Background, 1=Muon (cosmic), 2=Neutron, 3=Proton, 4=Electron, 5=neutrino, 6=Other
    else:
        raise ValueError('Unexpected dataset name: {}'.format(args.dataset))

    print('load cfg from file: {}'.format(args.cfg_file))
    cfg_from_file(args.cfg_file)
    # if (not torch.cuda.is_available()) and (args.cuda):
    #     sys.exit("Need a CUDA device to run the code.")
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    assert_and_infer_cfg(make_immutable=False)

    maskRCNN = Generalized_RCNN()
    # args.cuda = False
    # if args.cuda:
        # maskRCNN.cuda()
    maskRCNN = maskRCNN.to(torch.device(cfg.MODEL.DEVICE))


    if args.load_ckpt:
        load_name = args.load_ckpt
        print("loading checkpoint %s" % (load_name))
        if (cfg.MODEL.DEVICE != 'cpu'):
            checkpoint = torch.load(load_name, map_location={'cpu':cfg.MODEL.DEVICE,'cuda:0':cfg.MODEL.DEVICE,'cuda:1':cfg.MODEL.DEVICE,'cuda:2':cfg.MODEL.DEVICE})
        else:
            checkpoint = torch.load(load_name, map_location={'cpu':'cpu','cuda:0':'cpu','cuda:1':'cpu','cuda:2':'cpu'})

        net_utils.load_ckpt(maskRCNN, checkpoint['model'])

    if args.load_detectron:
        print("loading detectron weights %s" % args.load_detectron)
        load_detectron_weight(maskRCNN, args.load_detectron)

    # maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
    #                              minibatch=True, device_ids=[1], output_device=1)  # only support single GPU

    maskRCNN = mynn.DataSingular(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                 minibatch=True , device_id=[cfg.MODEL.DEVICE])

    maskRCNN = maskRCNN.eval()

    if args.image_dir:
        file_list = os.listdir(args.image_dir)
        for idx in range(len(file_list)):
            file_list[idx]=args.image_dir+file_list[idx]
    else:
        file_list = args.images

    #collect files
    # wire for full image, adc for crop image
    image2d_adc_crop_chain = ROOT.TChain("image2d_adc_tree")
    for file in file_list: image2d_adc_crop_chain.AddFile(file)


    num_images = image2d_adc_crop_chain.GetEntries()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.num_images > num_images:
        args.num_images = num_images
    start_image = args.start_image
    if start_image+args.num_images > num_images:
        start_image = 0

    print("Running through:", args.num_images, " images.")
    # Initialize Timing counts
    t_total = 0
    t_load =0
    t_start_b4_detect = 0
    t_detection = 0
    t_after_detect_vis = 0
    t_vis = 0
    for i in xrange(start_image, start_image+args.num_images):
        t_start = time.time()
        print('img', i)
        image2d_adc_crop_chain.GetEntry(i)
        entry_image2dadc_crop_data = image2d_adc_crop_chain.image2d_adc_branch # wire for full image, adc for crop image
        image2dadc_crop_array = entry_image2dadc_crop_data.as_vector()
        im_2d = np.transpose(larcv.as_ndarray(image2dadc_crop_array[cfg.PLANE]))
        height, width = im_2d.shape
        im = np.moveaxis(np.array([np.copy(im_2d),np.copy(im_2d),np.copy(im_2d)]),0,2)

        t_data_loaded = time.time()

        assert im is not None

        timers = defaultdict(Timer)
        t_before_detect = time.time()

        for loop_idx in xrange(args.num_loops):
            print("Looping for : ", loop_idx)
            # with torch.autograd.profiler.profile(use_cuda=False) as prof:
            if cfg.SYNCHRONIZE:
                torch.cuda.synchronize
            cls_boxes, cls_segms, cls_keyps, round_boxes = im_detect_all(maskRCNN, im, timers=timers, use_polygon=False)
            if cfg.SYNCHRONIZE:
                torch.cuda.synchronize
            # print(prof)



            print(len(cls_boxes[1]), "Boxes Made with Scores")
            t_after_detect = time.time()
            count_above =0
            for score_idx in range(len(cls_boxes[1])):
                if cls_boxes[1][score_idx][4] > 0.4:
                    print("Score above threshold!")
                    count_above = count_above+1
                # else:
                #     print(cls_boxes[1][score_idx][4])
            print(count_above, " Boxes above threshold")
            t_before_vis = time.time()

            if (args.vis == True):
                if cls_segms is None:
                    continue
                assert len(cls_boxes) == len(cls_segms)
                assert len(cls_boxes) == len(round_boxes)
                # im_vis2 = np.moveaxis(np.array([np.copy(im),np.copy(im),np.copy(im)]),0,2)

                im_vis2 = np.copy(im)
                im_vis2[im_vis2 < 10] = 0
                # for row in range(height):
                #     for col in range(width):
                #         if (im[row][col][0] > 10):
                #             im_vis2[row][col][:] = im[row][col][0]

                #Having this next few forloops prints the masks over the raw image.
                for cls in range(len(cls_boxes)):
                    for roi in range(len(cls_boxes[cls])):
                        if cls_boxes[cls][roi][4] > 0.7:
                            #code to adjust im_visualize
                            add_x = round_boxes[cls][roi][0]
                            add_y = round_boxes[cls][roi][1]
                            segm_coo = cls_segms[cls][roi].tocoo()
                            for ii,jj,vv in zip(segm_coo.row, segm_coo.col, segm_coo.data):
                                # print("In here")
                                im_vis2[add_y + ii][add_x + jj][:] = 100#1.0*(roi+1)
                                im[add_y + ii][add_x + jj][:] = 0
                vis_utils.vis_one_image(
                    im_vis2[:, :, ::-1],  # BGR -> RGB for visualization
                    "output_im_"+str(i)+"_loop_"+str(loop_idx),
                    args.output_dir,
                    cls_boxes,
                    None,
                    cls_keyps,
                    dataset=dataset,
                    box_alpha=0.3,
                    show_class=True,
                    thresh=0.7,
                    kp_thresh=2,
                    no_adc=False,
                    entry=i,
                )
                if loop_idx != args.num_loops:
                    vis_utils.vis_one_image(
                        im[:, :, ::-1],  # BGR -> RGB for visualization
                        "output_im_"+str(i)+"_loop_"+str(loop_idx)+"_between",
                        args.output_dir,
                        cls_boxes,
                        None,
                        cls_keyps,
                        dataset=dataset,
                        box_alpha=0.3,
                        show_class=True,
                        thresh=0.7,
                        kp_thresh=2,
                        plain_img=True,
                        entry=i,

                    )




        t_end = time.time()
        # differences:
        t_total = t_end - t_start + t_total
        t_load = t_data_loaded - t_start + t_load
        t_start_b4_detect = t_before_detect - t_data_loaded + t_start_b4_detect
        t_detection = t_after_detect - t_before_detect + t_detection
        t_after_detect_vis = t_before_vis - t_after_detect + t_after_detect_vis
        t_vis = t_end - t_before_vis + t_vis
    print()
    print("Time Load from ROOT:                            %.3f ( %.3f" % (t_load, t_load/t_total*100), "%)")
    print("Time ADC Im Threshold                           %.3f ( %.3f" % (t_start_b4_detect , t_start_b4_detect/t_total*100) , "%)")
    print("Time to Detect                                  %.3f ( %.3f" % (t_detection , t_detection/t_total*100) , "%)")
    print("Time to score check and Sparse mask to image    %.3f ( %.3f" % (t_after_detect_vis  , t_after_detect_vis/t_total*100) , "%)")
    print("Time to Visualize                               %.3f ( %.3f" % (t_vis , t_vis/t_total*100) , "%)")
    print("-------------------------------------------------------")
    print("Total Time:  %.3f" % t_total)


    if args.merge_pdfs and num_images > 1:
        merge_out_path = '{}/results.pdf'.format(args.output_dir)
        if os.path.exists(merge_out_path):
            os.remove(merge_out_path)
        command = "pdfunite {}/*.pdf {}".format(args.output_dir,
                                                merge_out_path)
        subprocess.call(command, shell=True)


if __name__ == '__main__':
    main()
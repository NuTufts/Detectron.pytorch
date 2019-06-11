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

def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn results')


    parser.add_argument(
        '--infile', required=True,
        help="path to file")

    parser.add_argument(
        '--outfile', required=False,
        help="path to out file",
        default='ubmrcnn.root')

    parser.add_argument(
        '--num_entries',
        help='Perform Infer on num_images or total images in file, whichever is less',
        default=-1, type=int)

    args = parser.parse_args()


    return args

def main():
            args = parse_args()
            print('Called with args:')
            print(args)

            larcv_adc_file = args.infile
            output_larcv_filename = args.outfile
            mrcnn_tree_name  = "masks"
            adc_producer     = "adc"
            tick_backwards = False
            log_protocol = larcv.msg.kNORMAL
            planes = [0,1,2]
            weight_files = ["/home/jmills/workdir/ubdl/ublarcvserver/app/ubmrcnn/mcc8_mrcnn_plane0.pth",
                            "/home/jmills/workdir/ubdl/ublarcvserver/app/ubmrcnn/mcc8_mrcnn_plane1.pth",
                            "/home/jmills/workdir/ubdl/ublarcvserver/app/ubmrcnn/mcc8_mrcnn_plane2.pth"
                            ]

            # setup the input and output larcv iomanager, input larlite manager
            tick_direction = larcv.IOManager.kTickForward
            if tick_backwards:
                tick_direction = larcv.IOManager.kTickBackward

            io_in = larcv.IOManager(larcv.IOManager.kREAD,"",
                                            tick_direction)
            io_in.add_in_file(larcv_adc_file)
            io_in.set_verbosity(log_protocol)

            io_in.initialize()

            io_out_final = larcv.IOManager(larcv.IOManager.kWRITE)
            io_out_final.set_out_file(args.outfile)
            io_out_final.set_verbosity(log_protocol)
            io_out_final.initialize()



            for plane in planes:
                out_managers = [larcv.IOManager(larcv.IOManager.kWRITE), larcv.IOManager(larcv.IOManager.kWRITE),larcv.IOManager(larcv.IOManager.kWRITE)]
                io_out = out_managers[plane]
                io_out = larcv.IOManager(larcv.IOManager.kWRITE)
                io_out.set_out_file("plane_"+str(plane)+".root")
                io_out.set_verbosity(log_protocol)
                io_out.initialize()
                print("Plane:", plane)
                #Get Configs going:
                dataset = datasets.get_particle_dataset()
                cfg.TRAIN.DATASETS = ('particle_physics_train')
                cfg.MODEL.NUM_CLASSES = 7

                print('load cfg from file: {}'.format("mills_config_"+str(plane)+".yaml"))
                cfg_from_file("configs/baselines/mills_config_"+str(plane)+".yaml")
                cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
                assert_and_infer_cfg(False)

                # ////////////////////////////////////
                model = Generalized_RCNN()

                locations = {}
                for x in range(6):
                    locations["cuda:%d"%(x)] = "cpu"
                checkpoint = torch.load(weight_files[plane], map_location=locations)

                device_id = torch.device("cpu")



                net_utils.load_ckpt(model, checkpoint['model'])
                model = mynn.DataSingular(model, cpu_keywords=['im_info', 'roidb'],
                                               minibatch=True, device_id=["cpu"])  # only support single GPU
                model.eval()
                # ////////////////////////////////////

                nentries = args.num_entries
                if args.num_entries < 1:
                    nentries = io_in.get_n_entries()


                for entry_num in range(nentries):
                    print("     Entry:", entry_num)

                    ok = io_in.read_entry(entry_num)
                    if not ok:
                        raise RuntimeError("could not read larcv entry %d"%(entry_num))

                    ev_adc = io_in.get_data(larcv.kProductImage2D,
                                                          adc_producer)

                    adc_v  = ev_adc.Image2DArray()
                    nplanes = adc_v.size()

                    run    = io_in.event_id().run()
                    subrun = io_in.event_id().subrun()
                    event  = io_in.event_id().event()
                    # print("num of planes in entry {}: ".format((run,subrun,event)),nplanes)

                    # define the roi_v images
                    # img2d_v = {}
                    #
                    # for plane in range(nplanes):
                    #     img2d_v[plane] = [adc_v.at(plane)]
                    adc_img = adc_v.at(plane)

                    # This is the worker
# Begin of Make reply
                    im_np = larcv.as_ndarray(adc_img)
                    meta   = adc_img.meta()
                    height = meta.rows()
                    width  = meta.cols()

                    im = np.array([np.copy(im_np),np.copy(im_np),np.copy(im_np)])
                    im = np.moveaxis(np.moveaxis(im,0,2),0,1)

                    assert im is not None
                    thresh = 0.7
                    # print("Using a score threshold of 0.7 to cut boxes. Hard Coded")
                    clustermasks_this_img = []
                    cls_boxes, cls_segms, cls_keyps, round_boxes = im_detect_all(model, im, timers=None, use_polygon=False)
                    np.set_printoptions(suppress=True)
                    nmasks = 0
                    for cls in range(len(cls_boxes)):
                        assert len(cls_boxes[cls]) == len(cls_segms[cls])
                        assert len(cls_boxes[cls]) == len(round_boxes[cls])
                        for roi in range(len(cls_boxes[cls])):
                            if cls_boxes[cls][roi][4] > thresh:
                                segm_coo = cls_segms[cls][roi].tocoo()
                                non_zero_num = segm_coo.count_nonzero()
                                segm_np = np.zeros((non_zero_num, 2), dtype=np.float32)
                                counter = 0
                                for i,j,v in zip(segm_coo.row, segm_coo.col, segm_coo.data):
                                    segm_np[counter][0] = j
                                    segm_np[counter][1] = i
                                    counter = counter+1
                                round_box = np.array(round_boxes[cls][roi], dtype=np.float32)
                                round_box = np.append(round_box, np.array([cls], dtype=np.float32))



                                clustermasks_this_img.append(larcv.as_clustermask(segm_np, round_box, meta, np.array([cls_boxes[cls][roi][4]], dtype=np.float32)))
                                nmasks = nmasks+1
# End of make reply
                    ev_clustermasks = io_out.\
                                    get_data(larcv.kProductClusterMask,
                                             mrcnn_tree_name)
                    masks_vv = ev_clustermasks.as_vector()
                    if len(masks_vv) != len(planes):
                        masks_vv.resize(1)
                    print("Length: ", len(clustermasks_this_img))
                    for mask in clustermasks_this_img:

                        masks_vv.at(0).push_back(mask)

                    io_out.set_id( io_in.event_id().run(),
                                       io_in.event_id().subrun(),
                                       io_in.event_id().event())
                    io_out.save_entry()

                    # End of the Worker
                io_out.finalize()

            # Out of the plane loop
            # Now must combine the 3 plane files
            ios_in = [larcv.IOManager(larcv.IOManager.kREAD,"",tick_direction), larcv.IOManager(larcv.IOManager.kREAD,"",tick_direction), larcv.IOManager(larcv.IOManager.kREAD,"",tick_direction)]
            for plane in planes:
                ios_in[plane].add_in_file("plane_"+str(plane)+".root")
                ios_in[plane].set_verbosity(log_protocol)

                ios_in[plane].initialize()

            if args.num_entries < 1:
                nentries = io_in.get_n_entries()

            for entry_num in range(nentries):
                for plane in planes:
                    ok = ios_in[plane].read_entry(entry_num)
                    if not ok:
                        raise RuntimeError("could not read larcv entry %d"%(entry_num))


                ev_clustermasks = io_out_final.get_data(larcv.kProductClusterMask, mrcnn_tree_name)
                ev_clust0 = ios_in[0].get_data(larcv.kProductClusterMask, mrcnn_tree_name)
                ev_clust1 = ios_in[1].get_data(larcv.kProductClusterMask, mrcnn_tree_name)
                ev_clust2 = ios_in[2].get_data(larcv.kProductClusterMask, mrcnn_tree_name)


                masks_vv = ev_clustermasks.as_vector()
                print("Nmasks")
                print(len(ev_clust0.as_vector()[0]))
                print(len(ev_clust1.as_vector()[0]))
                print(len(ev_clust2.as_vector()[0]))
                masks_vv.push_back(ev_clust0.as_vector()[0])
                masks_vv.push_back(ev_clust1.as_vector()[0])
                masks_vv.push_back(ev_clust2.as_vector()[0])
                io_out_final.set_id( ios_in[0].event_id().run(),
                                   ios_in[0].event_id().subrun(),
                                   ios_in[0].event_id().event())
                io_out_final.save_entry()


            io_out_final.finalize()
            for plane in planes:
                os.remove("plane_"+str(plane)+".root")











if __name__ == '__main__':
    main()

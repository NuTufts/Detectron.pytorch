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
# from core.test import im_detect_all, purity_calculation, efficiency_calculation, best_iou
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
import pycocotools.mask as mask_util
import time


# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn results')


    parser.add_argument('--load_root', help='path of root file to use')

    parser.add_argument(
        '--output_dir',
        help='directory to save deploy plots',
        default="outputs_plot/")

    parser.add_argument(
        '--num_entries',
        help='Perform plots on num_images or total images in file, whichever is less',
        default=10, type=int)

    parser.add_argument(
        '--start_entry',
        help='Start plots from this entry',
        default=0, type=int)

    parser.add_argument(
        '--no_save', help='do not save anything', action='store_true')

    args = parser.parse_args()


    return args


def main():

    """main function"""
    """Here we make a 2D histogram of
    Purity Avg and Efficiency Avg across Events"""



    args = parse_args()
    print('Called with args:')
    print(args)
    list_ckpts = ["Eff_Purity_0003999", "Eff_Purity_0319999"]

    if not os.path.exists(args.output_dir) and not args.no_save:
        os.makedirs(args.output_dir)

    read_file = ROOT.TFile(args.load_root, 'READ' )
    tree_dict = {}
    tree_keys = []
    for key in read_file.GetListOfKeys():
       tree_dict[key.GetName()] = key.ReadObj()
       if (key.GetName() in list_ckpts):
           tree_keys.append(key.GetName())

    print(tree_keys)

    tree_keys.sort()

    time_start = time.time()
    write_file = ROOT.TFile(args.output_dir+"manipulated.root", 'RECREATE')
    w = 1400
    h =  700
    ROOT.gStyle.SetStatY(0.9);
    ROOT.gStyle.SetStatX(0.30);
    can  = ROOT.TCanvas("can", "histograms   ", w, w)
    can.SetRightMargin(0.15)
    height = can.GetWindowHeight()
    # can.SetCanvasSize(height, height)
    ROOT.gPad.SetLogz();

    for key in tree_keys:


        suffix_iter = key.split("Eff_Purity_",1)[1]
        step_float = float(suffix_iter)
        epochs = 0.0
        epochs_specialized =0.0
        title = 'True Objects Vs Masks '
        title2 = 'Fractional Mask Coverage '
        title3 = 'GT Area vs Coverage '

        title_end=''
        if step_float < 201001:
            epochs = (step_float * 2.0) / 229495.0
            title_end = title_end + str(epochs)[0:4] + " Epochs"
        else:
            epochs = (201000.0*2.0) / 229495.0
            epochs_specialized = ((step_float - 201000.0) * 2.0) / 29839.0
            title_end = title_end + str(epochs)[0:4] + " Epochs, " + str(epochs_specialized)[0:5] + " Specialized Epochs"

        hist_num_objects       = ROOT.TH2D('True Objects Vs Masks '+suffix_iter, title+title_end, 20, 0.0, 20.0, 20, 0.0, 20.0)
        hist_num_objects.SetOption('COLZ')
        hist_num_objects.SetMaximum(5000)

        hist_coverage          = ROOT.TH1D('Mask Coverage '+suffix_iter,title2+title_end, 20,0.0,1.01)
        hist_gt_area                = ROOT.TH2D("GT Box Area vs Coverage "+suffix_iter, title3+title_end, 50,0.,50000., 20,0.,1.01)
        hist_gt_area.SetOption('COLZ')
        # hist_demo              = ROOT.TH1D('demo',"demo", 20,0.0,1.01)
        # for num in range(12+1):
        #     for num2 in range(num+1):
        #         if (num !=0):
        #             hist_demo.Fill(float(float(num2)/float(num)))
        # can_demo =  ROOT.TCanvas("can2", "histograms   ", w, w)
        # hist_demo.Draw()
        # can_demo.SaveAs("demo.png")







        # key = 'Eff_Purity_0200999'
        max_entries = tree_dict[key].GetEntries()
        num_entries = args.num_entries
        if ((args.start_entry !=0) and (args.start_entry > 0)):
            num_entries = args.num_entries+args.start_entry
        if (num_entries > max_entries):
            num_entries = max_entries

        print ('Found ', max_entries, 'entries in ', key)
        print ('Looping through ', num_entries, " starting at ", args.start_entry  )
        for entry in range(args.start_entry, num_entries):
            print()
            print("Entry:", entry)
            tree_dict[key].GetEntry(entry)
            n_gt_masks = len(tree_dict[key].Eff)
            n_covered = 0
            for idx in range(n_gt_masks):
                gt_area = tree_dict[key].GT_Area.at(idx)
                if (tree_dict[key].Eff.at(idx) > 0.9):
                    n_covered = n_covered + 1
                    print('covered!!!! ', tree_dict[key].Eff.at(idx))
                    print(gt_area)

                else:
                    print('fail        ', tree_dict[key].Eff.at(idx))
                    print(gt_area)
                hist_gt_area.Fill(gt_area, tree_dict[key].Eff.at(idx))
            hist_num_objects.Fill(n_gt_masks, n_covered)

            if (n_gt_masks !=0):
                hist_coverage.Fill(float(float(n_covered)/float(n_gt_masks)))
        hist_num_objects.Write()
        # hist_num_objects.SetMaximum(6000)
        # hist_num_objects.SetMinimum(1)
        hist_num_objects.SetXTitle("Number of True Objects")
        hist_num_objects.SetYTitle("Number of Objects Covered")
        hist_num_objects.Draw('colz')
        if not (args.no_save):
            can.SaveAs(args.output_dir+'True_vs_Coverage_'+suffix_iter+".png")

        hist_coverage.Write()
        # hist_coverage.SetMaximum(6000)
        # hist_coverage.SetMinimum(1)
        hist_coverage.SetXTitle("Fraction of Particles Covered")
        hist_coverage.SetYTitle("Count")
        hist_coverage.Draw('colz')
        if not (args.no_save):
            can.SaveAs(args.output_dir+'Coverage_'+suffix_iter+".png")
    time_diff = time.time() - time_start
    print("Time Taken")
    hours = np.floor(time_diff/3600)
    minutes =np.floor( (time_diff - hours*3600) /60)
    seconds = (time_diff - hours*3600 - minutes*60)
    print(hours, " Hours" )
    print(minutes, " Minutes" )
    print(seconds, " Seconds" )



    if not args.no_save:
        write_file.Write()
        write_file.Close()

if __name__ == '__main__':
    main()

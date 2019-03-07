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
from core.test import im_detect_all, purity_calculation, efficiency_calculation, best_iou
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
        default="outputs_plots/")

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


    if not os.path.exists(args.output_dir) and not args.no_save:
        os.makedirs(args.output_dir)

    read_file = ROOT.TFile(args.load_root, 'READ' )
    tree_dict = {}
    tree_keys = []
    for key in read_file.GetListOfKeys():
       tree_dict[key.GetName()] = key.ReadObj()
       tree_keys.append(key.GetName())

    tree_keys.sort()

    time_start = time.time()
    if not args.no_save:
        write_file = ROOT.TFile(args.output_dir+"manipulated.root", 'RECREATE')
    w = 1400
    h =  700
    ROOT.gStyle.SetStatY(0.9);
    ROOT.gStyle.SetStatX(0.45);
    can  = ROOT.TCanvas("can", "histograms   ", w, h)
    # ROOT.gPad.SetLogy();
    num_tree = 0
    for key in tree_keys:
        # if num_tree > 0:
        #     break
        num_tree = num_tree+1

        if not args.no_save:
            suffix_iter = key.split("Eff_Purity_",1)[1]
            step_float = float(suffix_iter)
            epochs = 0.0
            epochs_specialized =0.0
            title = ''
            if step_float < 201001:
                epochs = (step_float * 2.0) / 229495.0
                title = title + str(epochs)[0:4] + " Epochs"
            else:
                epochs = ( 201000.0  * 2.0) / 229495.0
                epochs_specialized = ((step_float - 201000.0) * 2.0) / 29839.0
                title = title + str(epochs)[0:4] + " Epochs, " + str(epochs_specialized)[0:5] + " Specialized Epochs"

            eff_hist       = ROOT.TH1D('Efficiency '+suffix_iter, "Eff "+title, 30, -0.001, 1.001)
            eff_hist_2     = ROOT.TH1D('Efficiency '+suffix_iter, "Eff "+title, 30, -0.001, 1.001)
            eff_hist_4     = ROOT.TH1D('Efficiency '+suffix_iter, "Eff "+title, 30, -0.001, 1.001)
            eff_hist_6     = ROOT.TH1D('Efficiency '+suffix_iter, "Eff "+title, 30, -0.001, 1.001)
            eff_hist_8     = ROOT.TH1D('Efficiency '+suffix_iter, "Eff "+title, 30, -0.001, 1.001)

            pur_hist       = ROOT.TH1D('Purity '+suffix_iter, "Pur "+title, 30, -0.001, 1.001)



        num_entries = tree_dict[key].GetEntries()
        print ('Found', num_entries, 'entries in ', key)
        for entry in range(0,num_entries):

            tree_dict[key].GetEntry(entry)
            Eff  = tree_dict[key].Eff
            Pur = tree_dict[key].Purities
            Eff_IoU = tree_dict[key].Eff_IoU_1
            Pur_IoU_2 = tree_dict[key].Pur_IoU_2

            assert len(Eff_IoU) == len(Eff)
            assert len(Pur_IoU_2) == len(Pur)

            if len(Eff) !=0:
                for idx in range(len(Eff)):
                    eff_hist.Fill(Eff[idx])
                    if Eff_IoU[idx] > 0.2:
                        eff_hist_2.Fill(Eff[idx])
                        if Eff_IoU[idx] > 0.4:
                            eff_hist_4.Fill(Eff[idx])
                            if Eff_IoU[idx] > 0.6:
                                eff_hist_6.Fill(Eff[idx])
                                if Eff_IoU[idx] > 0.8:
                                    eff_hist_8.Fill(Eff[idx])
            # if len(Pur) !=0:
            #     for idx in range(len(Pur)):
            #         if Pur_IoU_2[idx] > 0.2:
            #             pur_hist.Fill(Pur[idx])

        eff_hist.SetFillColor(ROOT.kBlack);
        eff_hist_2.SetFillColor(ROOT.kGreen);
        eff_hist_4.SetFillColor(ROOT.kBlue);
        eff_hist_6.SetFillColor(ROOT.kRed);
        eff_hist_8.SetFillColor(ROOT.kMagenta);

        eff_hist.Write()
        # eff_hist.SetMaximum(12000)
        # eff_hist.SetMinimum(1)
        # eff_hist.SetXTitle("Efficiency")
        # eff_hist.SetYTitle("Num RoI")
        # eff_hist.Draw()
        eff_hist_2.SetMaximum(2500)

        eff_hist_2.SetXTitle("Efficiency")
        eff_hist_2.SetYTitle("Num RoI")
        eff_hist_2.Draw()
        eff_hist_4.Draw("SAME")
        eff_hist_6.Draw("SAME")
        eff_hist_8.Draw("SAME")

        can.SaveAs(args.output_dir+'Efficiency_'+suffix_iter+"_fancy.png")

        # pur_hist.Write()
        # pur_hist.SetMaximum(1500)
        # # pur_hist.SetMinimum(1)
        # pur_hist.SetXTitle("Purity")
        # pur_hist.SetYTitle("Num RoI")
        # pur_hist.Draw()
        # can.SaveAs(args.output_dir+'Purity_'+suffix_iter+".png")

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

""" Training script for steps_with_decay policy"""

import argparse
import os
import sys
import pickle
import resource

import cv2
cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader



#to Vis
import numpy as np
import cv2
import ROOT
from larcv import larcv

import time



def main():
    """Main function"""
    _files = ['/media/disk1/jmills/crop_mask_train/crop_train1.root']

    write_file = ROOT.TFile("mask_distribution.root", 'RECREATE')

    image2d_adc_crop_chain = ROOT.TChain("image2d_adc_tree")
    clustermask_cluster_crop_chain = ROOT.TChain("clustermask_masks_tree")
    for _file in _files: image2d_adc_crop_chain.AddFile(_file)
    for _file in _files: clustermask_cluster_crop_chain.AddFile(_file)

    assert image2d_adc_crop_chain.GetEntries() == clustermask_cluster_crop_chain.GetEntries()

    # ROOT.gStyle.SetStatY(0.9);
    # ROOT.gStyle.SetStatX(0.30);
    w = 1400
    h =  700
    can_0  = ROOT.TCanvas("can_0", "histograms_0   ", w, h)
    hist_0       = ROOT.TH1D('Number of Masks Per Event 0-plane ', 'Number of Masks Per Event 0-plane ', 20, 0, 40)
    hist_0.SetOption('COLZ')

    can_1  = ROOT.TCanvas("can_1", "histograms_1   ", w, h)
    hist_1       = ROOT.TH1D('Number of Masks Per Event 1-plane ', 'Number of Masks Per Event 1-plane ', 20, 0, 40)
    hist_1.SetOption('COLZ')

    can_2  = ROOT.TCanvas("can_2", "histograms_2   ", w, h)
    hist_2       = ROOT.TH1D('Number of Masks Per Event 2-plane ', 'Number of Masks Per Event 2-plane ', 20, 0, 40)
    hist_2.SetOption('COLZ')
    max_0=0
    max_1=0
    max_2=0
    NUM_IMAGES=clustermask_cluster_crop_chain.GetEntries()
    for entry in range(0,NUM_IMAGES):
        if entry%20000 == 0:
            print("entry:" , entry)
        clustermask_cluster_crop_chain.GetEntry(entry)
        entry_clustermaskcluster_crop_data = clustermask_cluster_crop_chain.clustermask_masks_branch
        clustermaskcluster_crop_array = entry_clustermaskcluster_crop_data.as_vector()

        num_masks_0 = len(clustermaskcluster_crop_array[0])
        if num_masks_0 > max_0:
            max_0=num_masks_0
        num_masks_1 = len(clustermaskcluster_crop_array[1])
        if num_masks_1 > max_1:
            max_1=num_masks_1
        num_masks_2 = len(clustermaskcluster_crop_array[2])
        if num_masks_2 > max_2:
            max_2=num_masks_2


        hist_0.Fill(num_masks_0)
        hist_1.Fill(num_masks_1)
        hist_2.Fill(num_masks_2)
    hist_0.SetXTitle("Num Masks")
    hist_0.SetYTitle("Num Events")
    hist_0.Write()
    # hist_0.SetMaximum()
    # hist_0.SetMinimum(1)

    # hist_0.Draw('colz')
    # can_0.SaveAs('Mask_distr_0.png')
    hist_1.SetXTitle("Num Masks")
    hist_1.SetYTitle("Num Events")
    hist_1.Write()
    # hist_1.SetMaximum()
    # hist_1.SetMinimum(1)

    # hist_0.Draw('colz')
    # can_1.SaveAs('Mask_distr_1.png')

    hist_2.SetXTitle("Num Masks")
    hist_2.SetYTitle("Num Events")
    hist_2.Write()
    # hist_2.SetMaximum()
    # hist_2.SetMinimum(1)

    # hist_2.Draw('colz')
    # can_2.SaveAs('Mask_distr_2.png')
    print("max_0:", max_0)
    print("max_1:", max_1)
    print("max_2:", max_2)

if __name__ == '__main__':
    main()

import ROOT
from larcv import larcv
import numpy as np
import argparse
# This script is built to read from root files with trees featuring keys like:
 # Purities        = (vector<float>*)0x55e6002d1c00
 # Pur_Avg         = (vector<float>*)0x55e6002ca050
 # Pur_IoU_1       = (vector<float>*)0x55e6003b3c80
 # Pur_IoU_2       = (vector<float>*)0x55e6003976e0
 # Pred_Area       = (vector<int>*)0x55e6003979a0
 # Idx_Same        = (vector<int>*)0x55e600388670
 # Eff             = (vector<float>*)0x55e600387cb0
 # EffAvg          = (vector<float>*)0x55e5ffc25770
 # EffCharge       = (vector<float>*)0x55e5ff80d0e0
 # EffChargeAvg    = (vector<float>*)0x55e600397740
 # Eff_IoU_1       = (vector<float>*)0x55e6002c9ec0
 # GT_Area         = (vector<int>*)0x55e6003eda10
 # Ev_Num          = (vector<int>*)0x55e5ffc5b7a0

def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn results')
    parser.add_argument(
        '--input_file',
        help='path to input file')
    parser.add_argument(
        '--output_dir',
        default="./")

    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    input_file = args.input_file

    eff_pur_file = ROOT.TFile(input_file, 'READ')
    tree_dict = {}
    tree_keys = []
    for key in eff_pur_file.GetListOfKeys():
       tree_dict[key.GetName()] = key.ReadObj()
       tree_keys.append(key.GetName())

    for key in tree_keys:
        num_entries = tree_dict[key].GetEntries()
        print ('Found', num_entries, 'entries in ', key)
        nbins = 100
        myhist_eff = ROOT.TH1D("Efficiency","Efficiency",nbins,0,1.0001)
        myhist_pur = ROOT.TH1D("Purity","Purity",nbins,0,1.0001)

        for entry in range(0,num_entries):
            tree_dict[key].GetEntry(entry)
            eff_v      =  tree_dict[key].Eff    #length is number of true boxes
            pur_v      =  tree_dict[key].Purities    #length is number of predicted boxes
            for eff in eff_v:
                myhist_eff.Fill(eff)
            for pur in pur_v:
                myhist_pur.Fill(pur)
        ROOT.gStyle.SetStatX(0.4)
        ROOT.gStyle.SetStatY(0.9)
        can = ROOT.TCanvas("eff","eff",1000,500)
        myhist_eff.SetTitle("Efficiency")
        myhist_eff.SetXTitle("Efficiency")
        myhist_eff.SetYTitle("Number of True Particle Clusters")
        myhist_eff.Draw()
        can.SaveAs(args.output_dir+"/Efficiency_Plot.png")
        myhist_pur.SetTitle("Purity")
        myhist_pur.SetXTitle("Purity")
        myhist_pur.SetYTitle("Number of Predicted Particle Clusters")
        myhist_pur.Draw()
        can.SaveAs(args.output_dir+"/Purity_Plot.png")


if __name__ == '__main__':
    main()

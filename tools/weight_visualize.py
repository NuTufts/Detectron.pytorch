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
import modeling.SparseResNet as SparseResNet
import utils.sparseresnet_weights_helper as sparseresnet_utils
import sparseconvnet as scn

import matplotlib.pyplot as plt


import ROOT
from larcv import larcv


# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn truth')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')
    parser.add_argument(
        '--checkpoint_file',
        help='directory to load images for demo')

    parser.add_argument(
        '--output_dir',
        help='directory to save demo results',
        default="weights_vis")
    parser.add_argument(
        '--merge_pdfs', type=distutils.util.strtobool, default=True)

    parser.add_argument('--load_ckpt', help='path of checkpoint to load')

    args = parser.parse_args()

    return args


class ResNetVisualizer():
    def __init__(self, model):
        self.model = model
        print("Initialized ResNet Visualizer")
    def visualize(self, layer1,layer2,layer3, filter, lr=0.1, opt_steps=20, prefix=""):
        l1 = 0
        l2 = 0
        l3 = 0
        for name, child in self.model.named_children():
            # print(name, "Training :", child.training)
            if (l1 == layer1):
                print(name)
                for name2,child2 in child.named_children():
                    # print(name2, "Training :", child2.training)
                    if (l2 == layer2):
                        print(" ",name2)
                        if (layer3 != -1):
                            for name3,child3 in child2.named_children():
                                # print(name3, "Training :", child3.training)
                                if (l3 == layer3):
                                    print("     ",name3)
                                l3=l3+1
                    l2=l2+1
            l1=l1+1
        # img_var = torch.tensor(img, requires_grad=True).to(torch.device(cfg.MODEL.DEVICE))
        # img_var = (torch.rand((1,3,512,832),device=cfg.MODEL.DEVICE,requires_grad=True)+150*torch.ones((1,3,512,832),device=cfg.MODEL.DEVICE,requires_grad=True))/255
        # img = np.uint8(np.random.uniform(150, 180, (1,3,512,832)))/255  # generate random image
        img_var = torch.rand((1,3,512,832),device=cfg.MODEL.DEVICE,requires_grad=True)
        # 64 64 works
        # 128 128 works

        img_var = Variable(img_var, requires_grad=True)
        print()
        optimizer = torch.optim.Adam([img_var], lr=lr, weight_decay=1e-6)

        print("Img Sum:",torch.sum(img_var).detach().cpu().data.item())
        for n in range(opt_steps):  # optimize pixel values for opt_steps times
            print("             opt_step :",n)
            print()
            # print("torch.sum(img_var)",torch.sum(img_var))
            optimizer.zero_grad()
            # output = self.model(img_var)
            x = self.model.padder(img_var)
            x = self.model.sparsifier(x)
            # First run through macro layers up to the layer being examined
            desire_layer = -1
            if (layer1 > 0):
                print("     Passed 1")
                x = self.model.res1(x)
                if (layer1 > 1):
                    print("     Passed 2")
                    x = self.model.res2(x)
                    if (layer1 > 2):
                        print("     Passed 3")
                        x = self.model.res3(x)
                        #Do Res 4 Manually until Desired Layer
                        print("     Doing Res 4 Manually")
                        for sub in range(layer2-1): #save last sublayer to do manually
                            x = list(self.model.res4.children())[sub](x)
                        for subsub in range(layer3-1):
                            x = list(list(self.model.res4.children())[layer2].children())[subsub](x)
                        desire_layer = list(list(self.model.res4.children())[layer2].children())[layer3]
                    else: #Do res 3 manually until desired layer
                        print("     Doing Res 3 Manually")
                        for sub in range(layer2-1): #save last sublayer to do manually
                            x = list(self.model.res3.children())[sub](x)
                        for subsub in range(layer3-1):
                            x = list(list(self.model.res3.children())[layer2].children())[subsub](x)
                        desire_layer = list(list(self.model.res3.children())[layer2].children())[layer3]
                else: #Do Res 2 manually until desired layer
                    print("     Doing Res 2 Manually")
                    for sub in range(layer2-1): #save last sublayer to do manually
                        x = list(self.model.res2.children())[sub](x)
                    for subsub in range(layer3-1):
                        x = list(list(self.model.res2.children())[layer2].children())[subsub](x)
                    desire_layer = list(list(self.model.res2.children())[layer2].children())[layer3]
            else: #Do Res 1 manually until desired layer
                print("     Doing Res 1 Manually")
                for sub in range(layer2-1): #save last sublayer to do manually
                    x = list(self.model.res1.children())[sub](x)
                desire_layer = list(self.model.res1.children())[layer2]
            print()
            x = desire_layer(x)
            print("     Number of Filters", x.features.shape[1])
            loss = -x.features[:,filter].mean()
            print("     loss :", loss.detach().cpu().data.item())
            loss.backward()
            optimizer.step()
            print("     Img Sum:",torch.sum(img_var).detach().cpu().data.item())
            print("     Img Mean:",img_var.detach().cpu().data.mean().item())
        img = img_var.data.cpu().numpy()[0].transpose(1,2,0)
        self.output = img
            # if _ > 30:
        self.save(layer1,layer2,layer3, filter,prefix)

    def save(self, layer1,layer2,layer3, filter, prefix):
        plt.imsave("weights_vis/"+prefix+"layer_"+str(layer1)+"_"+str(layer2)+"_"+str(layer3)+"_filter_"+str(filter)+".jpg", np.clip(self.output, 0, 1))
    def save_custom(self, string, input):
        plt.imsave(string,np.clip(input,0,1))


class BoxVisualizer():
    def __init__(self, model):
        self.model = model
        print("Initialized Box Visualizer")
    def visualize(self, layer1,layer2,layer3, filter, lr=0.1, opt_steps=20, prefix=""):

        img_var = torch.rand((1,3,64,64),device=cfg.MODEL.DEVICE,requires_grad=True)
        # 64 64 works
        # 128 128 works
        img_var = Variable(img_var, requires_grad=True)
        optimizer = torch.optim.Adam([img_var], lr=lr, weight_decay=1e-6)

        print("Img Sum:",torch.sum(img_var).detach().cpu().data.item())
        for n in range(opt_steps):  # optimize pixel values for opt_steps times
            print("             opt_step :",n)
            print()
            # print("torch.sum(img_var)",torch.sum(img_var))
            optimizer.zero_grad()
            # output = self.model(img_var)
            im_info = torch.Tensor([[img_var.shape[2], img_var.shape[3],1]])
            blob_conv = self.model.Conv_Body(img_var)
            # print("type(blob_conv)",type(blob_conv),blob_conv.shape)
            rpn_ret = self.model.RPN(blob_conv,im_info,None)
            # print("type(rpn_ret)",type(rpn_ret))
            # for k,v in rpn_ret.items():
            #     print(" ",k, type(v),"  ", v.shape)
                # if (k == "rois"):
                #     print(v)
                # print(v)
            # print()
            rpn_ret["rpn_rois"] = np.array([[0,1,1,img_var.shape[0]-2,img_var.shape[1]-2]],dtype=np.float32)
            rpn_ret["rois"] = np.array([[0,1,1,img_var.shape[0]-2,img_var.shape[1]-2]],dtype=np.float32)
            rpn_ret["rpn_roi_probs"] = np.array([[1]],dtype=np.float32)

            box_feat = self.model.Box_Head(blob_conv, rpn_ret)
            # print("type(box_feat)",type(box_feat),box_feat.shape)

            cls_score, bbox_pred = self.model.Box_Outs(box_feat)
            print()
            print("cls_score",cls_score)
            # print(-cls_score[0][5])
            # print(-cls_score[0][5].detach().cpu().data.item())
            # print("bbox_pred.shape",bbox_pred.shape,bbox_pred)

            # assert 1==2
            # desire_layer = -1
            # x = desire_layer(x)
            # print("     Number of Filters", x.features.shape[1])
            loss = 2-cls_score[0][5]
            print("     loss :", loss.detach().cpu().data.item())
            loss.backward()
            optimizer.step()
            print("     Img Sum:",torch.sum(img_var).detach().cpu().data.item())
            print("     Img Mean:",img_var.detach().cpu().data.mean().item())
            # if n%50 == 0:
            img = img_var.data.cpu().numpy()[0].transpose(1,2,0)
            print(img.shape)
            self.output = np.kron(img, np.ones((4,4,1)))
            self.save(layer1,layer2,layer3, filter,prefix,n)

        img = img_var.data.cpu().numpy()[0].transpose(1,2,0)
        self.output = img
            # if _ > 30:
        self.save(layer1,layer2,layer3, filter,prefix)

    def save(self, layer1,layer2,layer3, filter, prefix,optional=-1):
        if optional != -1:
            plt.imsave("flipbook/"+prefix+"box_class_"+str(optional)+".jpg", np.clip(self.output, 0, 1))
        else:
            plt.imsave("weights_vis/"+prefix+"box_class.jpg", np.clip(self.output, 0, 1))

    def save_custom(self, string, input):
        plt.imsave(string,np.clip(input,0,1))


def main():
    """main function"""

    args = parse_args()
    print('Called with args:')
    print(args)
    print('load cfg from file: {}'.format(args.cfg_file))
    cfg_from_file(args.cfg_file)


    assert_and_infer_cfg(make_immutable=False)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    dataset = datasets.get_particle_dataset()

    maskRCNN = Generalized_RCNN()
    maskRCNN = maskRCNN.to(torch.device(cfg.MODEL.DEVICE)).eval()

    if args.load_ckpt:
        load_name = args.load_ckpt
        print("loading checkpoint %s" % (load_name))
        if (cfg.MODEL.DEVICE != 'cpu'):
            checkpoint = torch.load(load_name, map_location={'cpu':cfg.MODEL.DEVICE,'cuda:0':cfg.MODEL.DEVICE,'cuda:1':cfg.MODEL.DEVICE,'cuda:2':cfg.MODEL.DEVICE})
        else:
            checkpoint = torch.load(load_name, map_location={'cpu':'cpu','cuda:0':'cpu','cuda:1':'cpu','cuda:2':'cpu'})

        net_utils.load_ckpt(maskRCNN, checkpoint['model'])



    sparse_resnet = maskRCNN.Conv_Body
    print(sparse_resnet)
    # To See structure of sparseresnet
    print("Len of sparseresnet",len(list(sparse_resnet.children())))
    for name,child in sparse_resnet.named_children():
        num_child_child = len(list(child.named_children()))
        if name[0] == "c":
            print(" ",name,num_child_child)
        else:
            print(" ",name,num_child_child)

        if (num_child_child > 0):
            for name2,child2 in child.named_children():
                num_child_child2 = len(list(child2.named_children()))
                print("     ",name2,num_child_child2)

                if (num_child_child2 > 0):
                    for name3,child3 in child2.named_children():
                        num_child_child3 = len(list(child3.named_children()))
                        print("         ",name3,num_child_child3)



    layer1 = 0
    layer2 = 2
    layer3 = -1
    # filter = 1
    opt_steps = 100
    RV = BoxVisualizer(maskRCNN)
    prefix = ""
    if args.load_ckpt:
        prefix = "load_"
    for filter in range(0,1):
        print("-------------------------")
        print("Doing filter:", filter)
        print("-------------------------")

        RV.visualize(layer1,layer2,layer3, filter, opt_steps=opt_steps, prefix=prefix)



if __name__ == '__main__':
    main()

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

    parser.add_argument('--load_many_ckpts', help='path of many ckpts to load')

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
        '--num_images',
        help='Perform Infer on num_images or total images in file, whichever is less',
        default=10, type=int)

    parser.add_argument(
        '--start_image',
        help='Perform Infer on num_images or total images in file, whichever is less',
        default=0, type=int)

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
        cfg.MODEL.NUM_CLASSES = 7
        # 0=Background, 1=Muon (cosmic), 2=Neutron, 3=Proton, 4=Electron, 5=neutrino, 6=Other
    else:
        raise ValueError('Unexpected dataset name: {}'.format(args.dataset))

    plane = cfg.PLANE
    print("Plane set to: ", plane)

    print('load cfg from file: {}'.format(args.cfg_file))
    cfg_from_file(args.cfg_file)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    assert bool(args.load_ckpt) ^ bool(args.load_detectron) ^ bool(args.load_many_ckpts), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
    assert_and_infer_cfg()
    ### Multi ckpt code
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.load_many_ckpts:
        load_name = args.load_many_ckpts
        ckpt_files =[]
        for (dirpath, dirnames, filenames) in os.walk(load_name):
            for file in filenames:
                if file.endswith(".pth"):
                    ckpt_files.append(os.path.join(dirpath, file))
        # print()
        #
        # for ckpt in ckpt_files:
        #     print(ckpt)
        # print()

        ckpt_files_sorted=[]
        step2file = {}
        for file in ckpt_files:
            step = int((file.split("model_step",1)[1]).split(".pth",1)[0])
            step2file[step] = file

        steps = list(step2file.keys())
        steps.sort()
        for step in steps:
            ckpt_files_sorted.append(step2file[step])

        for ckpt in ckpt_files_sorted:
            print(ckpt)
        print()

    elif args.load_ckpt:
        ckpt_files_sorted = [args.load_ckpt]
    ### Multi ckpt code
    ckpt_tracker = 1
    time_this_ckpt = 0
    for ckpt in ckpt_files_sorted:
        start_ckpt_time = time.time()
        print("Checkpoint # ",ckpt_tracker, " of ", len(ckpt_files_sorted))
        ckpt_tracker+=1
        maskRCNN = Generalized_RCNN()

        if args.cuda:
            maskRCNN.cuda()
        load_name=ckpt
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(maskRCNN, checkpoint['model'])

        print(load_name)


        maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                     minibatch=True, device_ids=[0])  # only support single GPU

        maskRCNN.eval()
        #Get Our ROOT Storage Set Up
        step = str((load_name.split("model_step",1)[1]).split(".pth",1)[0])
        while len(step) < 7:
            step = "0"+step
        f = ROOT.TFile(args.output_dir+"Eff_Pur_"+step+'.root', 'recreate' )
        t = ROOT.TTree( 'Eff_Purity_'+step, 'Efficiency and Purity' )
        #Vectors for Buffer
        Purities = ROOT.vector('float')()
        Pur_Avg = ROOT.vector('float')()
        Pur_IoU_1 =  ROOT.vector('float')()
        Pur_IoU_2 =  ROOT.vector('float')()
        Pred_Area =  ROOT.vector('int')()
        Idx_Same =  ROOT.vector('int')()


        Eff = ROOT.vector('float')()
        EffAvg = ROOT.vector('float')()
        Eff_IoU_1 =  ROOT.vector('float')()
        GT_Area =  ROOT.vector('int')()

        EffCharge = ROOT.vector('float')()
        EffChargeAvg = ROOT.vector('float')()


        Ev_Num =  ROOT.vector('int')()



        #Branches for storage
        purity_branch = t.Branch( 'Purities' , Purities )
        purity_avg_branch = t.Branch( 'Pur_Avg' , Pur_Avg )
        purity_best_iou_branch = t.Branch( 'Pur_IoU_1' , Pur_IoU_1 )
        purity_next_best_iou_branch = t.Branch( 'Pur_IoU_2' , Pur_IoU_2 )
        purity_pred_area = t.Branch( 'Pred_Area' , Pred_Area )

        Idx_Same_branch = t.Branch( 'Idx_Same' , Idx_Same )


        efficiency_branch = t.Branch( 'Eff' , Eff)
        efficiency_avg_branch = t.Branch( 'EffAvg' , EffAvg)
        efficiency_charge_branch = t.Branch( 'EffCharge' , EffCharge)
        efficiency_charge_avg_branch = t.Branch( 'EffChargeAvg' , EffChargeAvg)
        efficiency_iou_branch = t.Branch( 'Eff_IoU_1' , Eff_IoU_1)
        efficiency_gt_area = t.Branch( 'GT_Area' , GT_Area)



        event_num_branch = t.Branch( 'Ev_Num' , Ev_Num)


        if args.image_dir:
            file_list = os.listdir(args.image_dir)
            for idx in range(len(file_list)):
                file_list[idx]=args.image_dir+file_list[idx]
        else:
            file_list = args.images

        #collect files
        image2d_adc_crop_chain = ROOT.TChain("image2d_adc_tree")
        clustermask_cluster_crop_chain = ROOT.TChain("clustermask_masks_tree")
        for file in file_list: image2d_adc_crop_chain.AddFile(file)
        for file in file_list: clustermask_cluster_crop_chain.AddFile(file)



        num_images = image2d_adc_crop_chain.GetEntries()
        print("Images in file:", num_images)


        if args.num_images > num_images:
            args.num_images = num_images
        print("Running through:", args.num_images, " images.")
        print("Starting with image:  ", args.start_image)
        start_time = time.time()
        for i in xrange(args.start_image, args.start_image + args.num_images):
            if i%15 ==0 or i==args.start_image + args.num_images-1:
                print('img', i)
            image2d_adc_crop_chain.GetEntry(i)
            entry_image2dadc_crop_data = image2d_adc_crop_chain.image2d_adc_branch
            image2dadc_crop_array = entry_image2dadc_crop_data.as_vector()
            im_2d = larcv.as_ndarray(image2dadc_crop_array[plane])
            height, width = im_2d.shape
            im = np.zeros ((height,width,3))
            im_visualize = np.zeros ((height,width,3), 'float32')


            clustermask_cluster_crop_chain.GetEntry(i)
            entry_mask_crop_data = clustermask_cluster_crop_chain.clustermask_masks_branch
            mask_crop_array =  entry_mask_crop_data.as_vector()
            gt_masks = np.zeros((512,832,len(mask_crop_array[plane])))
            gt_boxes = np.empty((len(mask_crop_array[plane]),5))

            for index, mask in enumerate(mask_crop_array[plane]):
                bbox = larcv.as_ndarray_bbox(mask)
                mask_in_bbox = larcv.as_ndarray_mask(mask)
                bbox[2] = bbox[2]+bbox[0]
                bbox[3] = bbox[3]+bbox[1]
                gt_boxes[index,:] = bbox
                # print("shape gt", gt_masks[int(bbox[1]):int(bbox[3]+1),int(bbox[0]):int(bbox[2]+1),index].shape)
                # print("shape mask", larcv.as_ndarray_mask(mask).shape)
                # print("bbox[0:4]", bbox[0:4])
                # print("832 axis:", bbox[2]-bbox[0]+1)
                # print("512 axis:", bbox[3]-bbox[1]+1)
                # print()
                if larcv.as_ndarray_mask(mask).shape[0] == gt_masks[int(bbox[1]):int(bbox[3]+1),int(bbox[0]):int(bbox[2]+1),index].shape[0]+1:
                    if larcv.as_ndarray_mask(mask).shape[1] == gt_masks[int(bbox[1]):int(bbox[3]+1),int(bbox[0]):int(bbox[2]+1),index].shape[1]+1:
                        gt_masks[int(bbox[1]-1):int(bbox[3]+1),int(bbox[0]-1):int(bbox[2]+1),index]= larcv.as_ndarray_mask(mask)
                    else:
                        gt_masks[int(bbox[1]-1):int(bbox[3]+1),int(bbox[0]):int(bbox[2]+1),index]= larcv.as_ndarray_mask(mask)
                else:
                    if larcv.as_ndarray_mask(mask).shape[1] == gt_masks[int(bbox[1]):int(bbox[3]+1),int(bbox[0]):int(bbox[2]+1),index].shape[1]+1:
                        gt_masks[int(bbox[1]):int(bbox[3]+1),int(bbox[0]-1):int(bbox[2]+1),index]= larcv.as_ndarray_mask(mask)
                    else:
                        gt_masks[int(bbox[1]):int(bbox[3]+1),int(bbox[0]):int(bbox[2]+1),index]= larcv.as_ndarray_mask(mask)




            im[:,:,0] = im_2d[:][:]
            im[:,:,1] = im_2d[:][:]
            im[:,:,2] = im_2d[:][:]

            ### Save for image pdf making
            # for dim1 in range(len(im_2d)):
            #     for dim2 in range(len(im_2d[0])):
            #         im[dim1][dim2][:] = im_2d[dim1][dim2]
            #         value = im_2d[dim1][dim2]
            #         if value > 255:
            #             value2 =250
            #         elif value < 0:
            #             value2 =0
            #         else:
            #             value2  = value
            #         im_visualize[dim1][dim2][:]= value2







            # np.set_printoptions(threshold=np.inf, precision=0, suppress=True)
            # print('start')
            # print(im[0:100,0:100,0])
            # print('stop')
            # im = cv2.imread(file_list[i])
            assert im is not None

            timers = defaultdict(Timer)

            cls_boxes, cls_segms, cls_keyps = im_detect_all(maskRCNN, im, timers=timers)


            pred_boxes, segms, keypoints, pred_classes = vis_utils.convert_from_cls_format(
                cls_boxes, cls_segms, cls_keyps)


            if segms is not None:
                pred_masks = mask_util.decode(segms)

            #let's delete predictions below some threshold 0.7 what's been used in
            #imaging deploy

            threshold = 0.7
            pred_ind = 0
            if pred_boxes is None:
                continue
            while pred_ind < pred_boxes.shape[0]:

                if pred_boxes[pred_ind,:][4] < threshold:
                    pred_masks = np.delete(pred_masks, pred_ind, 2)
                    pred_boxes = np.delete(pred_boxes, pred_ind, 0)
                    if pred_boxes is None:
                        break
                else:
                    pred_ind +=1
            if pred_boxes is None:
                continue
            # print("-----------------------------------")
            # print("///////////////Start///////////////")
            # print("-----------------------------------")
            #Threshold the adc_image_bin to be conveyed to our purity/efficiency calcs
            adc_image_bin = im_2d
            # adc_image_bin[adc_image_bin != 1]  = 1
            adc_image_bin[adc_image_bin < 10] = 0
            adc_no_bin = np.copy(adc_image_bin)

            adc_image_bin[adc_image_bin >= 10] = 1

            # print("-----------------------------------")
            # print("///////////////Purity//////////////")
            # print("-----------------------------------")
            # print()

            sum_purity=0.0

            for pred_index in range(pred_masks.shape[2]):
                purity, best_iou_purity, next_best_iou, same = purity_calculation(pred_boxes[pred_index,:], pred_masks[:,:,pred_index], gt_boxes, gt_masks, adc_image_bin)
                sum_purity += purity
                area = (np.ceil(pred_boxes[pred_index,3])+1 - np.floor(pred_boxes[pred_index,1])) * (np.ceil(pred_boxes[pred_index,2])+1 - np.floor(pred_boxes[pred_index,0]))
                Purities.push_back(purity)
                Pur_IoU_1.push_back(best_iou_purity)
                Pur_IoU_2.push_back(next_best_iou)
                Pred_Area.push_back(int(area))
                Idx_Same.push_back(same)


            if pred_masks.shape[2] ==0:
                avg_purity = 0
            else:
                Pur_Avg.push_back(float(sum_purity)/float(pred_masks.shape[2]))
            # print("-----------------------------------")
            # print("/////////////Efficiency////////////")
            # print("-----------------------------------")
            # print()
            sum_efficiency=0.0
            num_uncounted = 0
            for gt_index in range(gt_masks.shape[2]):
                efficiency = efficiency_calculation(gt_boxes[gt_index,:], gt_masks[:,:,gt_index], pred_boxes, pred_masks, adc_image_bin)
                if efficiency == -1:
                    num_uncounted +=1
                    continue
                best_iou_efficiency = best_iou(gt_index, gt_boxes)
                sum_efficiency+=efficiency
                area = (gt_boxes[gt_index,2]+1-gt_boxes[gt_index,0])*(gt_boxes[gt_index,3]+1-gt_boxes[gt_index,1])
                #Record the things we need
                GT_Area.push_back(int(area))
                Eff.push_back(efficiency)
                Eff_IoU_1.push_back(best_iou_efficiency)

            if gt_masks.shape[2] - num_uncounted == 0:
                avg_efficiency=0
            else:
                EffAvg.push_back(float(sum_efficiency)/float(gt_masks.shape[2] - num_uncounted))

            #Do Charge efficiencysum_efficiency=0.0
            num_uncounted = 0
            sum_efficiency_charge = 0.0
            for gt_index in range(gt_masks.shape[2]):
                efficiency_charge = efficiency_calculation(gt_boxes[gt_index,:], gt_masks[:,:,gt_index], pred_boxes, pred_masks, adc_no_bin)
                if efficiency_charge == -1:
                    num_uncounted +=1
                    continue
                sum_efficiency_charge +=efficiency_charge
                #Record the things we need

                EffCharge.push_back(efficiency_charge)

            if gt_masks.shape[2] - num_uncounted == 0:
                avg_efficiency_charge=0
            else:
                EffChargeAvg.push_back(float(sum_efficiency_charge)/float(gt_masks.shape[2] - num_uncounted))




            #End of entry loop, fill root file
            Ev_Num.push_back(i)
            t.Fill()
            Purities.clear()
            Pur_Avg.clear()
            Pur_IoU_1.clear()
            Pur_IoU_2.clear()
            Pred_Area.clear()
            Idx_Same.clear()

            Eff.clear()
            EffAvg.clear()
            EffCharge.clear()
            EffChargeAvg.clear()
            Eff_IoU_1.clear()
            GT_Area.clear()

            Ev_Num.clear()


        time_this_ckpt = time.time() - start_ckpt_time
        print("Time for this ckpt:",time_this_ckpt)
        print("ETA:               ", time_this_ckpt * (len(ckpt_files_sorted)-ckpt_tracker+1))
        # print("Total Time in Event Loop:", end_time - start_time)
        # print("Time per event:          ", (end_time - start_time) / args.num_images)
        # t.Scan("*", '', 'colsize=10')

        f.Write()
        f.Close()
        print("End Time is: ", time.time())
        # file_list = ["/home/jmills/workdir/mask-rcnn.pytorch/"+"Eff_Pur_"+step+'.root']
        # fakechain = ROOT.TChain("Eff_Purity_"+step)
        # for file in file_list: fakechain.AddFile(file)
        # num_images = fakechain.GetEntries()
        # print("Entries",num_images)


            ### Make Image as PDF output Code
            #     im_name, _ = os.path.splitext(os.path.basename(file_list[0]))
            #     im_name = im_name+str(i)
            #     vis_utils.vis_one_image(
            #         im_visualize[:, :, ::-1],  # BGR -> RGB for visualization
            #         im_name,
            #         args.output_dir,
            #         cls_boxes,
            #         cls_segms,
            #         cls_keyps,
            #         dataset=dataset,
            #         box_alpha=0.3,
            #         show_class=True,
            #         thresh=0.7,
            #         kp_thresh=2,
            #         no_adc=False,
            #         entry=i
            #     )
            #
            # if args.merge_pdfs and num_images > 1:
            #     merge_out_path = '{}/results.pdf'.format(args.output_dir)
            #     if os.path.exists(merge_out_path):
            #         os.remove(merge_out_path)
            #     command = "pdfunite {}/*.pdf {}".format(args.output_dir,
            #                                             merge_out_path)
            #     subprocess.call(command, shell=True)

if __name__ == '__main__':
    main()

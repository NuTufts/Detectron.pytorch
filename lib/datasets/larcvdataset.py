#This is my attempt to splice our larcvdataset pytorch dataset into the json one
#Joshua Mills, Tufts University
#You will have to have larcv


#json_dataset original imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
from six.moves import cPickle as pickle
import logging
import numpy as np
import os
import scipy.sparse

# Must happen before importing COCO API (which imports matplotlib)
import utils.env as envu
envu.set_up_matplotlib()
# COCO API
from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

import utils.boxes as box_utils
from core.config import cfg
from utils.timer import Timer
from .dataset_catalog import ANN_FN
from .dataset_catalog import DATASETS
from .dataset_catalog import IM_DIR
from .dataset_catalog import IM_PREFIX

#larcvdataset original imports
import os,time
import ROOT
from larcv import larcv
import numpy as np
from torch.utils.data import Dataset
#new imports:
import cv2
import matplotlib.pyplot as pl


logger = logging.getLogger(__name__)

class LArCVDataset(object):
    """ LArCV2 data set interface for PyTorch"""

    def __init__(self, name):

        from larcv.dataloader2 import larcv_threadio
        assert name in DATASETS.keys(), \
            'Unknown dataset name: {}'.format(name)
        assert os.path.exists(DATASETS[name][IM_DIR]), \
            'Image directory \'{}\' not found'.format(DATASETS[name][IM_DIR])

        # We don't have annotation files
        # assert os.path.exists(DATASETS[name][ANN_FN]), \
        #     'Annotation file \'{}\' not found'.format(DATASETS[name][ANN_FN])


        #could replace with loops for lots of files in dir? -j
        # self.file_path =  [DATASETS[name][IM_DIR]+"croppedmask_lf.root"]
        # # self.file = ROOT.TFile(self.file_path[0])
        # print('')
        # print('')
        # #create TChains
        # #this is like our version of COCO jpg image:
        # image2d_adc_crop_chain = ROOT.TChain("image2d_adc_tree")
        # #this is like our version of COCO annotations:
        # clustermask_cluster_crop_chain = ROOT.TChain("clustermask_masks_tree")
        # #fill TChains
        # for file in self.file_path: image2d_adc_crop_chain.AddFile(file)
        # print ('Found', image2d_adc_crop_chain.GetEntries(), 'entries in image2d adc values')
        # for file in self.file_path: clustermask_cluster_crop_chain.AddFile(file)
        # print ('Found', clustermask_cluster_crop_chain.GetEntries(), 'entries in clustermask clusters cropped ')
        # print('')
        # print('')



        logger.debug('Creating: {}'.format(name))
        self.name = name
        self.image_directory = DATASETS[name][IM_DIR]
        self.image_prefix = (
            '' if IM_PREFIX not in DATASETS[name] else DATASETS[name][IM_PREFIX]
        )
        # self.COCO = COCO(DATASETS[name][ANN_FN])
        self.debug_timer = Timer()
        # Set up dataset classes
        category_ids = [1,2,3,4,5,6]
        # category_ids = self.COCO.getCatIds()
        categories = ['Cosmic', 'Neutron', 'Proton', 'Electron', 'Neutrino', 'Other']
        # categories = [c['name'] for c in self.COCO.loadCats(category_ids)]
        self.category_to_id_map = dict(zip(categories, category_ids))
        self.classes = ['__background__'] + categories
        self.num_classes = len(self.classes)
        self.plane =2
        self.keypoints = None
        if self.name == 'particle_physics_valid':
            print("Validation Set")
            self.validation = True
        else:
            self.validation =False
        # print(category_ids)
        # print(categories)
        # print(self.classes)
        # self.json_category_id_to_contiguous_id = {
        #     v: i + 1
        #     for i, v in enumerate(self.COCO.getCatIds())
        # }
        # self.contiguous_category_id_to_json_id = {
        #     v: k
        #     for k, v in self.json_category_id_to_contiguous_id.items()
        # }
        #Note we don't need these, but for now I include them so that we don't have to find where they get used -j
        self.json_category_id_to_contiguous_id = {1:1 , 2:2, 3:3, 4:4, 5:5, 6:6}
        self.contiguous_category_id_to_json_id = {1:1 , 2:2, 3:3, 4:4, 5:5, 6:6}
        # I don't think we need this, -j
        # self._init_keypoints()
        # print(self.json_category_id_to_contiguous_id)
        # print(self.contiguous_category_id_to_json_id)
        # # Set cfg.MODEL.NUM_CLASSES
        # if cfg.MODEL.NUM_CLASSES != -1:
        #     assert cfg.MODEL.NUM_CLASSES == 2 if cfg.MODEL.KEYPOINTS_ON else self.num_classes, \
        #         "number of classes should equal when using multiple datasets"
        # else:
        #     cfg.MODEL.NUM_CLASSES = 2 if cfg.MODEL.KEYPOINTS_ON else self.num_classes

        #try making an io
        # self.cfg="io_config.cfg"
        # self.filler_cfg = {}
        # self.filler_cfg["filler_name"] = 'TemporaryName'
        # self.filler_cfg["verbosity"]   = 2
        # self.filler_cfg["filler_cfg"]  = self.cfg
        # self.io = larcv_threadio()
        # self.io.configure(self.filler_cfg)
        # print('GOT THIS FAR!')
        # print('')
        # self.batchsize=1
        # if self.batchsize is not None:
        #     self.start(self.batchsize)
        # print('GOT THIS FAR!')
        # print('')

    @property
    def cache_path(self):
        cache_path = os.path.abspath(os.path.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def valid_cached_keys(self):
        """ Can load following key-ed values from the cached roidb file

        'image'(image path) and 'flipped' values are already filled on _prep_roidb_entry,
        so we don't need to overwrite it again.
        """
        keys = ['boxes', 'segms', 'gt_classes', 'seg_areas', 'gt_overlaps',
                'is_crowd', 'box_to_gt_ind_map', 'width', 'height', 'image',
                'id', 'plane' , 'flipped', 'chain_adc', 'chain_cluster',
                'max_iou']
        if self.keypoints is not None:
            keys += ['gt_keypoints', 'has_visible_keypoints']
        return keys

    def get_roidb(
            self,
            gt=False,
            proposal_file=None,
            min_proposal_size=2,
            proposal_limit=-1,
            crowd_filter_thresh=0
        ):
        """Return an roidb corresponding to the larcv dataset. Optionally:
           - include ground truth boxes in the roidb
           - add proposals specified in a proposals file
           - filter proposals based on a minimum side length
           - filter proposals that intersect with crowd regions
        """
        assert gt is True , \
            'Ground Truth Annotations Are Required to work with ROOT.'
        ###
        ###Try making my own Roidb using root file
        ###
        roidb = []
        # _files = ['/home/jmills/workdir/mask-rcnn.pytorch/data/particle_physics_train/root_files/croppedmask_lf_001.root']
        _files = ['/media/disk1/jmills/crop_mask_train/crop_train1.root']
        if self.validation == True:
            _files = ['/media/disk1/jmills/crop_mask_valid/crop_valid.root']
        # _f = ROOT.TFile(_files[0])
        image2d_adc_crop_chain = ROOT.TChain("image2d_adc_tree")
        clustermask_cluster_crop_chain = ROOT.TChain("clustermask_masks_tree")
        # print()
        for _file in _files: image2d_adc_crop_chain.AddFile(_file)
        # print ('Found', image2d_adc_crop_chain.GetEntries(), 'entries in image2d adc values')
        for _file in _files: clustermask_cluster_crop_chain.AddFile(_file)
        # print ('Found', clustermask_cluster_crop_chain.GetEntries(), 'entries in clustermask clusters cropped ')
        # print()
        assert image2d_adc_crop_chain.GetEntries() == clustermask_cluster_crop_chain.GetEntries()

        self.NUM_IMAGES=clustermask_cluster_crop_chain.GetEntries()
        # if self.validation ==True:
            # self.NUM_IMAGES=2
        self.SPECIFIC_IMAGE_START=0


        for entry in range(self.SPECIFIC_IMAGE_START,self.SPECIFIC_IMAGE_START+self.NUM_IMAGES):
            dict = {
                "height":                   512,
                "width":                    832,
                "coco_url":                 'https://bellenot.web.cern.ch/bellenot/images/logo_full-plus-text-hor2.png',
                "flickr_url":               'https://bellenot.web.cern.ch/bellenot/images/logo_full-plus-text-hor2.png',
                "id":                       entry,
                "image":                    _files[0],
                "date_captured":             'Tomorrow',
                "license":                  3,
                "plane":                    2,
                "chain_adc":                image2d_adc_crop_chain,
                "chain_cluster":            clustermask_cluster_crop_chain,
                }
            roidb.append(dict)
        #end of COCO's copy.deepcopy(self.COCO.loadImgs(image_ids)) command equivalent
        for entry in roidb:
            self._prep_roidb_entry(entry)
        print("YOU GOT HERE JOSH!")
        # print(roidb[0]['chain_adc'])


        # Include ground-truth object annotations
        cache_filepath = os.path.join(self.cache_path, self.name+'_gt_roidb.pkl')
        if os.path.exists(cache_filepath) and not cfg.DEBUG:
            self.debug_timer.tic()
            roidb = [{"dataset":                   self,} for ind in range(self.NUM_IMAGES)]
            self._add_gt_from_cache(roidb, cache_filepath)
            logger.debug(
                '_add_gt_from_cache took {:.3f}s'.
                format(self.debug_timer.toc(average=False))
            )
            # print(roidb[0]['chain_adc'])

        else:
            self.debug_timer.tic()
            print('No Cache Found, Preparing to load from ROOT Files.\n    GOOD LUCK!   \n')
            print(len(roidb), ' Entries to Load in.')

            update_every_percent = 5
            print_cond = float(len(roidb))*float(update_every_percent)/100.0
            print("Updating Every ", update_every_percent, "%")
            if int(print_cond) ==0:
                print_cond =1
            print("Updating Every ", int(print_cond), " Entries")
            count =0
            max_iou = np.zeros((0), np.float)
            all_ious = np.zeros((0), np.float)
            for entry in roidb:
                count = count+1
                if count%int(print_cond)==0:
                    print(count, " Complete took time: ", self.debug_timer.toc(average=False))
                self._add_gt_annotations(entry, clustermask_cluster_crop_chain)


            logger.debug(
            '_add_gt_annotations took {:.3f}s'.
            format(self.debug_timer.toc(average=False))
                )
            print()
            print('YOU GOT HERE NOW JOSH!!')
            print()
            if not cfg.DEBUG:
                    with open(cache_filepath, 'wb') as fp:
                        pickle.dump(roidb, fp, pickle.HIGHEST_PROTOCOL)
                    logger.info('Cache ground truth roidb to %s', cache_filepath)

        _add_class_assignments(roidb)

        t0 = time.time()
        thresh = cfg.TRAIN.GT_IOU_THRESH
        orig_length = len(roidb)
        if thresh > 0.0:
            roidb = _cull_roidb_iou(roidb, thresh)
        t1 = time.time()
        total = t1-t0
        print("Original Roidb lenth: ", orig_length)
        print("New      Roidb lenth: ", len(roidb))
        print("Time to Cull: ", total)
        return roidb




    def _prep_roidb_entry(self, entry):
        """Adds empty metadata fields to an roidb entry."""
        # Reference back to the parent dataset
        entry['dataset'] = self
        # Make image file exists
        assert os.path.exists(entry['image']), 'Image \'{}\' not found'.format(im_path)
        entry['flipped'] = False
        entry['has_visible_keypoints'] = False
        # Empty placeholders
        entry['boxes'] = np.empty((0, 4), dtype=np.float32)
        entry['max_iou'] = np.empty((0), dtype=np.float32)
        entry['segms'] = []
        entry['gt_classes'] = np.empty((0), dtype=np.int32)
        entry['seg_areas'] = np.empty((0), dtype=np.float32)
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(
            np.empty((0, self.num_classes), dtype=np.float32)
        )
        entry['is_crowd'] = np.empty((0), dtype=np.bool)
        # 'box_to_gt_ind_map': Shape is (#rois). Maps from each roi to the index
        # in the list of rois that satisfy np.where(entry['gt_classes'] > 0)
        entry['box_to_gt_ind_map'] = np.empty((0), dtype=np.int32)


        if self.keypoints is not None:
            entry['gt_keypoints'] = np.empty(
                (0, 3, self.num_keypoints), dtype=np.int32
            )
        # Remove unwanted fields that come from the json file (if they exist)
        for k in ['date_captured', 'url', 'license', 'file_name']:
            if k in entry:
                del entry[k]

    def _add_gt_annotations(self, entry, clustermask_cluster_crop_chain):
        """Add ground truth annotation metadata to an roidb entry."""
        clustermask_cluster_crop_chain.GetEntry(entry['id'])
        entry_clustermaskcluster_crop_data = clustermask_cluster_crop_chain.clustermask_masks_branch
        clustermaskcluster_crop_array = entry_clustermaskcluster_crop_data.as_vector()


        # ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
        # objs = self.COCO.loadAnns(ann_ids)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        valid_segms = []
        width = entry['width']
        height = entry['height']

        for idx, mask in enumerate(clustermaskcluster_crop_array[entry['plane']]):
            # crowd regions are RLE encoded and stored as dicts
            # if isinstance(obj['segmentation'], list):
            #     # Valid polygons have >= 3 points, so require >= 6 coordinates
            #     obj['segmentation'] = [
            #         p for p in obj['segmentation'] if len(p) >= 6
            #     ]
            obj = {}
            mask_bin_arr = larcv.as_ndarray_mask(mask)
            mask_box_arr = larcv.as_ndarray_bbox(mask)
            new_mask = mask_bin_arr.astype(np.uint8).copy()
            # opencv 3.2
            mask_new, contours, hierarchy = cv2.findContours((new_mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # n=0
            polygon_list = []
            for contour in contours:
                # n=n+1
                # print('Contour # ', n)
                contour = contour.flatten().tolist()
                if len(contour) >= 6:
                    this_poly =[]
                    for i in range(len(contour)):
                        if i%2==0:
                            #index is even, x coord
                            this_poly.append(float(contour[i]+mask_box_arr[0]))
                        elif i%2==1:
                            #index is odd, y coord
                            this_poly.append(float(contour[i]+mask_box_arr[1]))
                        else:
                            #index isn't even or oddself.
                            assert 1==2
                    polygon_list.append(this_poly)
            # if len(polygon_list) == 0:
            #     #Nothing in this segment don't include
            #     # print('idx inside: ',idx)
            #     # delete_rows.append(idx)
            #     # print('idx inside: ',idx)
            #     continue
            # print(polygon_list)
            obj['segmentation'] = polygon_list
            obj['area'] = np.sum(mask_bin_arr)
            if obj['area'] < cfg.TRAIN.GT_MIN_AREA:
                continue

            # Convert form (x1, y1, w, h) to (x1, y1, x2, y2)
            x1, y1, x2, y2 = box_utils.xywh_to_xyxy([mask_box_arr[0], mask_box_arr[1], mask_box_arr[2], mask_box_arr[3]])
            x1, y1, x2, y2 = box_utils.clip_xyxy_to_image(
                x1, y1, x2, y2, height, width
            )
            # Require non-zero seg area and more than 1x1 box size
            if obj['area'] > 0 and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                obj['mask'] = mask
                obj['iscrowd'] = 0
                valid_objs.append(obj)
                valid_segms.append(polygon_list)

        num_valid_objs = len(valid_objs)

        boxes = np.zeros((num_valid_objs, 4), dtype=entry['boxes'].dtype)
        max_iou = np.zeros((num_valid_objs), dtype=entry['max_iou'].dtype)
        gt_classes = np.zeros((num_valid_objs), dtype=entry['gt_classes'].dtype)
        gt_overlaps = np.zeros(
            (num_valid_objs, self.num_classes),
            dtype=entry['gt_overlaps'].dtype
        )
        seg_areas = np.zeros((num_valid_objs), dtype=entry['seg_areas'].dtype)
        is_crowd = np.zeros((num_valid_objs), dtype=entry['is_crowd'].dtype)
        box_to_gt_ind_map = np.zeros(
            (num_valid_objs), dtype=entry['box_to_gt_ind_map'].dtype
        )
        #This should be None always for us
        if self.keypoints is not None:
            gt_keypoints = np.zeros(
                (num_valid_objs, 3, self.num_keypoints),
                dtype=entry['gt_keypoints'].dtype
            )

        im_has_visible_keypoints = False
        for ix, obj in enumerate(valid_objs):
            cls = int(larcv.as_ndarray_bbox(obj['mask'])[4])

            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            seg_areas[ix] = obj['area']
            is_crowd[ix] = obj['iscrowd']
            box_to_gt_ind_map[ix] = ix
            #Again this keypoints shouldn't happen
            if self.keypoints is not None:
                gt_keypoints[ix, :, :] = self._get_gt_keypoints(obj)
                if np.sum(gt_keypoints[ix, 2, :]) > 0:
                    im_has_visible_keypoints = True
            #This should also not trigger, we are using iscrowd =0
            if obj['iscrowd']:
                # Set overlap to -1 for all classes for crowd objects
                # so they will be excluded during training
                gt_overlaps[ix, :] = -1.0
            else:
                gt_overlaps[ix, cls] = 1.0

        entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)

        entry_ious = get_ious(entry['boxes'])
        if len(entry['boxes']) > 1:
            max_iou = entry_ious.max()
        else:
            max_iou = 0.0

        entry['max_iou'] = np.append(entry['max_iou'], max_iou)


        entry['segms'].extend(valid_segms)
        # To match the original implementation:
        # entry['boxes'] = np.append(
        #     entry['boxes'], boxes.astype(np.int).astype(np.float), axis=0)
        entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
        entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'], box_to_gt_ind_map
        )
        if self.keypoints is not None:
            entry['gt_keypoints'] = np.append(
                entry['gt_keypoints'], gt_keypoints, axis=0
            )
            entry['has_visible_keypoints'] = im_has_visible_keypoints

    def _add_gt_from_cache(self, roidb, cache_filepath):
        """Add ground truth annotation metadata from cached file."""
        logger.info('Loading cached gt_roidb from %s', cache_filepath)
        with open(cache_filepath, 'rb') as fp:
            cached_roidb = pickle.load(fp)

        # print(len(roidb), len(cached_roidb))
        assert len(roidb) == len(cached_roidb)

        for entry, cached_entry in zip(roidb, cached_roidb):
            values = [cached_entry[key] for key in self.valid_cached_keys]
            boxes, segms, gt_classes, seg_areas, gt_overlaps, is_crowd, \
                box_to_gt_ind_map, width, height, image, id, plane, flipped, chain_adc, \
                chain_cluster, max_iou = values[:16]
            if self.keypoints is not None:
                gt_keypoints, has_visible_keypoints = values[16:]
            # entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
            # entry['segms'].extend(segms)
            # # To match the original implementation:
            # # entry['boxes'] = np.append(
            # #     entry['boxes'], boxes.astype(np.int).astype(np.float), axis=0)
            # entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
            # entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
            # entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)
            # entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
            # entry['box_to_gt_ind_map'] = np.append(entry['box_to_gt_ind_map'], box_to_gt_ind_map
            entry['flipped'] = flipped
            entry['plane'] = plane
            entry['id'] = id
            entry['image'] = image
            entry['width'] = width
            entry['height'] = height
            entry['boxes'] = boxes
            entry['segms'] = segms
            # To match the original implementation:
            # entry['boxes'] = np.append(
            #     entry['boxes'], boxes.astype(np.int).astype(np.float), axis=0)
            entry['gt_classes'] = gt_classes
            entry['seg_areas'] = seg_areas
            entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)
            entry['is_crowd'] = is_crowd
            entry['box_to_gt_ind_map'] = box_to_gt_ind_map
            entry['chain_adc'] = chain_adc
            entry['chain_cluster'] = chain_cluster
            entry['max_iou'] = max_iou
            if self.keypoints is not None:
                entry['gt_keypoints'] = gt_keypoints
                entry['has_visible_keypoints'] = has_visible_keypoints


    def _add_proposals_from_file(
        self, roidb, proposal_file, min_proposal_size, top_k, crowd_thresh
    ):
        """Add proposals from a proposals file to an roidb."""
        logger.info('Loading proposals from: {}'.format(proposal_file))
        with open(proposal_file, 'r') as f:
            proposals = pickle.load(f)
        id_field = 'indexes' if 'indexes' in proposals else 'ids'  # compat fix
        _sort_proposals(proposals, id_field)
        box_list = []
        for i, entry in enumerate(roidb):
            if i % 2500 == 0:
                logger.info(' {:d}/{:d}'.format(i + 1, len(roidb)))
            boxes = proposals['boxes'][i]
            # Sanity check that these boxes are for the correct image id
            assert entry['id'] == proposals[id_field][i]
            # Remove duplicate boxes and very small boxes and then take top k
            boxes = box_utils.clip_boxes_to_image(
                boxes, entry['height'], entry['width']
            )
            keep = box_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = box_utils.filter_small_boxes(boxes, min_proposal_size)
            boxes = boxes[keep, :]
            if top_k > 0:
                boxes = boxes[:top_k, :]
            box_list.append(boxes)
        _merge_proposal_boxes_into_roidb(roidb, box_list)
        if crowd_thresh > 0:
            _filter_crowd_proposals(roidb, crowd_thresh)

    # def _init_keypoints(self):
    #     """Initialize COCO keypoint information."""
    #     self.keypoints = None
    #     self.keypoint_flip_map = None
    #     self.keypoints_to_id_map = None
    #     self.num_keypoints = 0
    #     # Thus far only the 'person' category has keypoints
    #     if 'person' in self.category_to_id_map:
    #         cat_info = self.COCO.loadCats([self.category_to_id_map['person']])
    #     else:
    #         return
    #
    #     # Check if the annotations contain keypoint data or not
    #     if 'keypoints' in cat_info[0]:
    #         keypoints = cat_info[0]['keypoints']
    #         self.keypoints_to_id_map = dict(
    #             zip(keypoints, range(len(keypoints))))
    #         self.keypoints = keypoints
    #         self.num_keypoints = len(keypoints)
    #         if cfg.KRCNN.NUM_KEYPOINTS != -1:
    #             assert cfg.KRCNN.NUM_KEYPOINTS == self.num_keypoints, \
    #                 "number of keypoints should equal when using multiple datasets"
    #         else:
    #             cfg.KRCNN.NUM_KEYPOINTS = self.num_keypoints
    #         self.keypoint_flip_map = {
    #             'left_eye': 'right_eye',
    #             'left_ear': 'right_ear',
    #             'left_shoulder': 'right_shoulder',
    #             'left_elbow': 'right_elbow',
    #             'left_wrist': 'right_wrist',
    #             'left_hip': 'right_hip',
    #             'left_knee': 'right_knee',
    #             'left_ankle': 'right_ankle'}

    def _get_gt_keypoints(self, obj):
        """Return ground truth keypoints."""
        if 'keypoints' not in obj:
            return None
        kp = np.array(obj['keypoints'])
        x = kp[0::3]  # 0-indexed x coordinates
        y = kp[1::3]  # 0-indexed y coordinates
        # 0: not labeled; 1: labeled, not inside mask;
        # 2: labeled and inside mask
        v = kp[2::3]
        num_keypoints = len(obj['keypoints']) / 3
        assert num_keypoints == self.num_keypoints
        gt_kps = np.ones((3, self.num_keypoints), dtype=np.int32)
        for i in range(self.num_keypoints):
            gt_kps[0, i] = x[i]
            gt_kps[1, i] = y[i]
            gt_kps[2, i] = v[i]
        return gt_kps
    ############################################################################################################
    ############################################################################################################
    #From this point on we're josh's additions purely to bridge the gap
    ############################################################################################################
    ############################################################################################################
    # def __len__(self):
    #     if not self.loadallinmem:
    #         return int(self.io.fetch_n_entries())
    #     else:
    #         return int(self.alldata[self.datalist[0]].shape[0])
    #
    # def __getitem__(self, idx):
    #     if not self.loadallinmem:
    #         #self.io.next(store_event_ids=self.store_eventids)
    #         self.io.next()
    #         out = {}
    #         for dtype,name in zip(self.dtypelist,self.datalist):
    #             out[name] = self.io.fetch_data(name).data()
    #
    #         if self.store_eventids:
    #             out["event_ids"] = self.io.fetch_event_ids()
    #     else:
    #         indices = np.random.randint(len(self),size=self.batchsize)
    #         out = {}
    #         for name in self.datalist:
    #             out[name] = np.zeros( (self.batchsize,self.alldata[name].shape[1]), self.alldata[name].dtype )
    #             for n,idx in enumerate(indices):
    #                 out[name][n,:] = self.alldata[name][idx,:]
    #     return out
    #
    # def _loadinmem(self):
    #     """load data into memory"""
    #     nevents = int(self.io.fetch_n_entries())
    #     if self.max_inmem_events>0 and nevents>self.max_inmem_events:
    #         nevents = self.max_inmem_events
    #
    #     print("Attempting to load all ",nevents," into memory. good luck")
    #     start = time.time()
    #
    #     # start threadio
    #     self.start(1)
    #
    #     # get one data element to get shape
    #     self.io.next(store_event_ids=self.store_eventids)
    #     firstout = {}
    #     for name in self.datalist:
    #         firstout[name] = self.io.fetch_data(name).data()
    #         self.alldata = {}
    #     for name in self.datalist:
    #         self.alldata[name] = np.zeros( (nevents,firstout[name].shape[1]), firstout[name].dtype )
    #         self.alldata[name][0] = firstout[name][0,:]
    #     for i in range(1,nevents):
    #         self.io.next(store_event_ids=self.store_eventids)
    #         if i%100==0:
    #             print("loading event %d of %d"%(i,nevents))
    #         for name in self.datalist:
    #             out = self.io.fetch_data(name).data()
    #             self.alldata[name][i,:] = out[0,:]
    #
    #     print("elapsed time to bring data into memory: ",time.time()-start,"sec")
    #
    #     # stop threads. don't need them anymore
    #     self.stop()

    # def __str__(self):
    #     return dumpcfg()

    # def start(self,batchsize):
    #     """exposes larcv_threadio::start which is used to start the thread managers"""
    #     self.batchsize = batchsize
    #     self.io.start_manager(self.batchsize)
    #
    # def stop(self):
    #     """ stops the thread managers"""
    #     self.io.stop_manager()

    # def dumpcfg(self):
    #     """dump the configuration file to a string"""
    #     print(open(self.cfg).read())

    ############################################################################################################
    ############################################################################################################
    #End Josh additions back to maskrcnn
    ############################################################################################################
    ############################################################################################################

def add_proposals(roidb, rois, scales, crowd_thresh):
    """Add proposal boxes (rois) to an roidb that has ground-truth annotations
    but no proposals. If the proposals are not at the original image scale,
    specify the scale factor that separate them in scales.
    """

    box_list = []
    for i in range(len(roidb)):
        inv_im_scale = 1. / scales[i]
        idx = np.where(rois[:, 0] == i)[0]
        box_list.append(rois[idx, 1:] * inv_im_scale)

    _merge_proposal_boxes_into_roidb(roidb, box_list)
    if crowd_thresh > 0:
        _filter_crowd_proposals(roidb, crowd_thresh)
    _add_class_assignments(roidb)


def _merge_proposal_boxes_into_roidb(roidb, box_list):
    """Add proposal boxes to each roidb entry."""
    assert len(box_list) == len(roidb)
    for i, entry in enumerate(roidb):
        boxes = box_list[i]
        num_boxes = boxes.shape[0]
        gt_overlaps = np.zeros(
            (num_boxes, entry['gt_overlaps'].shape[1]),
            dtype=entry['gt_overlaps'].dtype
        )
        box_to_gt_ind_map = -np.ones(
            (num_boxes), dtype=entry['box_to_gt_ind_map'].dtype
        )

        # Note: unlike in other places, here we intentionally include all gt
        # rois, even ones marked as crowd. Boxes that overlap with crowds will
        # be filtered out later (see: _filter_crowd_proposals).
        gt_inds = np.where(entry['gt_classes'] > 0)[0]
        if len(gt_inds) > 0:
            gt_boxes = entry['boxes'][gt_inds, :]
            gt_classes = entry['gt_classes'][gt_inds]
            proposal_to_gt_overlaps = box_utils.bbox_overlaps(
                boxes.astype(dtype=np.float32, copy=False),
                gt_boxes.astype(dtype=np.float32, copy=False)
            )
            # Gt box that overlaps each input box the most
            # (ties are broken arbitrarily by class order)
            argmaxes = proposal_to_gt_overlaps.argmax(axis=1)
            # Amount of that overlap
            maxes = proposal_to_gt_overlaps.max(axis=1)
            # Those boxes with non-zero overlap with gt boxes
            I = np.where(maxes > 0)[0]
            # Record max overlaps with the class of the appropriate gt box



            gt_overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
            box_to_gt_ind_map[I] = gt_inds[argmaxes[I]]
        # print("entry['boxes'].shape",entry['boxes'].shape)
        # print("boxes.astype(entry['boxes'].dtype, copy=False).shape",boxes.astype(entry['boxes'].dtype, copy=False).shape)

        entry['boxes'] = np.append(
            entry['boxes'],
            boxes.astype(entry['boxes'].dtype, copy=False),
            axis=0
        )
        entry['gt_classes'] = np.append(
            entry['gt_classes'],
            np.zeros((num_boxes), dtype=entry['gt_classes'].dtype)
        )
        entry['seg_areas'] = np.append(
            entry['seg_areas'],
            np.zeros((num_boxes), dtype=entry['seg_areas'].dtype)
        )
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(
            entry['is_crowd'],
            np.zeros((num_boxes), dtype=entry['is_crowd'].dtype)
        )
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'],
            box_to_gt_ind_map.astype(
                entry['box_to_gt_ind_map'].dtype, copy=False
            )
        )


def _filter_crowd_proposals(roidb, crowd_thresh):
    """Finds proposals that are inside crowd regions and marks them as
    overlap = -1 with each ground-truth rois, which means they will be excluded
    from training.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        crowd_inds = np.where(entry['is_crowd'] == 1)[0]
        non_gt_inds = np.where(entry['gt_classes'] == 0)[0]
        if len(crowd_inds) == 0 or len(non_gt_inds) == 0:
            continue
        crowd_boxes = box_utils.xyxy_to_xywh(entry['boxes'][crowd_inds, :])
        non_gt_boxes = box_utils.xyxy_to_xywh(entry['boxes'][non_gt_inds, :])
        iscrowd_flags = [int(True)] * len(crowd_inds)
        ious = COCOmask.iou(non_gt_boxes, crowd_boxes, iscrowd_flags)
        bad_inds = np.where(ious.max(axis=1) > crowd_thresh)[0]
        gt_overlaps[non_gt_inds[bad_inds], :] = -1
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)


def _add_class_assignments(roidb):
    """Compute object category assignment for each box associated with each
    roidb entry.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        entry['max_classes'] = max_classes
        entry['max_overlaps'] = max_overlaps
        # sanity checks
        # if max overlap is 0, the class must be background (class 0)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # if max overlap > 0, the class must be a fg class (not class 0)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)


def _sort_proposals(proposals, id_field):
    """Sort proposals by the specified id field."""
    order = np.argsort(proposals[id_field])
    fields_to_sort = ['boxes', id_field, 'scores']
    for k in fields_to_sort:
        proposals[k] = [proposals[k][i] for i in order]

def get_ious(boxes_in):
    """Acquire a list of all the IoUs across the N boxes
    in, returns (N-1)^2 elements in a list"""
    numpy_iou = np.zeros((0), np.float)
    for nbox1 in range(len(boxes_in)-1):
        for nbox2 in range(nbox1+1, len(boxes_in)):
            IOU = IoU(boxes_in[nbox1], boxes_in[nbox2])
            numpy_iou = np.append(numpy_iou, IOU)
    return numpy_iou



def IoU(box1, box2):
    """Return IoU between box1 and box2"""
    assert len(box1) == 4
    assert len(box2) == 4
    iou=0.0
    if box2[2] < box1[0] or box2[3] < box1[1] or box2[0] > box1[2] or box2[1]  > box1[3]:
        return iou

    box1_area = ( box1[2]+1 - box1[0] ) * ( box1[3]+1 - box1[1] )
    box2_area = ( box2[2]+1 - box2[0] ) * ( box2[3]+1 - box2[1] )
    overlap_minx = max(box1[0], box2[0])
    overlap_maxx = min(box1[2], box2[2])
    overlap_miny = max(box1[1], box2[1])
    overlap_maxy = min(box1[3], box2[3])
    overlap_area = float(( overlap_maxx+1 - overlap_minx ) * ( overlap_maxy+1 - overlap_miny ))
    union_area = float(box1_area + box2_area - overlap_area)
    iou = overlap_area/union_area
    return iou

def _cull_roidb_iou(roidb, thresh=0.0):
    """This function takes in the roidb as a list of dictionaries
    and a threshold for the iou to be above, and cuts the roidb to only
    include entries where the max_iou is above the threshhold."""
    if thresh == 0.0:
        return roidb
    new_roidb =[]
    num_above_1 =0
    num_above_2 =0
    num_above_3 =0
    num_above_4 =0
    num_above_5 =0
    num_above_6 =0
    num_above_7 =0
    num_above_8 =0
    num_above_9 =0
    for entry in roidb:
        if entry['max_iou'] >= thresh:
            new_roidb.append(entry)
        if entry['max_iou'] >= .1:
            num_above_1 = num_above_1 +1
            if entry['max_iou'] >= .2:
                num_above_2 = num_above_2 +1
                if entry['max_iou'] >= .3:
                    num_above_3 = num_above_3 +1
                    if entry['max_iou'] >= .4:
                        num_above_4 = num_above_4 +1
                        if entry['max_iou'] >= .5:
                            num_above_5 = num_above_5 +1
                            if entry['max_iou'] >= .6:
                                num_above_6 = num_above_6 +1
                                if entry['max_iou'] >= .7:
                                    num_above_7 = num_above_7 +1
                                    if entry['max_iou'] >= .8:
                                        num_above_8 = num_above_8 +1
                                        if entry['max_iou'] >= .9:
                                            num_above_9 = num_above_9 +1
    print("Threshold Counts:")
    print('----------------')
    print()
    print('Above 0.1: ',num_above_1)
    print('----------------')
    print()
    print('Above 0.2: ',num_above_2)
    print('----------------')
    print()
    print('Above 0.3: ',num_above_3)
    print('----------------')
    print()
    print('Above 0.4: ',num_above_4)
    print('----------------')
    print()
    print('Above 0.5: ',num_above_5)
    print('----------------')
    print()
    print('Above 0.6: ',num_above_6)
    print('----------------')
    print()
    print('Above 0.7: ',num_above_7)
    print('----------------')
    print()
    print('Above 0.8: ',num_above_8)
    print('----------------')
    print()
    print('Above 0.9: ',num_above_9)


    return new_roidb

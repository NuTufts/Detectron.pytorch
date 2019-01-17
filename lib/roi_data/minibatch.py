import numpy as np
import cv2

from core.config import cfg
import utils.blob as blob_utils
import roi_data.rpn

#larcvdataset imports:
import os,time
import ROOT
from larcv import larcv
import numpy as np
from torch.utils.data import Dataset

#to vis
import datasets.dummy_datasets as datasets
import numpy as np
import utils.vis as vis_utils
import cv2



def get_minibatch_blob_names(is_training=True):
    """Return blob names in the order in which they are read by the data loader.
    """
    # data blob: holds a batch of N images, each with 3 channels
    blob_names = ['data']
    if cfg.RPN.RPN_ON:
        # RPN-only or end-to-end Faster R-CNN
        blob_names += roi_data.rpn.get_rpn_blob_names(is_training=is_training)
    elif cfg.RETINANET.RETINANET_ON:
        raise NotImplementedError
    else:
        # Fast R-CNN like models trained on precomputed proposals
        blob_names += roi_data.fast_rcnn.get_fast_rcnn_blob_names(
            is_training=is_training
        )
    return blob_names


def get_minibatch(roidb):
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    blobs = {k: [] for k in get_minibatch_blob_names()}

    # Get the input image blob
    im_blob, im_scales = _get_image_blob(roidb)
    blobs['data'] = im_blob
    if cfg.RPN.RPN_ON:
        # RPN-only or end-to-end Faster/Mask R-CNN
        valid = roi_data.rpn.add_rpn_blobs(blobs, im_scales, roidb)
    elif cfg.RETINANET.RETINANET_ON:
        raise NotImplementedError
    else:
        # Fast R-CNN like models trained on precomputed proposals
        valid = roi_data.fast_rcnn.add_fast_rcnn_blobs(blobs, im_scales, roidb)
    return blobs, valid


def _get_image_blob(roidb):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images)
    processed_ims = []
    im_scales = []

    # image2d_adc_crop_chain = ROOT.TChain("image2d_adc_tree")
    # image2d_adc_crop_chain.AddFile(roidb[i]['image'])
    # for k,v in roidb[0].items():
    #     print('key', k)
    for i in range(num_images):
        #for root files:
        image2d_adc_crop_chain = roidb[i]['chain_adc']
        image2d_adc_crop_chain.GetEntry(roidb[i]['id'])
        entry_image2dadc_crop_data = image2d_adc_crop_chain.image2d_adc_branch
        image2dadc_crop_array = entry_image2dadc_crop_data.as_vector()
        im_2d = larcv.as_ndarray(image2dadc_crop_array[roidb[i]['plane']])
        im = np.zeros ((roidb[i]['height'],roidb[i]['width'],3))

        
        for dim1 in range(len(im_2d)):
            for dim2 in range(len(im_2d[0])):
                im[dim1][dim2][0] = im_2d[dim1][dim2]
                im[dim1][dim2][1] = im_2d[dim1][dim2]
                im[dim1][dim2][2] = im_2d[dim1][dim2]
        # if cfg.TRAIN.MAKE_IMAGES:
        #     boxes = np.array([[50,50,60,60,.99],[1,1,5,5,.99]])
        #
        #     im_numpy = im
        #     # im_numpy = np.swapaxes(im_numpy,2,1)
        #     # im_numpy = np.swapaxes(im_numpy,2,0)
        #     im_numpy[im_numpy>0] = 100
        #     im_numpy[im_numpy<=0] =0
        #     vis_utils.vis_one_image(
        #         im_numpy,
        #         'GoodImage'+str(i),
        #         'hmmm/',
        #         boxes,
        #         None,
        #         None,
        #         dataset=datasets.get_particle_dataset(),
        #         box_alpha=0.3,
        #         show_class=True,
        #         thresh=0.7,
        #         kp_thresh=2,
        #         plain_img=True
        #     )

        # #for jpgs:
        # im = cv2.imread(roidb[i]['image'])
        assert im is not None, \
            'Failed to read image \'{}\''.format(roidb[i]['image'])
        # If NOT using opencv to read in images, uncomment following lines
        # if len(im.shape) == 2:
        #     im = im[:, :, np.newaxis]
        #     im = np.concatenate((im, im, im), axis=2)
        # # flip the channel, since the original one using cv2
        # # rgb -> bgr
        # im = im[:, :, ::-1]
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        target_size = cfg.TRAIN.SCALES[scale_inds[i]]

        im, im_scale = blob_utils.prep_im_for_blob(
            im, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale[0])
        processed_ims.append(im[0])

    # Create a blob to hold the input images [n, c, h, w]
    blob = blob_utils.im_list_to_blob(processed_ims)


    return blob, im_scales

import random
random.seed(42)
import os
import pickle
import numpy
from copy import deepcopy
import cv2

from superpixel import get_superpixel_labels
from reader import get_generators
from utilities import get_flair_file_names, traintestvalid_split5,\
crop_resample_img, normalize, show_an_image_slice, overlay_grid
from config import *

def inpaint_one(flair_img, t1ce_img, seg_img, superpixels, debug):
    min_area = None
    labels = None
    repl_factor = None
    if superpixels:
        labels = get_superpixel_labels(flair_img)
        min_area = MIN_AREA_SUPERPIXEL
        repl_factor = REPLICATION_FACTOR_FOR_NO_ROI_SUPERPIXEL_INPAINTING
    else:
        labels = overlay_grid(flair_img.shape)
        min_area = MIN_AREA_GRID
        repl_factor = REPLICATION_FACTOR_FOR_NO_ROI_GRID_INPAINTING

    if debug:
        show_an_image_slice(labels, "label grid")

    selected_labels = None
    unique_lab = None
    if seg_img is not None: # this means that we have the segmentation map, and should impose ROI
        selected_labels = labels*seg_img
        unique_lab = numpy.unique(selected_labels)
    else: # this means we do not have ROI, so we select random superpixelized regions but towards the center of the image
        selected_labels = labels
        unique_lab = numpy.unique(selected_labels[60:100, 90:130, :])
        random.shuffle(unique_lab) 

    if debug:
        show_an_image_slice(selected_labels, "selected labels")

    for idx, i in enumerate(unique_lab):
        if i==0:
            continue
        selected_labels_copy = deepcopy(selected_labels)
        selected_labels_copy[selected_labels_copy!=i] = 1
        selected_labels_copy[selected_labels_copy==i] = 0

        if (len(numpy.where(selected_labels_copy.flatten() == 0)[0])) > min_area:
            flair_img_inp = deepcopy(flair_img)
            t1ce_img_inp = deepcopy(t1ce_img)
            flair_img_inp = flair_img_inp*selected_labels_copy
            t1ce_img_inp = t1ce_img_inp*selected_labels_copy

            if debug:
                show_an_image_slice(flair_img_inp, "flair inpaint")
                show_an_image_slice(t1ce_img_inp, "t1ce inpaint")

            yield flair_img_inp, t1ce_img_inp

        # keep only the needed regions
        # otherwise for non-ROI driven inpainting, we will have a large amount of regions
        # corresponding to all of them
        if seg_img is None and idx >= repl_factor:
            break

def get_inpainter_generators(batch_size = 1, split_number = 1, debug = False, to_count = False, with_roi = True, with_superpixel = True):
    train_gen, _, valid_gen = get_generators(batch_size = 1, split_number = split_number, to_count = to_count)
    return __get_generator(train_gen, batch_size, debug, with_roi, with_superpixel),\
    __get_generator(valid_gen, batch_size, debug, with_roi, with_superpixel)

def __get_generator(generator, batch_size, debug, with_roi, with_superpixel):
    batched_overall_input = []
    batched_overall_output = []
    for overall, seg in generator:
        flair = overall[0:1, :, :, :, 0:1]
        t1ce = overall[0:1, :, :, :, 1:2]

        s = flair.shape
        flair_3d_shape = numpy.reshape(flair, (s[1], s[2], s[3]))
        t1ce_3d_shape = numpy.reshape(t1ce, (s[1], s[2], s[3]))
        seg_3d_shape = None
        if with_roi:
            seg_3d_shape = numpy.reshape(seg, (s[1], s[2], s[3]))

        for flair_img_inp, t1ce_img_inp in inpaint_one(flair_3d_shape, t1ce_3d_shape, seg_3d_shape, with_superpixel, debug):

            flair_img_inp = numpy.reshape(flair_img_inp, s)
            t1ce_img_inp = numpy.reshape(t1ce_img_inp, s)

            overall_input = numpy.concatenate((flair_img_inp, t1ce_img_inp), axis = -1)
            overall_output = numpy.concatenate((flair, t1ce), axis = -1)

            batched_overall_input.append(overall_input)
            batched_overall_output.append(overall_output)

            if len(batched_overall_input) == batch_size:
                yield numpy.concatenate(batched_overall_input, axis = 0), numpy.concatenate(batched_overall_output, axis = 0)
                batched_overall_input = []
                batched_overall_output = []

    if len(batched_overall_input) != 0:
        yield numpy.concatenate(batched_overall_input, axis = 0), numpy.concatenate(batched_overall_output, axis = 0)

if __name__ == "__main__":
    from tqdm import tqdm

    train, valid = get_inpainter_generators(1, 1, True, True, False, False)

    count = 0
    for inpt, outpt in tqdm(train):
        count += 1

    print (count)
    #1005 -> ROI
    #1052 -> no ROI
    #964 -> ROI + grid
    #790 -> no ROI + grid

    count = 0
    for inpt, outpt in tqdm(valid):
        count += 1

    print (count)
    #149 -> ROI
    #188 -> no ROI
    #148 -> ROI + grid
    #153 -> no ROI + grid
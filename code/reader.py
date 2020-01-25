"""
Reads the original 3D scans!
subhradeep.kayal@gmail.com
Erasmus MC
"""
import nibabel as nib
import numpy
import os

from utilities import get_flair_file_names, traintestvalid_split5, normalize, crop_resample_img
from config import *

def get_scan(filepath):
    img = nib.load(filepath).get_fdata()
    return img

def get_generators(batch_size = 1, split_number = 1):
    flair_names, seg_names = get_flair_file_names()
    train_flair_names, test_flair_names, valid_flair_names,\
    train_seg_names, test_seg_names, valid_seg_names = traintestvalid_split5(flair_names, seg_names)[split_number]

    return __get_generator(train_flair_names, train_seg_names, batch_size),\
           __get_generator(test_flair_names, test_seg_names, 1, True),\
           __get_generator(valid_flair_names, valid_seg_names, batch_size)

def __get_generator(flair_names, seg_names, batch_size, is_test = False):
    while True:
        batched_overall = []
        batched_seg = []
        for flair, seg in zip(flair_names, seg_names):
            flair_img = get_scan(os.path.join(BRATS_PARENT, flair))
            t1ce = flair.replace("flair", "t1ce")
            t1ce_img = get_scan(os.path.join(BRATS_PARENT, t1ce))
            flair_img = crop_resample_img(normalize(flair_img))
            t1ce_img = crop_resample_img(normalize(t1ce_img))

            seg_img = get_scan(os.path.join(BRATS_PARENT, seg))
            seg_img = crop_resample_img(seg_img)
            seg_img[seg_img >= 1] = 1

            s = flair_img.shape
            flair_img = numpy.reshape(flair_img, (1, s[0], s[1], s[2], 1))
            t1ce_img = numpy.reshape(t1ce_img, (1, s[0], s[1], s[2], 1))
            overall = numpy.concatenate((flair_img, t1ce_img), axis = -1)
            seg_img = numpy.reshape(seg_img, (1, s[0], s[1], s[2], 1))

            batched_overall.append(overall)
            batched_seg.append(seg_img)

            if len(batched_overall) == batch_size:
                yield numpy.concatenate(batched_overall, axis = 0), numpy.concatenate(batched_seg, axis = 0)
                batched_overall = []
                batched_seg = []

        if len(batched_overall) != 0:
            yield numpy.concatenate(batched_overall, axis = 0), numpy.concatenate(batched_seg, axis = 0)

        if is_test:
            break

if __name__ == "__main__":
    fpath = "/home/kayald/Code/inpainting-pretraining/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_2013_10_1/Brats18_2013_10_1_seg.nii.gz"
    img = get_scan(fpath)
    for i in range(img.shape[2]):
        print (i, numpy.unique(img[:, :, i]))
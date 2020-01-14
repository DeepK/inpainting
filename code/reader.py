"""
Reads the original 3D scans!
subhradeep.kayal@gmail.com
Erasmus MC
"""
import nibabel as nib
import numpy
import cv2

import os
from utilities import get_flair_file_names, traintestvalid_split5

def get_scan(filepath):
    img = nib.load(filepath).get_fdata()
    return img

def get_generators(brats_parent, split_number = 1):
    flair_names, seg_names = get_flair_file_names(brats_parent)
    train_flair_names, test_flair_names, valid_flair_names,\
    train_seg_names, test_seg_names, valid_seg_names = traintestvalid_split5(flair_names, seg_names)[split_number]

    return __get_generator(brats_parent, train_flair_names, train_seg_names),\
           __get_generator(brats_parent, test_flair_names, test_seg_names, True),\
           __get_generator(brats_parent, valid_flair_names, valid_seg_names)

def __get_generator(brats_parent, flair_names, seg_names, is_test = False):
    while True:
        for flair, seg in zip(flair_names, seg_names):
            flair_img = get_scan(os.path.join(brats_parent, flair))
            flair_img = cv2.normalize(flair_img, None, 255.0, 0.0, cv2.NORM_MINMAX)
            flair_img = flair_img/255.0

            t1ce = flair.replace("flair", "t1ce")
            t1ce_img = get_scan(os.path.join(brats_parent, t1ce))
            t1ce_img = cv2.normalize(t1ce_img, None, 255.0, 0.0, cv2.NORM_MINMAX)
            t1ce_img = t1ce_img/255.0

            t2 = flair.replace("flair", "t2")
            t2_img = get_scan(os.path.join(brats_parent, t2))
            t2_img = cv2.normalize(t2_img, None, 255.0, 0.0, cv2.NORM_MINMAX)
            t2_img = t2_img/255.0

            seg_img = get_scan(os.path.join(brats_parent, seg))

            # crop near the center where the brain image is located
            # 160 x 216 x 128
            flair_img = flair_img[40:200, 12:228, 11:139]
            t1ce_img = t1ce_img[40:200, 12:228, 11:139]
            t2_img = t2_img[40:200, 12:228, 11:139]
            seg_img = seg_img[40:200, 12:228, 11:139]
            seg_img[seg_img >= 1] = 1

            s = flair_img.shape
            flair_img = numpy.reshape(flair_img, (1, s[0], s[1], s[2], 1))
            t1ce_img = numpy.reshape(t1ce_img, (1, s[0], s[1], s[2], 1))
            t2_img = numpy.reshape(t2_img, (1, s[0], s[1], s[2], 1))
            overall = numpy.concatenate((flair_img, t1ce_img, t2_img), axis = -1)
            seg_img = numpy.reshape(seg_img, (1, s[0], s[1], s[2], 1))

            yield overall, seg_img

        if is_test:
            break

if __name__ == "__main__":
    fpath = "/home/kayald/Code/inpainting-pretraining/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_2013_10_1/Brats18_2013_10_1_seg.nii.gz"
    img = get_scan(fpath)
    for i in range(img.shape[2]):
        print (i, numpy.unique(img[:, :, i]))
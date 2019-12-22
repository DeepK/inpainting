"""
Reads the original 3D scans!
subhradeep.kayal@gmail.com
Erasmus MC
"""
import nibabel as nib
import numpy
import cv2

import os
from utilities import get_flair_file_names, traintestvalid_split5, windowed_scan_gen_xaxis

def get_scan(filepath):
    img = nib.load(filepath).get_fdata()
    img = cv2.normalize(img, None, 255.0, 0.0, cv2.NORM_MINMAX)
    img = img/255.0
    return img

def get_generators(brats_parent, split_number = 1):
    flair_names, seg_names = get_flair_file_names(brats_parent)
    train_flair_names, test_flair_names, valid_flair_names,\
    train_seg_names, test_seg_names, valid_seg_names = traintestvalid_split5(flair_names, seg_names)[split_number]

    return __get_generator(brats_parent, train_flair_names, train_seg_names),\
           __get_generator(brats_parent, test_flair_names, test_seg_names),\
           __get_generator(brats_parent, valid_flair_names, valid_seg_names)

def __get_generator(brats_parent, flair_names, seg_names):
    while True:
        for flair, seg in zip(flair_names, seg_names):
            flair_img = get_scan(os.path.join(brats_parent, flair))
            seg_img = get_scan(os.path.join(brats_parent, seg))

            # crop near the center where the brain image is located
            # 160 x 160 x 120
            flair_img = flair_img[40:200, 40:200, 15:135]
            seg_img = seg_img[40:200, 40:200, 15:135]

            for tup in windowed_scan_gen_xaxis(flair_img, 80, seg_img):
                flair_img_w = tup[0]
                seg_img_w = tup[1]
                s = flair_img_w.shape
                flair_img_w = numpy.reshape(flair_img_w, (1, s[0], s[1], s[2], 1))
                seg_img_w = numpy.reshape(seg_img_w, (1, s[0], s[1], s[2], 1))

                yield flair_img_w, seg_img_w

if __name__ == "__main__":
    fpath = "/home/kayald/Code/inpainting-pretraining/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_TCIA02_222_1/Brats18_TCIA02_222_1_flair.nii.gz"
    img = get_scan(fpath)
    print (img.shape)
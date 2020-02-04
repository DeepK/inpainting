import random
random.seed(42)
import os
from sklearn.model_selection import train_test_split
import cv2
from matplotlib import pyplot as plt
import numpy
numpy.random.seed(42)

from config import *

def get_flair_file_names():
    names = os.listdir(BRATS_PARENT)
    flair_names = [n + "/" + n + "_flair.nii.gz" for n in names]
    seg_names = [n + "/" + n + "_seg.nii.gz" for n in names]
    return flair_names, seg_names

def traintestvalid_split5(flair_names, seg_names):
    "5 different train-test splits"
    seeds = [1, 2, 3, 4, 5]

    train_test_splits = {}
    for seed in seeds:
        train_flair_names, test_flair_names, train_seg_names, test_seg_names = train_test_split(flair_names, seg_names, test_size=0.15, random_state=seed)
        train_flair_names, valid_flair_names, train_seg_names, valid_seg_names = train_test_split(train_flair_names, train_seg_names, test_size=0.15, random_state=seed)
        train_test_splits[seed] = (train_flair_names,\
                                   test_flair_names,\
                                   valid_flair_names,\
                                   train_seg_names,\
                                   test_seg_names,\
                                   valid_seg_names)

    return train_test_splits

def normalize(img):
    img = cv2.normalize(img, None, 255.0, 0.0, cv2.NORM_MINMAX)
    img = img/255.0
    return img

def crop_resample_img(img):
    return (img[40:200, 12:228, ::4])[:,:,:32]

def show_an_image_slice(img, name):
    fig = plt.figure(name)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img[:, :, 18], interpolation = "nearest")
    plt.axis("off")
    plt.show()

def overlay_grid(img_shape):
    grid_size_random = numpy.random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    label = 1
    labels = numpy.zeros(img_shape)
    for i in range(0, img_shape[0], grid_size_random):
        for j in range(0, img_shape[1], grid_size_random):
            labels[i:i+grid_size_random, j:j+grid_size_random, :] = label
            label += 1
    return labels

if __name__ == "__main__":
    from reader import get_scan
    fpath = "/home/kayald/Code/inpainting-pretraining/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_TCIA02_222_1/Brats18_TCIA02_222_1_flair.nii.gz"
    img = get_scan(fpath)
    
    img = img[40:200, 40:200, :]
    
    for img_tmp in windowed_scan_gen_xaxis(img, 80):
        for i in range(10, 140):
            fig = plt.figure("image")
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(img_tmp[:, :, i], interpolation = "nearest")
            plt.axis("off")
            plt.show()

    flair_names, seg_names = get_flair_file_names()
    train_test_splits = traintestvalid_split5(flair_names, seg_names)
    print (len(train_test_splits[1][0]),\
           len(train_test_splits[1][1]),\
           len(train_test_splits[1][2]),\
           len(train_test_splits[1][3]),\
           len(train_test_splits[1][4]),\
           len(train_test_splits[1][5]))
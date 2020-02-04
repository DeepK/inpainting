"""
SLIC
subhradeep.kayal@gmail.com
Erasmus MC
"""
import os
import pickle

from tqdm import tqdm
from skimage import segmentation

from utilities import get_flair_file_names, normalize, crop_resample_img
from config import *
from reader import get_scan

def get_superpixel_labels(img):
    label = segmentation.slic(img, n_segments = 400, compactness = 0.15, multichannel = False)
    return label

def precalculate_superpixel_labels():
    if os.path.exists(SUPERPIXEL_FOLDER):
        return

    os.makedirs(SUPERPIXEL_FOLDER, exist_ok = True)

    flair_names, seg_names = get_flair_file_names()

    for name in tqdm(flair_names):
        flair_img = get_scan(os.path.join(BRATS_PARENT, name))
        flair_img = crop_resample_img(normalize(flair_img))
        label = get_superpixel_labels(flair_img)

        name = name.split("/")[1].split(".")[0]
        pickle.dump(label, open(os.path.join(SUPERPIXEL_FOLDER, name + ".p"), "wb"))

if __name__ == "__main__":
    from reader import get_scan
    fpath = "/home/kayald/Code/inpainting-pretraining/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_TCIA02_222_1/Brats18_TCIA02_222_1_flair.nii.gz"
    segfpath = "/home/kayald/Code/inpainting-pretraining/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_TCIA02_222_1/Brats18_TCIA02_222_1_seg.nii.gz"
    img = get_scan(fpath)
    seg = get_scan(segfpath)

    from matplotlib import pyplot as plt
    import numpy

    img = crop_resample_img(normalize(img))
    seg = crop_resample_img(normalize(seg))
    label = get_superpixel_labels(img)

    for i in range(32):
        overlayed_img = numpy.maximum(img[:, :, i], seg[:, :, i])
        boundaries = segmentation.mark_boundaries(overlayed_img, label[:, :, i])
        fig = plt.figure("Boundaries")
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(boundaries, interpolation = "nearest")
        plt.axis("off")
        plt.show()
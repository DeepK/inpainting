import os
import pickle
import numpy
from copy import deepcopy
from tqdm import tqdm
import cv2

from superpixel import get_superpixel_labels
from reader import get_scan
from utilities import get_flair_file_names, traintestvalid_split5
from config import *

def precalculate_superpixel_labels():
    if os.path.exists(SUPERPIXEL_FOLDER):
        return

    os.makedirs(SUPERPIXEL_FOLDER, exist_ok = True)

    flair_names, seg_names = get_flair_file_names(BRATS_PARENT)

    for name in tqdm(flair_names):
        flair_img = get_scan(os.path.join(BRATS_PARENT, name))
        label = get_superpixel_labels(flair_img)

        name = name.split("/")[1].split(".")[0]
        pickle.dump(label, open(os.path.join(SUPERPIXEL_FOLDER, name + ".p"), "wb"))

def inpaint_one(flair_img, t1ce_img, seg_img, superpixels, min_area = 5000):
    diseased_superpixels = superpixels*seg_img
    unique_diseased = numpy.unique(diseased_superpixels)

    for i in unique_diseased:
        if i==0:
            continue
        diseased_superpixels_copy = deepcopy(diseased_superpixels)
        diseased_superpixels_copy[diseased_superpixels_copy!=i] = 1
        diseased_superpixels_copy[diseased_superpixels_copy==i] = 0

        if (len(numpy.where(diseased_superpixels_copy.flatten() == 0)[0])) > min_area:
            flair_img_inp = deepcopy(flair_img)
            t1ce_img_inp = deepcopy(t1ce_img)
            flair_img_inp = flair_img_inp*diseased_superpixels_copy
            t1ce_img_inp = t1ce_img_inp*diseased_superpixels_copy
            yield flair_img, flair_img_inp, t1ce_img, t1ce_img_inp

def get_generators(batch_size = 1, split_number = 1):
    flair_names, seg_names = get_flair_file_names()
    train_flair_names, test_flair_names, valid_flair_names,\
    train_seg_names, test_seg_names, valid_seg_names = traintestvalid_split5(flair_names, seg_names)[split_number]

    return __get_generator(train_flair_names, train_seg_names, batch_size),\
           __get_generator(valid_flair_names, valid_seg_names, batch_size)

def __get_generator(flair_names, seg_names, batch_size, for_nn = True):
    while True:
        batched_overall = []
        batched_seg = []
        for flair_name, seg_name in zip(flair_names, seg_names):
            flair_img = get_scan(os.path.join(BRATS_PARENT, flair))
            t1ce = flair.replace("flair", "t1ce")
            t1ce_img = get_scan(os.path.join(BRATS_PARENT, t1ce))
            seg_img = get_scan(os.path.join(BRATS_PARENT, seg_name))

            superpixel_filename = flair_name.split("/")[1].split(".")[0]
            superpixel = pickle.load(open(os.path.join("../superpixels", superpixel_filename + ".p"), "rb"))

            for flair_img, flair_img_inp, t1ce_img, t1ce_img_inp in inpaint_one(flair_img, t1ce_img, seg_img, superpixel):
                flair_img = crop_resample_img(normalize(flair_img))
                t1ce_img = crop_resample_img(normalize(t1ce_img))
                flair_img_inp = crop_resample_img(normalize(flair_img_inp))
                t1ce_img_inp = crop_resample_img(normalize(t1ce_img_inp))

                s = flair_img.shape
                flair_img = numpy.reshape(flair_img, (1, s[0], s[1], s[2], 1))
                flair_img_inp = numpy.reshape(flair_img_inp, (1, s[0], s[1], s[2], 1))
                t1ce_img = numpy.reshape(t1ce_img, (1, s[0], s[1], s[2], 1))
                t1ce_img_inp = numpy.reshape(t1ce_img_inp, (1, s[0], s[1], s[2], 1))

                overall_inp = numpy.concatenate((flair_img, t1ce_img), axis = -1)
                overall_outp = numpy.concatenate((flair_img_inp, t1ce_img_inp), axis = -1)

                batched_overall_inp.append(overall_inp)
                batched_overall_outp.append(overall_outp)

                if len(batched_overall_inp) == batch_size:
                    yield numpy.concatenate(batched_overall_inp, axis = 0), numpy.concatenate(batched_overall_outp, axis = 0)
                    batched_overall_inp = []
                    batched_overall_outp = []

        if len(batched_overall) != 0:
            yield numpy.concatenate(batched_overall_inp, axis = 0), numpy.concatenate(batched_overall_outp, axis = 0)

        # a backdoor, to break out of the while True
        # in case we are not using this for keras fit_generator
        if not for_nn:
            break

if __name__ == "__main__":
    pass
    #precalculate_superpixel_labels()

    #from matplotlib import pyplot as plt

    #BRATS_PARENT = "/home/kayald/Code/inpainting-pretraining/MICCAI_BraTS_2018_Data_Training/HGG/"
    #flair_names, seg_names = get_flair_file_names(BRATS_PARENT)
    #train_flair_names, test_flair_names, valid_flair_names,\
    #train_seg_names, test_seg_names, valid_seg_names = traintestvalid_split5(flair_names, seg_names)[1]

    #counter_train = 0
    #for tup in inpainter_generator(BRATS_PARENT, train_flair_names, train_seg_names, for_nn = False):
        #inp = tup[0]
        #inp = numpy.reshape(inp, (80, 216, 128))

        #for i in range(100):
        #    fig = plt.figure("Inpainted")
        #    ax = fig.add_subplot(1, 1, 1)
        #    ax.imshow(inp[:,:,i], interpolation = "nearest")
        #    plt.axis("off")
        #    plt.show()
    #    counter_train+=1
    #print (counter_train)
    # 1584

    #counter_valid = 0
    #for tup in inpainter_generator(BRATS_PARENT, valid_flair_names, valid_seg_names, for_nn = False):
    #    counter_valid+=1
    #print (counter_valid)
    # 244
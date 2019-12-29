from superpixel import get_superpixel_labels

import os
from reader import get_scan
from utilities import get_flair_file_names, windowed_scan_gen_xaxis, traintestvalid_split5
import pickle
import numpy
from copy import deepcopy
from tqdm import tqdm
import cv2

def precalculate_superpixel_labels():
    os.makedirs("../superpixels/", exist_ok = True)

    brats_parent = "/home/kayald/Code/inpainting-pretraining/MICCAI_BraTS_2018_Data_Training/HGG/"
    flair_names, seg_names = get_flair_file_names(brats_parent)

    for name in tqdm(flair_names):
        flair_img = get_scan(os.path.join(brats_parent, name))
        label = get_superpixel_labels(flair_img)

        name = name.split("/")[1].split(".")[0]
        pickle.dump(label, open(os.path.join("../superpixels", name + ".p"), "wb"))

def inpaint_one(flair_img, seg_img, superpixels, min_area = 5000):
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
            flair_img_inp = flair_img_inp*diseased_superpixels_copy
            yield flair_img[40:200, 12:228, 11:139], flair_img_inp[40:200, 12:228, 11:139]

def get_generators(brats_parent, split_number):
    flair_names, seg_names = get_flair_file_names(brats_parent)
    train_flair_names, test_flair_names, valid_flair_names,\
    train_seg_names, test_seg_names, valid_seg_names = traintestvalid_split5(flair_names, seg_names)[split_number]

    return inpainter_generator(brats_parent, train_flair_names, train_seg_names),\
           inpainter_generator(brats_parent, valid_flair_names, valid_seg_names)


def inpainter_generator(brats_parent, flair_names, seg_names, for_nn = True):
    while True:
        for flair_name, seg_name in zip(flair_names, seg_names):
            flair_img = get_scan(os.path.join(brats_parent, flair_name))
            flair_img = cv2.normalize(flair_img, None, 255.0, 0.0, cv2.NORM_MINMAX)
            flair_img = flair_img/255.0
            seg_img = get_scan(os.path.join(brats_parent, seg_name))

            superpixel_filename = flair_name.split("/")[1].split(".")[0]
            superpixel = pickle.load(open(os.path.join("../superpixels", superpixel_filename + ".p"), "rb"))

            for flair_centered, flair_inp in inpaint_one(flair_img, seg_img, superpixel):
                for tup in windowed_scan_gen_xaxis(flair_centered, 80, flair_inp):
                    flair_img_w = tup[0]
                    inpainted_img_w = tup[1]
                    s = flair_img_w.shape
                    flair_img_w = numpy.reshape(flair_img_w, (1, s[0], s[1], s[2], 1))
                    inpainted_img_w = numpy.reshape(inpainted_img_w, (1, s[0], s[1], s[2], 1))

                    yield inpainted_img_w, flair_img_w

        # a backdoor, to break out of the while True
        # in case we are not using this for keras fit_generator
        if not for_nn:
            break

if __name__ == "__main__":
    pass
    #precalculate_superpixel_labels()

    #from matplotlib import pyplot as plt

    #brats_parent = "/home/kayald/Code/inpainting-pretraining/MICCAI_BraTS_2018_Data_Training/HGG/"
    #flair_names, seg_names = get_flair_file_names(brats_parent)
    #train_flair_names, test_flair_names, valid_flair_names,\
    #train_seg_names, test_seg_names, valid_seg_names = traintestvalid_split5(flair_names, seg_names)[1]

    #counter_train = 0
    #for tup in inpainter_generator(brats_parent, train_flair_names, train_seg_names, for_nn = False):
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
    #for tup in inpainter_generator(brats_parent, valid_flair_names, valid_seg_names, for_nn = False):
    #    counter_valid+=1
    #print (counter_valid)
    # 244
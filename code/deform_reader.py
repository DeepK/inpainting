import numpy, elasticdeform
numpy.random.seed(42)
import random
random.seed(42)
from tqdm import tqdm
import os
import cv2
import pickle

from reader import get_scan
from utilities import get_flair_file_names, windowed_scan_gen_xaxis, traintestvalid_split5

def precalculate_deformed_img():
    os.makedirs("../deformed/", exist_ok = True)

    brats_parent = "/home/kayald/Code/inpainting-pretraining/MICCAI_BraTS_2018_Data_Training/HGG/"
    flair_names, seg_names = get_flair_file_names(brats_parent)

    for name in tqdm(flair_names):
        flair_img = get_scan(os.path.join(brats_parent, name))
        flair_img = cv2.normalize(flair_img, None, 255.0, 0.0, cv2.NORM_MINMAX)
        flair_img = flair_img/255.0
        flair_img = flair_img[40:200, 12:228, 11:139]
        
        for idx, flair_w_img in enumerate(windowed_scan_gen_xaxis(flair_img, 80, extra = None)):

            for sigma, points in [(2, 3), (3, 3), (5, 3), (2, 5), (3, 5)]:
                flair_w_img_deformed = elasticdeform.deform_random_grid(flair_w_img, sigma=sigma, points=points, axis=(0, 1))
                
                name_spl = name.split("/")[1].split(".")[0]
                pickle.dump([flair_w_img, flair_w_img_deformed], open(os.path.join("../deformed", name_spl + "-%s-%s-%s.p"%(idx, sigma, points)), "wb"))

def deform_generator(flair_names, for_nn = True):
    while True:
        for flair_name in flair_names:
            name_spl = flair_name.split("/")[1].split(".")[0]

            for idx, sigma, points in [(0, 2, 3), (0, 3, 3), (0, 5, 3), (0, 2, 5), (0, 3, 5),\
                                        (1, 2, 3), (1, 3, 3), (1, 5, 3), (1, 2, 5), (1, 3, 5)]:
                flair_img_w, flair_w_img_deformed = \
                pickle.load(open(os.path.join("../deformed", name_spl + "-%s-%s-%s.p"%(idx, sigma, points)), "rb"))

                s = flair_img_w.shape
                flair_img_w = numpy.reshape(flair_img_w, (1, s[0], s[1], s[2], 1))
                flair_w_img_deformed = numpy.reshape(flair_w_img_deformed, (1, s[0], s[1], s[2], 1))

                yield flair_w_img_deformed, flair_img_w

        # a backdoor, to break out of the while True
        # in case we are not using this for keras fit_generator
        if not for_nn:
            break

def get_generators(brats_parent, split_number):
    flair_names, seg_names = get_flair_file_names(brats_parent)
    train_flair_names, test_flair_names, valid_flair_names,\
    train_seg_names, test_seg_names, valid_seg_names = traintestvalid_split5(flair_names, seg_names)[split_number]

    return deform_generator(train_flair_names),\
           deform_generator(valid_flair_names)

if __name__ == "__main__":
    #pass

    from matplotlib import pyplot as plt

    brats_parent = "/home/kayald/Code/inpainting-pretraining/MICCAI_BraTS_2018_Data_Training/HGG/"
    flair_names, seg_names = get_flair_file_names(brats_parent)
    train_flair_names, test_flair_names, valid_flair_names,\
    train_seg_names, test_seg_names, valid_seg_names = traintestvalid_split5(flair_names, seg_names)[1]

    counter = 0
    for flair_w_img_deformed, flair_w_img in tqdm(deform_generator(train_flair_names, for_nn = False)):
        counter+=1

        flair_w_img_deformed = numpy.reshape(flair_w_img_deformed, (80, 216, 128))
        flair_w_img = numpy.reshape(flair_w_img, (80, 216, 128))

        for i in range(100):
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(flair_w_img[:, :, i], interpolation = "nearest")
            ax[1].imshow(flair_w_img_deformed[:, :, i], interpolation = "nearest")
            plt.show()
    #print (counter)
    # 1510

    #counter = 0
    #for flair_w_img_deformed, flair_w_img in tqdm(deform_generator(valid_flair_names, for_nn = False)):
    #    counter+=1
    #print (counter)
    # 270
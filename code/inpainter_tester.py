import random
random.seed(42)
import os
import numpy

from unet import DICTAVAILNETWORKS3D
from inpaint_reader import get_inpainter_generators
from config import *
from utilities import show_an_image_slice

unet_path = "../models/inpainter_unet_roi_superpixel/"
model = DICTAVAILNETWORKS3D(SIZE_OF_IMAGE, INPAINTER_UNET).getModel()
model.load_weights(os.path.join(unet_path, "weights1.h5"))

train_gen, valid_gen = get_inpainter_generators(batch_size = 1, split_number = 1,\
    with_roi = True, with_superpixel = True)

for inpt, outpt in train_gen:
    pred = model.predict(inpt)

    flair_inpt = inpt[:, :, :, :, 0:1]
    t1_inpt = inpt[:, :, :, :, 1:2]
    s = flair_inpt.shape
    flair_inpt = numpy.reshape(flair_inpt, (s[1], s[2], s[3]))
    t1_inpt = numpy.reshape(t1_inpt, (s[1], s[2], s[3]))

    flair_outpt = outpt[:, :, :, :, 0:1]
    t1_outpt = outpt[:, :, :, :, 1:2]
    flair_outpt = numpy.reshape(flair_outpt, (s[1], s[2], s[3]))
    t1_outpt = numpy.reshape(t1_outpt, (s[1], s[2], s[3]))

    flair_pred = pred[:, :, :, :, 0:1]
    t1_pred = pred[:, :, :, :, 1:2]
    flair_pred = numpy.reshape(flair_pred, (s[1], s[2], s[3]))
    t1_pred = numpy.reshape(t1_pred, (s[1], s[2], s[3]))

    show_an_image_slice(flair_inpt, "flair inpaint")
    show_an_image_slice(t1_inpt, "t1 inpaint")
    show_an_image_slice(flair_outpt, "flair output")
    show_an_image_slice(t1_outpt, "t1 output")
    show_an_image_slice(flair_pred, "flair pred")
    show_an_image_slice(t1_pred, "t1 pred")
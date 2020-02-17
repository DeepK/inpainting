import random
random.seed(42)
import os
import sys

from keras.optimizers import Adam
from keras import callbacks

from unet import DICTAVAILNETWORKS3D
from inpaint_reader import get_inpainter_generators
from config import *

batchsize = 5
epochs = 100
lr = 0.0001
steps_per_epoch = int(1000./batchsize)
validation_steps = int(150./batchsize)

unet_path = None

with_roi = int(sys.argv[1])
with_superpixel = int(sys.argv[2])
if with_roi == 1 and with_superpixel == 1:
    with_roi = True
    with_superpixel = True
    unet_path = "../models/inpainter_unet_roi_superpixel/"
elif with_roi == 0 and with_superpixel == 1:
    with_roi = False
    with_superpixel = True
    unet_path = "../models/inpainter_unet_noroi_superpixel/"
elif with_roi == 1 and with_superpixel == 0:
    with_roi = True
    with_superpixel = False
    unet_path = "../models/inpainter_unet_roi_grid/"
elif with_roi == 0 and with_superpixel == 0:
    with_roi = False
    with_superpixel = False
    unet_path = "../models/inpainter_unet_noroi_grid/"

for split_number in range(1,4):
    model = DICTAVAILNETWORKS3D(SIZE_OF_IMAGE, INPAINTER_UNET).getModel()
    a = Adam(lr=lr)
    model.compile(optimizer= a, loss = "mse")
    print (model.summary())

    train_gen, valid_gen = get_inpainter_generators(batch_size = batchsize, split_number = split_number,\
        with_roi = with_roi, with_superpixel = with_superpixel)

    os.makedirs(unet_path, exist_ok = True)
    save_best_model = callbacks.ModelCheckpoint(unet_path + "weights%s.h5"%split_number, monitor='val_loss',\
                                                  verbose=1, save_best_only=True, mode='min')

    history = model.fit_generator(train_gen,\
                            steps_per_epoch = steps_per_epoch,\
                            epochs = epochs,\
                            validation_data = valid_gen,\
                            validation_steps = validation_steps,\
                            shuffle = True,
                            callbacks = [save_best_model])

    import pickle
    pickle.dump(history.history, open(unet_path + "history%s.p"%split_number, "wb"))

import random
random.seed(42)
import os

from keras.optimizers import Adam
from keras import callbacks

from unet import DICTAVAILNETWORKS3D
from deform_reader import get_generators

split_number = 1

model = DICTAVAILNETWORKS3D((80, 216, 128), 'Unet3D_Shallow_Batchnorm').getModel()
a = Adam(lr=0.001)
model.compile(optimizer= a, loss = "mse")
print (model.summary())

brats_parent = "/home/kayald/Code/inpainting-pretraining/MICCAI_BraTS_2018_Data_Training/HGG/"
train_gen, valid_gen = get_generators(brats_parent, split_number = split_number)

save_path = "../models/deform-unet/"
os.makedirs(save_path, exist_ok = True)
save_best_model = callbacks.ModelCheckpoint(save_path + "weights%s.h5"%split_number, monitor='val_loss',\
                                              verbose=1, save_best_only=True, mode='min')

history = model.fit_generator(train_gen, steps_per_epoch = 1510, epochs = 50,\
					validation_data = valid_gen, validation_steps = 270, shuffle = True,\
					callbacks = [save_best_model])

import pickle
pickle.dump(history.history, open(save_path + "history%s.p"%split_number, "wb"))

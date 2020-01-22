import random
random.seed(42)
import os

from keras.optimizers import Adam
from keras import callbacks

from unet import DICTAVAILNETWORKS3D
from reader import get_generators
from utilities import traintestvalid_split5, get_flair_file_names
from losses import dice_coef_loss

split_number = 1

model = DICTAVAILNETWORKS3D((160, 224, 128), 'Unet3D_Batchnorm').getModel()
a = Adam(lr=5e-5)
model.compile(optimizer= a, loss = dice_coef_loss)
print (model.summary())

brats_parent = "/home/kayald/Code/inpainting-pretraining/MICCAI_BraTS_2018_Data_Training/HGG/"
flair_names, seg_names = get_flair_file_names(brats_parent)
train_flair_names, _, valid_flair_names, _, _, _ = traintestvalid_split5(flair_names, seg_names)[split_number]
steps_per_epoch = len(train_flair_names)
validation_steps = len(valid_flair_names)

train_gen, test_gen, valid_gen = get_generators(brats_parent, split_number = split_number)

save_path = "../models/vanilla_unet/"
os.makedirs(save_path, exist_ok = True)
save_best_model = callbacks.ModelCheckpoint(save_path + "weights%s.h5"%split_number, monitor='val_loss',\
                                              verbose=1, save_best_only=True, mode='min')

history = model.fit_generator(train_gen,\
					steps_per_epoch = steps_per_epoch,\
					epochs = 200,\
					validation_data = valid_gen,\
					validation_steps = validation_steps,\
					shuffle = True,
					callbacks = [save_best_model])

import pickle
pickle.dump(history.history, open(save_path + "history%s.p"%split_number, "wb"))

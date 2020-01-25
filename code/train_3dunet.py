import random
random.seed(42)
import os
import sys

from keras.optimizers import Adam
from keras import callbacks

from unet import DICTAVAILNETWORKS3D
from reader import get_generators
from utilities import traintestvalid_split5, get_flair_file_names
from losses import dice_coef_loss
from config import *

unet_path = sys.argv[1]
epochs = int(sys.argv[2])
lr = float(sys.argv[3])

batchsize = 5

for split_number in range(1,6):

	model = DICTAVAILNETWORKS3D(SIZE_OF_IMAGE, UNET_TYPE).getModel()
	a = Adam(lr=lr)
	model.compile(optimizer= a, loss = dice_coef_loss)

	flair_names, seg_names = get_flair_file_names()
	train_flair_names, _, valid_flair_names, _, _, _ = traintestvalid_split5(flair_names, seg_names)[split_number]
	steps_per_epoch = int(len(train_flair_names)/batchsize)
	validation_steps = int(len(valid_flair_names)/batchsize)

	train_gen, test_gen, valid_gen = get_generators(batch_size = batchsize, split_number = split_number)

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

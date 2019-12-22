import random
random.seed(42)

from keras.optimizers import Adam

from unet import DICTAVAILNETWORKS3D
from reader import get_generators
from utilities import traintestvalid_split5, get_flair_file_names

split_number = 1

model = DICTAVAILNETWORKS3D((80, 160, 120), 'Unet3D_Shallow_Batchnorm').getModel()
a = Adam(lr=1e-5)
model.compile(optimizer= a, loss = "binary_crossentropy")
print (model.summary())

brats_parent = "/home/kayald/Code/inpainting-pretraining/MICCAI_BraTS_2018_Data_Training/HGG/"
flair_names, seg_names = get_flair_file_names(brats_parent)
train_flair_names, _, valid_flair_names, _, _, _ = traintestvalid_split5(flair_names, seg_names)[split_number]
steps_per_epoch = len(train_flair_names)
validation_steps = len(valid_flair_names)

train_gen, test_gen, valid_gen = get_generators(brats_parent, split_number = split_number)

model.fit_generator(train_gen,\
					steps_per_epoch = steps_per_epoch,\
					epochs = 50,\
					validation_data = valid_gen,\
					validation_steps = validation_steps,\
					shuffle = True)

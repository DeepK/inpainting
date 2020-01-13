from unet import DICTAVAILNETWORKS3D
from reader import get_generators
from losses import dice_coef_np

import numpy as np
from matplotlib import pyplot as plt

model = DICTAVAILNETWORKS3D((160, 216, 128), 'Unet3D_Shallow_Batchnorm').getModel()
model.load_weights('../models/inpainter_unet_finetuned/weights1.h5')

brats_parent = "/home/kayald/Code/inpainting-pretraining/MICCAI_BraTS_2018_Data_Training/HGG/"
train_gen, test_gen, valid_gen = get_generators(brats_parent, split_number = 1)

dices = []
for img, seg in test_gen:
	pred = model.predict(img)
	pred[pred > 0.5] = 1

	d = dice_coef_np(seg, pred)
	dices.append(d)

print (np.mean(dices))

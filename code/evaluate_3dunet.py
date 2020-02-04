import numpy as np
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from unet import DICTAVAILNETWORKS3D
from config import *
from reader import get_generators
from losses import dice_coef_np

unet_path = sys.argv[1]

with open("results.txt", "a") as resultsfile:
    for split_number in range(1,2):
        model = DICTAVAILNETWORKS3D(SIZE_OF_IMAGE, UNET_TYPE).getModel()
        model.load_weights(os.path.join(unet_path, "weights{}.h5".format(split_number)))
        _, test_gen, _ = get_generators(batch_size = 1, split_number = split_number)

        dices = []
        for img, seg in test_gen:
            pred = model.predict(img)
            pred[pred > 0.5] = 1

            d = dice_coef_np(seg, pred)
            dices.append(d)

    resultsfile.write(unet_path + " : " + str(np.mean(dices)) + "\n")
from unet import DICTAVAILNETWORKS3D
from config import *
from keras.layers import Conv3D
from keras.models import Model

def get_model(pretrain_model_path):
    if pretrain_model_path is None:
        return DICTAVAILNETWORKS3D(SIZE_OF_IMAGE, UNET_TYPE).getModel()
    else:
        print ("Loading from " + pretrain_model_path)
        model = DICTAVAILNETWORKS3D(SIZE_OF_IMAGE, INPAINTER_UNET).getModel()
        model.load_weights(pretrain_model_path)

        penultimate_layer = model.layers[-2]
        # new layer, with one channel
        new_conv = Conv3D(filters=1,
                        kernel_size=(1, 1, 1),
                        padding="same",
                        activation="sigmoid")
        # Reconnect the layers
        x = new_conv(penultimate_layer.output)
        # Create a new model
        new_model = Model(input = model.input, output = x)
        
        return new_model

if __name__ == "__main__":
    model = get_model("../models/inpainter_unet_True/weights1.h5")
    print (model.summary())
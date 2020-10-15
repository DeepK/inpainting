BRATS_PARENT = "../data/MICCAI_BraTS_2018_Data_Training/HGG/"
UNET_TYPE = "Unet3D_Shallow_Batchnorm"
INPAINTER_UNET = "Unet3D_Shallow_Batchnorm_2Channel_recon"
SIZE_OF_IMAGE = (160, 216, 32)
MIN_GRID_SIZE = 20
MAX_GRID_SIZE = 45
MIN_AREA_SUPERPIXEL = 1500
MIN_AREA_GRID = 400
REPLICATION_FACTOR_FOR_NO_ROI_SUPERPIXEL_INPAINTING = 6
REPLICATION_FACTOR_FOR_NO_ROI_GRID_INPAINTING = 9
SUPERPIXEL_FOLDER = "../superpixels/"

"""
Reads the original 3D scans!
subhradeep.kayal@gmail.com
Erasmus MC
"""
import nibabel as nib
import numpy
import cv2

def get_original_scan(filepath):
	data = nib.load(filepath).get_fdata()
	data = cv2.normalize(data, None, 255.0, 0.0, cv2.NORM_MINMAX)
	data = data/255.0
	return data

if __name__ == "__main__":
	fpath = "/home/kayald/Code/inpainting-pretraining/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_TCIA02_222_1/Brats18_TCIA02_222_1_flair.nii.gz"
	data = get_original_scan(fpath)
	print (data.shape)
"""
SLIC
subhradeep.kayal@gmail.com
Erasmus MC
"""
from skimage import segmentation

def get_superpixel_labels(img):
	label = segmentation.slic(img, n_segments = 2000, compactness = 0.25, multichannel = False)
	return label

if __name__ == "__main__":
	from reader import get_scan
	fpath = "/home/kayald/Code/inpainting-pretraining/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_TCIA02_222_1/Brats18_TCIA02_222_1_flair.nii.gz"
	segfpath = "/home/kayald/Code/inpainting-pretraining/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_TCIA02_222_1/Brats18_TCIA02_222_1_seg.nii.gz"
	img = get_scan(fpath)
	seg = get_scan(segfpath)

	from matplotlib import pyplot as plt
	import numpy

	img = img[40:200, 40:200, :]
	seg = seg[40:200, 40:200, :]
	label = get_superpixel_labels(img)

	for i in range(10, 140):
		overlayed_img = numpy.maximum(img[:, :, i], seg[:, :, i])
		boundaries = segmentation.mark_boundaries(overlayed_img, label[:, :, i])
		fig = plt.figure("Boundaries")
		ax = fig.add_subplot(1, 1, 1)
		ax.imshow(boundaries, interpolation = "nearest")
		plt.axis("off")
		plt.show()
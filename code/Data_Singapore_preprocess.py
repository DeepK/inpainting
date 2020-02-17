"""
Pytorch framework for Medical Image Analysis

Create data

Author(s): Shuai Chen
PhD student in Erasmus MC, Rotterdam, the Netherlands
Biomedical Imaging Group Rotterdam

If you have any questions or suggestions about the code, feel free to contact me:
Email: chenscool@gmail.com

Date: 22 Jan 2019
"""

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import numpy.ma as ma
from matplotlib.colors import ListedColormap
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from tqdm import tqdm
from glob import glob
import SimpleITK as sitk
import scipy.ndimage

import common_module as mkd
import warnings
warnings.filterwarnings('ignore')
matplotlib.use("TkAgg")

plt.ion()

# Data_root_dir = os.path.join(os.path.abspath(__file__), '60_WMH')
Hospital_dir = sorted(glob('../60_WMH/*'))
Singapore_list = sorted(glob(Hospital_dir[1] + "/*"))

print('-' * 30)
print('Loading files...')
print('-' * 30)

num = 20
num_modality = 2

for nb_file in range(len(Singapore_list)):

    # Set image path
    T1File = (Singapore_list[nb_file] + "/pre/T1.nii.gz")
    FLAIRFile = (Singapore_list[nb_file] + "/pre/FLAIR.nii.gz")
    maskFile = (Singapore_list[nb_file] + "/wmh.nii.gz")


    # Read T1 image
    T1Image = sitk.ReadImage(T1File)
    T1Vol = sitk.GetArrayFromImage(T1Image).astype(float)
    T1mask = np.where(T1Vol >= 30, 1, 0)

    T1Vol = T1Vol * T1mask

    # Set padding and cut parameters
    cut_slice = 0

    size_h = 256
    size_w = 256

    if (size_h - T1Vol.shape[1]) % 2 == 0:
        pad_h1 = (size_h - T1Vol.shape[1])/2
        pad_h2 = (size_h - T1Vol.shape[1])/2
    else:
        pad_h1 = (size_h - T1Vol.shape[1])/2
        pad_h2 = (size_h - T1Vol.shape[1])/2 + 1

    if (size_w - T1Vol.shape[2]) % 2 == 0:
        pad_w1 = (size_w - T1Vol.shape[2])/2
        pad_w2 = (size_w - T1Vol.shape[2])/2
    else:
        pad_w1 = (size_w - T1Vol.shape[2])/2
        pad_w2 = (size_w - T1Vol.shape[2])/2 + 1

    # Padding and cut image
    T1Vol = np.pad(T1Vol, ((0, 0), (int(pad_h1), int(pad_h2)), (int(pad_w1), int(pad_w2))), mode='constant', constant_values=0)
    T1Vol = T1Vol[cut_slice:, :, :]

    # Gaussion Normalization
    # T1Vol -= np.mean(T1Vol)
    # T1Vol /= np.std(T1Vol)

    T1Vol *= (1.0 / T1Vol.max())

    # Visualize image
    # plt.figure(figsize=(10, 6))
    # plt.suptitle("T1 image of patient {}".format(Singapore_list[nb_file].split('/')[-1]), fontsize=18)
    # start_slice = 20
    # for i in range(0, 6):
    #     T1slice = T1Vol[i + start_slice]
    #     ax = plt.subplot(2, 3, i + 1)
    #     plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    #     ax.set_title('Slice {}'.format(i + start_slice), fontsize=14)
    #     ax.axis('off')
    #     plt.imshow(T1slice, cmap='gray')
    #     plt.pause(0.001)
    # plt.show()
    # plt.close()

    #  Read FLAIR image
    FLAIRImage = sitk.ReadImage(FLAIRFile)
    FLAIRVol = sitk.GetArrayFromImage(FLAIRImage).astype(float)
    FLAIRmask = np.where(FLAIRVol >= 30, 1, 0)

    FLAIRVol = FLAIRVol * FLAIRmask

    # Padding and cut image
    FLAIRVol = np.pad(FLAIRVol, ((0, 0), (int(pad_h1), int(pad_h2)), (int(pad_w1), int(pad_w2))), mode='constant', constant_values=0)
    FLAIRVol = FLAIRVol[cut_slice:, :, :]

    # Gaussion Normalization
    FLAIRVol -= np.mean(FLAIRVol)
    FLAIRVol /= np.std(FLAIRVol)

    FLAIRVol *= (1.0 / FLAIRVol.max())


    # Visualize image
    # plt.figure(figsize=(10, 6))
    # plt.suptitle("FLAIR image of patient {}".format(Singapore_list[nb_file].split('/')[-1]), fontsize=18)
    # start_slice = 20
    # for i in range(0, 6):
    #     FLAIRslice = FLAIRVol[i + start_slice]
    #     ax = plt.subplot(2, 3, i + 1)
    #     plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    #     ax.set_title('Slice {}'.format(i + start_slice), fontsize=14)
    #     ax.axis('off')
    #     plt.imshow(FLAIRslice, cmap='gray')
    #     plt.pause(0.001)
    # plt.show()
    # plt.close()

    # Read mask file
    maskImage = sitk.ReadImage(maskFile)
    maskVol = sitk.GetArrayFromImage(maskImage).astype(float)

    # Padding and cut image
    maskVol = np.pad(maskVol, ((0, 0), (int(pad_h1), int(pad_h2)), (int(pad_w1), int(pad_w2))), mode='constant', constant_values=0)
    maskVol = maskVol[cut_slice:, :, :]

    # Only keep WMH label
    maskVol = np.where(maskVol != 1, 0, maskVol)

    # Visualize image
    # transparent1 = 1.0
    # transparent2 = 1.0
    # cmap = pl.cm.viridis
    # my_cmap = cmap(np.arange(cmap.N))
    # my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
    # my_cmap = ListedColormap(my_cmap)
    #
    # plt.figure(figsize=(10, 6))
    # plt.suptitle("Mask on FLAIR image of patient {}".format(Singapore_list[nb_file].split('/')[-1]), fontsize=18)
    # start_slice = 20
    # for i in range(0, 6):
    #     FLAIRslice = FLAIRVol[i + start_slice]
    #     maskslice = maskVol[i + start_slice]
    #     ax = plt.subplot(2, 3, i + 1)
    #     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #     ax.set_title('Slice {}'.format(i + start_slice))
    #     ax.axis('off')
    #     plt.imshow(FLAIRslice, cmap='gray', alpha=transparent1)
    #     plt.imshow(maskslice, cmap=my_cmap, alpha=transparent2)
    #     plt.pause(0.001)
    # plt.show()
    # plt.close()

    imageVol = np.concatenate((np.expand_dims(T1Vol, axis=-1), np.expand_dims(FLAIRVol, axis=-1)), axis=-1)

    mkd.mkdir('../data')
    mkd.mkdir('../data/Singapore')
    np.save('../data/Singapore/img_%04d.npy' % (int(Singapore_list[nb_file].split('/')[-1])), imageVol)
    np.save('../data/Singapore/mask_%04d.npy' % (int(Singapore_list[nb_file].split('/')[-1])), maskVol)

    print('Singapore Image process {}/{} finished'.format(nb_file, len(Singapore_list)))


print('finished')

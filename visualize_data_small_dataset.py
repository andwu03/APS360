#from https://www.kaggle.com/code/marvyshaker123/tumor-segmentation
#essentially exact code from kaggle

import nibabel as nib
from matplotlib import pyplot as plt
import os
TRAIN_DATASET_PATH = 'C:/Users/grace/OneDrive/Surface Laptop Desktop/UofT/APS360/Project/BraTS2020_TrainingData_Small/MICCAI_BraTS2020_TrainingData/Small_Dataset/'

test_image_flair = nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_361/BraTS20_Training_361_flair.nii').get_fdata()
test_image_t1 = nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_361/BraTS20_Training_361_t1.nii').get_fdata()
test_image_t1ce = nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_361/BraTS20_Training_361_t1ce.nii').get_fdata()
test_image_t2 = nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_361/BraTS20_Training_361_t2.nii').get_fdata()
test_mask = nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_361/BraTS20_Training_361_seg.nii').get_fdata()

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize = (20, 10))
slice_w = 27
slice_number = test_image_flair.shape[0]//2-slice_w
ax1.imshow(test_image_flair[:,:,slice_number], cmap='gray')
ax1.set_title('Image flair')
ax2.imshow(test_image_t1[:,:,slice_number], cmap='gray')
ax2.set_title('Image t1')
ax3.imshow(test_image_t1ce[:,:,slice_number], cmap='gray')
ax3.set_title('Image t1ce')
ax4.imshow(test_image_t2[:,:,slice_number], cmap='gray')
ax4.set_title('Image t2')
ax5.imshow(test_mask[:,:,slice_number])
ax5.set_title('Mask')
# plt.show()

print(test_image_flair.shape, test_image_t1.shape, test_image_t1ce.shape, test_image_t2.shape, test_mask.shape)

#plot the slices
from skimage.util import montage
from skimage.transform import rotate
import numpy as np

#visualize t1
# We can skip 50:-50 slices since there is not much to see
fig, ax1 = plt.subplots(1, 1, figsize = (15,15))
ax1.imshow(montage(np.transpose(test_image_flair[..., 50:-50], (2, 0, 1))), cmap ='gray')
# plt.show()

#visualize image
fig, ax1 = plt.subplots(1, 1, figsize = (15,15))
ax1.imshow(montage(np.transpose(test_mask[..., 50:-50], (2, 0, 1))), cmap ='gray')
# plt.show()
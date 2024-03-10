import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchsummary import summary
import numpy as np
import nibabel as nib
import os

TRAIN_DATASET_PATH = 'C:/Users/grace/OneDrive/Surface Laptop Desktop/UofT/APS360/Project/BraTS2020_TrainingData_Small/MICCAI_BraTS2020_TrainingData/Small_Dataset/'


#split into train, test and validation
from sklearn.model_selection import train_test_split
import os

#need to select only flair?
# lists of directories with studies
train_and_val_directories = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]

# file BraTS20_Training_355 has ill formatted name for for seg.nii file
train_and_val_directories.remove(TRAIN_DATASET_PATH+'BraTS20_Training_355')


def pathListIntoIds(dirList):
    x = []
    for i in range(0,len(dirList)):
        x.append(dirList[i][dirList[i].rfind('/')+1:])
    return x

train_and_test_ids = pathListIntoIds(train_and_val_directories);

train_test_ids, val_ids = train_test_split(train_and_test_ids,test_size=0.15)
train_ids, test_ids = train_test_split(train_test_ids,test_size=0.15)

print(f"Train: {len(train_ids)} | Validation: {len(val_ids)} | Test: {len(test_ids)}")


#Load dataset
def load_dataset(ids, path):
    images = np.zeros((len(ids)*10, 240, 240), np.float32)
    masks = np.zeros((len(ids)*10, 240, 240), np.float32)

    i = 0
    for id in ids:
        print(i)
        t2 = nib.load(f"{path}{id}/{id}_t2.nii").get_fdata()
        seg = nib.load(f"{path}{id}/{id}_seg.nii").get_fdata()

        for s in range(50, seg.shape[2]-50, 10):
            images[i] = t2[:, :, s] / t2.max()
            masks[i] = seg[:, :, s] > 0
            i += 1

    images = np.expand_dims(images[:i], axis=1)
    masks = np.expand_dims(masks[:i], axis=1)

    return images, masks

train_images, train_masks = load_dataset(train_ids, TRAIN_DATASET_PATH)
val_images, val_masks = load_dataset(val_ids, TRAIN_DATASET_PATH)


print(train_images.shape, val_masks.shape)

plt.subplot(121)
plt.imshow(train_images[100, 0], cmap='gray')
plt.subplot(122)
plt.imshow(train_masks[100, 0], cmap='gray')


#make dataloaders from the images
from torch.utils.data import TensorDataset, DataLoader

batch_size = 32

train_dataset = TensorDataset(torch.from_numpy(train_images).type(torch.float32), torch.from_numpy(train_masks).type(torch.float32))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(torch.from_numpy(val_images).type(torch.float32), torch.from_numpy(val_masks).type(torch.float32))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#plot the dataloader
X, Y = next(iter(train_dataloader))
plt.subplot(121)
plt.imshow(X[0, 0], cmap='gray')
plt.subplot(122)
plt.imshow(Y[0, 0], cmap='gray')

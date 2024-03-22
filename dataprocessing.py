# A file to handle data processing and loading.

# The dataset is quite big so we want to use an address-based system instead of 
# loading the entire dataset into memory. This is done using the ImageFolder
# class from torchvision.datasets. This class allows us to load images from a
# directory and apply transformations to them. We can then use the DataLoader
# class to load the images into memory in batches.

# The dataset is structured as follows:
# BraTS2020_TrainingData/
#     BraTS20_Training_001/
#         BraTS20_Training_001_flair.nii
#         BraTS20_Training_001_t1.nii
#         BraTS20_Training_001_t1ce.nii
#         BraTS20_Training_001_t2.nii
#         BraTS20_Training_001_seg.nii
#     BraTS20_Training_002/
#         ...



import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

# Define the dataset class


# Flip operation (to double the size of the dataset)
# VERY IMPORTANT
# TODO TODO TODO
# Flip is very fundamentally broken. I need to fix this. 
# It just refuses to flip the label. 
# I must rewrite this logic so that we can handle both 
# flipped and non-flipped images within the same dataset.
class Flip:
    def __call__(self, sample):
        # double check the axis. We can only flip in the coronal plane. 
        # (Flipping left and right brains, while maintaining the same orientation)
        print("Sample shape:", sample.shape)
        if len(sample.shape) == 3:
            # flip label
            print("Flipping label via FLIP, Axis: 0")
            return torch.flip(sample, [0])
        else:
            # flip image
            print("Flipping image via FLIP, Axis: 1")
            return torch.flip(sample, [1])
            
    

# Stack operation (Combines _flair, _t1, _t1ce, _t2 images to a single tensor)
class Stack:
    def __call__(self, image_path):
        # Load the image data
        image = nib.load(image_path).get_fdata()
        return image
    
# Normalize operation (Normalizes the image to have a mean of 0 and a standard deviation of 1)
class Normalize:
    def __call__(self, sample):
        # Don't normalize if it's a label we're passing in here.
        if len(sample.shape) == 3:
            return sample

        sample = (sample - torch.mean(sample)) / torch.std(sample)
        return sample
    
# ToTensor operation (Converts the numpy array to a PyTorch tensor)
class ToTensor:
    def __call__(self, sample):
        if isinstance(sample, np.ndarray):
            sample = torch.from_numpy(sample)
        return sample

# Define the dataset class
class BraTSDataset(Dataset):
    def __init__(self, data_dir, transform=None, flip = True, norm = True, stack = True):
        # Flip doubles the size of the dataset by flipping the images in the coronal plane.
        # For now we just double the dataset nominally with the exact file names, 
        # and make an index that tracks if an image is flipped or not.
        self.data_dir = data_dir
        self.transform = transform
        self.flip = flip
        self.norm = norm
        self.stack = stack
        self.images = []
        self.labels = []
        self.flip_images = []

        # Get the subdirectories in the data_dir
        subdirs = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

        # Iterate over the subdirectories
        for subdir in subdirs:
            subdir_path = os.path.join(data_dir, subdir)
            images = [f for f in os.listdir(subdir_path) if f.endswith('.nii') and not f.endswith('seg.nii')]
            labels = [f for f in os.listdir(subdir_path) if f.endswith('seg.nii')]
            images.sort()
            labels.sort()
            # print(images)
            # print(labels)
            self.images.extend(images)
            self.labels.extend(labels)
            print("Added to set: ", images, labels)

            # break # ONLY ONE FOR NOW TODO

        # also make an array for flipped images

        # Sort the images and labels
        self.images.sort()
        self.labels.sort()

        # Group the images
        self.images = [self.images[i:i+4] for i in range(0, len(self.images), 4)]
        self.labels = [self.labels[i] for i in range(0, len(self.labels))]

        if flip:
            self.flip_images = np.concatenate((self.flip_images, np.zeros(len(self.images))))
            self.flip_images = np.concatenate((self.flip_images, np.ones(len(self.images))))
            self.images = self.images + self.images
            print("Images: ", self.images)
            self.labels = self.labels + self.labels
            # When retrieving the images, we will need to check if the image is flipped or not.
            # potentially extremely inefficient! but i don't care


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # data dir is the directory of the folders, each containing the images.
        # the name of the folder is the first part of the file name.
        # BraTS20_Training_001 
        updated_path = os.path.join(self.data_dir, self.images[idx][0])
        updated_path = updated_path[:updated_path.rfind('_')]
        
        
        images = [torch.from_numpy(Stack()(os.path.join(updated_path, img))) for img in self.images[idx]]
        # Stack the images along a new dimension
        if self.stack:
            images = torch.stack(images, dim=0)
        # Load the label
        label = torch.from_numpy(Stack()(os.path.join(updated_path, self.labels[idx])))

        # Normalize the images to have a mean of 0 and a standard deviation of 1
        # if self.norm:
        #     images = (images - images.mean()) / images.std()
        
        # Update all '4' labels to '3' labels
        label[label == 4] = 3

        

        # Apply user-defined transformations
        if self.transform:
            images = self.transform(images)
            label = self.transform(label)
        
        # Flip the images back if there's a flag for it in self.flip_images
        # god. what beautiful code. i've outdone myself
        # if anyone reads this i'm really sorry
        # print("Flip_images: ", self.flip_images)
        if self.flip:
            if self.flip_images[idx] == 1:
                print("Flipping image via getitem: ", self.flip_images[idx])
                images = torch.flip(images, [1])
                label = torch.flip(label, [0])
            else: 
                print("Not flipping image: ", self.flip_images[idx])

        return images, label
    

def generate_dataloaders(data_dir='', batch_size=1, flip = False, norm = False, stack = True):
    if data_dir == '':
        data_dir = r"C:\Users\sparq\Videos\EngsciMisc\APS360\Project\BraTS\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
    dataproctransform = transforms.Compose([
        Flip(),
        Normalize(),
        ToTensor()
    ])

    # Create the dataset
    dataset = BraTSDataset(data_dir=data_dir, transform=dataproctransform, flip=flip, norm=norm, stack=stack)

    # Set up the dataloaders
    # Train, Val, Test in a 80/20 split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True)

    print("Train size: ", len(train_dataloader))
    print("Val size: ", len(val_dataloader))


    return train_dataloader, val_dataloader

def TESTFUNCTIONDONOTRUN():
    # Primarily testing to make sure everything works. 
    import matplotlib.pyplot as plt

    # Define the data directory
    data_dir = r"C:\Users\sparq\Videos\EngsciMisc\APS360\Project\BraTS\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"


    # Check path
    # print(os.path.exists(data_dir))
    # print(os.listdir(data_dir))


    # Define the transformations
    transform = transforms.Compose([
        Flip(),
        Normalize(),
        ToTensor()
    ])

    # Create the dataset
    dataset = BraTSDataset(data_dir=data_dir, transform=transform, flip = False, norm = False, stack = True)

    # Create a data loader
    train_loader, val_loader = generate_dataloaders(data_dir, batch_size=1, flip = True, norm = False, stack = True)    


    # Visualize the dataset
    for i, (images, label) in enumerate(train_loader):
        # Access the first brain scan in the batch
        brain_scan = images[0]
        
        # Visualize the brain scan
        slice_index = brain_scan.shape[3] // 2
        slice_image = brain_scan[0, :, :, slice_index]

        # print("Image shape: ", brain_scan.shape)
        # print("Image index: ", slice_index)

        # print("Train loader length: ", len(train_loader))

        
        plt.subplot(1, 2, 1)
        plt.imshow(slice_image, cmap='gray')
        plt.title('Image')
        plt.subplot(1, 2, 2)
        plt.imshow(label[0, :, :, slice_index], cmap='gray')
        plt.title('Label')
        plt.show()
        

        

if __name__ == "__main__":
    TESTFUNCTIONDONOTRUN()



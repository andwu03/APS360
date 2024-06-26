import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchsummary import summary
import numpy as np
import nibabel as nib
import os
from load_images import test_dataloader, train_dataloader
from visualize_data import test_image_flair, test_mask, slice_number

import time

TEST_DATASET_PATH = "/home/andrew/APS360_Project/Data/MICCAI_BraTS2020_TestingData/"
checkpoint = '/home/andrew/APS360_Project/APS360/model_epoch_80.pth.tar'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def double_convolution(in_channels, out_channels):
    """
    Creates a double convolutional layer with batch normalization and Leaky ReLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Returns:
        nn.Sequential: Double convolutional layer with batch normalization and Leaky ReLU activation.
    """
    conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, affine=False, track_running_stats=False),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, affine=False, track_running_stats=False),
            nn.LeakyReLU(0.1,inplace=True)
        )
    return conv_op

class UNet(nn.Module):
    """
    UNet is a convolutional neural network architecture used for image segmentation tasks.
    It consists of a contracting path and an expanding path, which allows for capturing both
    local and global information in the input image.
    """

    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        # Contracting path.
        # Each convolution is applied twice.
        # Convolutional laters in the contracting path
        self.down_convolution_1 = double_convolution(1, 64)
        self.down_convolution_2 = double_convolution(64, 128)
        self.down_convolution_3 = double_convolution(128, 256)
        self.down_convolution_4 = double_convolution(256, 512)
        self.down_convolution_5 = double_convolution(512, 1024)

        # Expanding path.
        #transpose convolutional laters and expanding path
        self.up_transpose_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512,
            kernel_size=2,
            stride=2)
        # Below, `in_channels` again becomes 1024 as we are concatinating.
        self.up_convolution_1 = double_convolution(1024, 512)
        self.up_transpose_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256,
            kernel_size=2,
            stride=2)
        self.up_convolution_2 = double_convolution(512, 256)
        self.up_transpose_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128,
            kernel_size=2,
            stride=2)
        self.up_convolution_3 = double_convolution(256, 128)
        self.up_transpose_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64,
            kernel_size=2,
            stride=2)
        self.up_convolution_4 = double_convolution(128, 64)
        # output => `out_channels` as per the number of classes.
        self.out = nn.Conv2d(
            in_channels=64, out_channels=num_classes,
            kernel_size=1
        )

    def forward(self, x):
        # TODO: Write here!
        down_1 = self.down_convolution_1(x)
        down_2 = self.max_pool2d(down_1)
        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool2d(down_3)
        down_5 = self.down_convolution_3(down_4)
        down_6 = self.max_pool2d(down_5)
        down_7 = self.down_convolution_4(down_6)
        down_8 = self.max_pool2d(down_7)
        down_9 = self.down_convolution_5(down_8)

        up_1 = self.up_transpose_1(down_9)
        up_2 = self.up_convolution_1(torch.cat([down_7, up_1], 1))
        up_3 = self.up_transpose_2(up_2)
        up_4 = self.up_convolution_2(torch.cat([down_5, up_3], 1))
        up_5 = self.up_transpose_3(up_4)
        up_6 = self.up_convolution_3(torch.cat([down_3, up_5], 1))
        up_7 = self.up_transpose_4(up_6)
        up_8 = self.up_convolution_4(torch.cat([down_1, up_7], 1))

        out = self.out(up_8)

        return out
    
TRAIN_DATASET_PATH = "/home/andrew/APS360_Project/Data/MICCAI_BraTS2020_TestingData/"

#load the checkpoint
def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Update the learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

model = UNet(num_classes=1).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
load_checkpoint(checkpoint, model, optimizer, 0.001)

def predict_image(model, device, image):
    """
    Function to predict the segmentation mask of an input image using the trained model.

    Parameters:
    - model: The trained segmentation model.
    - device: The device (CPU or GPU) to perform computation on.
    - image: The input image tensor.

    Returns:
    - The predicted segmentation mask.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Move the image to the specified device
    image = image.to(device)

    # Add a batch dimension if necessary
    if image.dim() == 3:
        image = image.unsqueeze(0)

    with torch.no_grad(): # Do not compute gradient
        output = model(image)
        # Apply a sigmoid since the last layer is a logits layer
        probs = torch.sigmoid(output)
        # Threshold the probabilities to create a binary mask
        preds = (probs > 0.5).float()
    return preds


# Example usage:
# Assuming `test_image` is a tensor of shape [1, 240, 240] representing a single channel MRI slice
# And the model and device are already defined and the model is loaded with trained weights
test_image_tensor = torch.from_numpy(test_image_flair[:,:,slice_number]).unsqueeze(0).unsqueeze(0).float()
prediction = predict_image(model, device, test_image_tensor)

# Visualizing the prediction
plt.imshow(prediction.squeeze().cpu(), cmap='gray')
plt.title (f'Predicted Segmentation Mask: Slice Number: {slice_number}')
plt.show()
plt.imshow(test_mask[:,:,slice_number])
plt.title("Actual Mask")
plt.show()
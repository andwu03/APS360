

#https://www.kaggle.com/code/ankruteearora/tumor-segmentation

#code same as kaggle

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

# TRAIN_DATASET_PATH = 'C:/Users/grace/OneDrive/Surface Laptop Desktop/UofT/APS360/Project/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
TRAIN_DATASET_PATH = "/home/andrew/APS360_Project/Data/MICCAI_BraTS2020_TrainingData/"

def double_convolution(in_channels, out_channels):

    conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
    return conv_op

class UNet(nn.Module):
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

model = UNet(num_classes=1)

#connect to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

model.to(device);

summary(model, (1, 240, 240)) #give summary of the layers, output shape, number of parameters

#losses

import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice
    



#save checkpoint
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

# Setup loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()# DiceLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-2)

# Import tqdm for progress bar
from tqdm.auto import tqdm

# Define model
model = UNet(num_classes=1).to(device)

# Setup loss function and optimizer
loss_fn_1 = DiceLoss()
loss_fn_2 = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

# Set the seed and start the timer
torch.manual_seed(42)

# Set the number of epochs
epochs = 10

# Create training and testing loop
#record start time
start_time = time.time()

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch+1} of {epochs}")
    ### Training
    ### Training
    train_loss_1, train_loss_2, train_loss = 0, 0, 0
    #
    model.train()
    # Add a loop to loop through training batches
    print(f"Number of Batches: {len(train_dataloader)}")
    num=0
    for batch, (X, y) in enumerate(train_dataloader):
        #
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2 Calculate loss (per batch)
        loss_1 = loss_fn_1(y_pred, y)
        loss_2 = loss_fn_2(y_pred, y)
        loss = loss_1 + loss_2
        train_loss += loss # accumulatively add up the loss per epoch)
        train_loss_1 += loss_1
        train_loss_2 += loss_2

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        num += 1

        batch_time = time.time()
        total_batch_time = batch_time-start_time
        print(f"Elapsed Time (batch: {num}): {total_batch_time}")

    ##MODEL HAS NOT BEEN RUN PAST THIS POINT (8pm March 9th)

    # Divide total train loss by length of train dataloader (average loss per batch per epoch)
    train_loss /= len(train_dataloader)
    train_loss_1 /= len(train_dataloader)
    train_loss_2 /= len(train_dataloader)

    ### Testing
    # Setup variables for accumulatively adding up loss and accuracy
    test_loss_1, test_loss_2, test_loss = 0, 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            #
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate loss (accumatively)
            loss_1 = loss_fn_1(y_pred, y)
            loss_2 = loss_fn_2(y_pred, y)
            loss = loss_1 + loss_2
            test_loss += loss
            test_loss_1 += loss_1
            test_loss_2 += loss_2

        # Calculations on test metrics need to happen inside torch.inference_mode()
        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(test_dataloader)
        test_loss_1 /= len(test_dataloader)
        test_loss_2 /= len(test_dataloader)
    epoch_time = time.time()
    total_epoch_time = epoch_time-start_time
    print(f"Elapsed Time: {total_epoch_time}")

   ## Print out what's happening
    print(f"Train loss: {train_loss:.5f}, Dice: {train_loss_1:.5f}, BCE: {train_loss_2:.5f} | Test loss: {test_loss:.5f}, Dice: {test_loss_1:.5f}, BCE: {test_loss_2:.5f}\n")

    # Save checkpoint after every epoch
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }

    path = f"model_epoch_{epoch+1}.pth.tar"
    save_checkpoint(checkpoint, filename=path)


    # if epoch % 10 == 0:
    #     plt.subplot(231)
    #     plt.imshow(X[0, 0].cpu().detach().numpy(),cmap='gray')
    #     plt.axis('off')
    #     plt.subplot(232)
    #     plt.imshow(y[0, 0].cpu().detach().numpy(),cmap='gray')
    #     plt.axis('off')
    #     plt.subplot(233)
    #     plt.imshow(y_pred[0, 0].cpu().detach().numpy(),cmap='gray')
    #     plt.axis('off')
    #     plt.subplot(234)
    #     plt.imshow(X[12, 0].cpu().detach().numpy(),cmap='gray')
    #     plt.axis('off')
    #     plt.subplot(235)
    #     plt.imshow(y[12, 0].cpu().detach().numpy(),cmap='gray')
    #     plt.axis('off')
    #     plt.subplot(236)
    #     plt.imshow(y_pred[12, 0].cpu().detach().numpy(),cmap='gray')
    #     plt.axis('off')
    #     plt.show()


#load the checkpoint
def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Update the learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Example usage:
# model = UNet(num_classes=1).to(device)
# optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
# load_checkpoint('/kaggle/working/checkpoint_epoch_1.pth.tar', model, optimizer, 0.001)


y1 = y[0, 0].cpu().detach().numpy()
y2 = y_pred[0, 0].cpu().detach().numpy()
y = np.zeros((*y1.shape, 3))
y[..., 0] = y1
y[..., 1] = y2
y[..., 2] = y2

plt.subplot(131)
plt.imshow(X[0, 0].cpu().detach().numpy())
plt.subplot(132)
plt.imshow(y)

#save_model_path = 'C:/Users/grace/OneDrive/Surface Laptop Desktop/UofT/APS360/Project/model.pth'
save_model_path = "/home/andrew/APS360_Project/Trained_Models/model_0.pth"
torch.save(model.state_dict(), save_model_path)


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
# This is a simplified version of the U-Net architecture with fewer layers.
# It is designed for testing on a smaller dataset.
 
# We are using it as a baseline to compare with the more complex U-Net architecture
# we are implementing. 

# The code is adapted from: 
# Link: https://www.kaggle.com/code/ankruteearora/tumor-segmentation
# We have simplified it for ease of training and usage as a baseline. 


import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchsummary import summary
import numpy as np
import nibabel as nib
import os
# from load_images_small_dataset import test_dataloader, train_dataloader
# from visualize_data_small_dataset import test_image_flair, test_mask, slice_number

from load_images import test_dataloader, train_dataloader
from visualize_data import test_image_flair, test_mask, slice_number


import time

# Training Path. Change this to the local path. 
TRAIN_DATASET_PATH = 'C:/Users/grace/OneDrive/Surface Laptop Desktop/UofT/APS360/Project/BraTS2020_TrainingData_Small/MICCAI_BraTS2020_TrainingData/Small_Dataset/'

class UNet_baseline(nn.Module):
    def __init__(self, num_classes):
        super(UNet_baseline, self).__init__()
        # An autoencoder, of sorts. 
        # We add convolutional layers, in accordance with the U-Net architecture.

        # Encode
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # Batch normalization added
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decode
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, out_channels=num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoding
        x1 = torch.relu(self.bn1(self.conv1(x)))  # Apply batch normalization before ReLU
        x2 = self.maxpool(x1)

        # Decoding
        x3 = self.upconv1(x2)
        x4 = self.conv2(x3)

        return (x4)  # Sigmoid activation to ensure outputs are in [0, 1] range

model = UNet_baseline(num_classes=1)

# Connect to GPU.
# Only Andrew can actually make use of CUDA; though training through Colab is possible.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

model.to(device);

summary(model, (1, 240, 240)) 
# Summarize the model



# Losses of the model, plus plotting and saving checkpoints.

import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Comment out if model contains a sigmoid or equivalent activation layer.
        inputs = F.sigmoid(inputs)

        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice
    

# Save the model locally. 
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

# Setup loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss() # DiceLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-2)

# Import tqdm for progress bar
from tqdm.auto import tqdm

# Define model and loss
model = UNet_baseline(num_classes=1).to(device)

loss_fn_1 = DiceLoss()
loss_fn_2 = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

# Set the seed and start the timer
torch.manual_seed(42)

# Set the number of epochs
epochs = 50


# Training & Testing

# Start the timer. 
start_time = time.time()

DICE_train_loss = np.zeros(epochs)
BCE_train_loss = np.zeros(epochs)
combined_train_loss = np.zeros(epochs)

DICE_val_loss = np.zeros(epochs)
BCE_val_loss = np.zeros(epochs)
combined_val_loss = np.zeros(epochs)

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch+1} of {epochs}")
    # Training. 
    model.train()
    
    print(f"Number of Batches: {len(train_dataloader)}")
    num=0
    for batch, (X, y) in enumerate(train_dataloader):
        #
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss (per batch)
        loss_1 = loss_fn_1(y_pred, y)
        loss_2 = loss_fn_2(y_pred, y)
        loss = loss_1 + loss_2
        combined_train_loss[epoch] += loss 
        DICE_train_loss[epoch] += loss_1
        BCE_train_loss[epoch] += loss_2
        # Losses are the sum of all batch losses.

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


    # Average total train loss
    combined_train_loss /= len(train_dataloader)
    DICE_train_loss /= len(train_dataloader)
    BCE_train_loss /= len(train_dataloader)

    ### Validation
    # Setup variables for cumulatively adding up loss and accuracy

    model.eval()
    i = 0
    with torch.inference_mode():
        for X, y in test_dataloader:
            #
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(X)

            # 2. Loss (Same as above)
            loss_1 = loss_fn_1(y_pred, y)
            loss_2 = loss_fn_2(y_pred, y)
            loss = loss_1 + loss_2
            combined_val_loss[epoch] += loss
            DICE_val_loss[epoch] += loss_1
            BCE_val_loss[epoch] += loss_2

        # Calculations on test metrics need to happen inside torch.inference_mode()
        # Divide total test loss by length of test dataloader (per batch)
        combined_val_loss[epoch] /= len(test_dataloader)
        DICE_val_loss[epoch] /= len(test_dataloader)
        BCE_val_loss[epoch] /= len(test_dataloader)
    epoch_time = time.time()
    total_epoch_time = epoch_time-start_time
    print(f"Elapsed Time: {total_epoch_time}")

    # Specs for the epoch
    print(f"Train loss: {combined_train_loss[epoch]:.5f}, Dice: {DICE_train_loss[epoch]:.5f}, BCE: {BCE_train_loss[epoch]:.5f} | Test loss: {combined_val_loss[epoch]:.5f}, Dice: {DICE_val_loss[epoch]:.5f}, BCE: {DICE_val_loss[epoch]:.5f}\n")


    # Save checkpoint after every epoch
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }

    path = f"model_epoch_{epoch+1}.pth.tar"
    save_checkpoint(checkpoint, filename=path)


    if epoch % 10 == 0:
        plt.subplot(231)
        plt.imshow(X[0, 0].cpu().detach().numpy(),cmap='gray')
        plt.axis('off')
        plt.subplot(232)
        plt.imshow(y[0, 0].cpu().detach().numpy(),cmap='gray')
        plt.axis('off')
        plt.subplot(233)
        plt.imshow(y_pred[0, 0].cpu().detach().numpy(),cmap='gray')
        plt.axis('off')
        plt.subplot(234)
        plt.imshow(X[12, 0].cpu().detach().numpy(),cmap='gray')
        plt.axis('off')
        plt.subplot(235)
        plt.imshow(y[12, 0].cpu().detach().numpy(),cmap='gray')
        plt.axis('off')
        plt.subplot(236)
        plt.imshow(y_pred[12, 0].cpu().detach().numpy(),cmap='gray')
        plt.axis('off')
        plt.show()


# Plotting training and validation curves
plt.title("Training Curve")
plt.plot(range(1 ,epochs + 1), DICE_train_loss, label="Train")
plt.plot(range(1 ,epochs + 1), DICE_val_loss, label="Validation")
plt.legend(loc='best')
plt.xlabel("Epochs")
plt.ylabel("DICE")
plt.show()

plt.title("Training Curve")
plt.plot(range(1 ,epochs + 1), BCE_train_loss, label="Train")
plt.plot(range(1 ,epochs + 1), BCE_val_loss, label="Validation")
plt.legend(loc='best')
plt.xlabel("Epochs")
plt.ylabel("BCE")
plt.show()

plt.title("Training Curve")
plt.plot(range(1 ,epochs + 1), combined_train_loss, label="Train")
plt.plot(range(1 ,epochs + 1), combined_val_loss, label="Validation")
plt.legend(loc='best')
plt.xlabel("Epochs")
plt.ylabel("Combined DICE and BCE")
plt.show()
    

# Load the model from a checkpoint
def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Update the learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Example usage:
# model = UNet_baseline(num_classes=1).to(device)
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

save_model_path = 'C:/Users/grace/OneDrive/Surface Laptop Desktop/UofT/APS360/Project/model.pth'
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
plt.show()
# This is a simplified version of the U-Net architecture with fewer layers.
# It is designed for testing on a smaller dataset.
 
# We are using it as a baseline to compare with the more complex U-Net architecture
# we are implementing. 

# The code is adapted from: 
# Link: https://www.kaggle.com/code/ankruteearora/tumor-segmentation
# We have simplified it for ease of training and usage as a baseline. 


import numpy as np
import nibabel as nib
import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchsummary import summary
from load_images_small_dataset import test_dataloader, train_dataloader
from visualize_data_small_dataset import test_image_flair, test_mask, slice_number
import time


# TRAIN_DATASET_PATH = 'C:/Users/grace/OneDrive/Surface Laptop Desktop/UofT/APS360/Project/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
TRAIN_DATASET_PATH = "/home/andrew/APS360_Project/Data/MICCAI_BraTS2020_TrainingData/"


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
# Only Andrew can use this for now. 
device = 'cuda' if torch.cuda.is_available() else 'cpu' #if there is no gpu then use cpu
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
        inputs = F.sigmoid(inputs)

        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth) #calculate dice

        return 1 - dice #higher output is worse, lower output is better. 

# Save the model locally.
def checkpoint_saving(state, filename="my_checkpoint.pth.tar"):
    torch.save(state, filename)

# optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)

# Defining model, loss and optimizer
model = UNet(num_classes=1).to(device)

DICE_loss_function = DiceLoss()
BCE_loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

torch.manual_seed(42) #setting seed. Random number generator in PyTorch

epochs = 300 #define number of epochs

start_time = time.time() #timer start

DICE_train_loss = np.zeros(epochs)
BCE_train_loss = np.zeros(epochs)
combined_train_loss = np.zeros(epochs)

DICE_val_loss = np.zeros(epochs)
BCE_val_loss = np.zeros(epochs)
combined_val_loss = np.zeros(epochs)

for epoch in range(epochs):
    print(f"Epoch: {epoch+1}")
    # Training##########################################################
    model.train()
    print(f"Number of Batches: {len(train_dataloader)}")
    num=0
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X) #forward pass

        # Calculating losses
        DICE_loss = DICE_loss_function(y_pred, y)
        BCE_loss = BCE_loss_function(y_pred, y)
        loss = DICE_loss + BCE_loss
        combined_train_loss[epoch] += loss
        DICE_train_loss[epoch] += DICE_loss
        BCE_train_loss[epoch] += BCE_loss

        optimizer.zero_grad()
        loss.backward() #backward pass
        optimizer.step() 

        num += 1

        batch_time = time.time() #set batch time
        total_batch_time = batch_time-start_time #calculate batch time
        print(f"Elapsed Time (batch: {num}): {total_batch_time}")

    #Average loss per batch per epoch)
    combined_train_loss[epoch] /= len(train_dataloader)
    DICE_train_loss[epoch] /= len(train_dataloader)
    BCE_train_loss[epoch] /= len(train_dataloader)

    # Testing##############################################################
    model.eval()
    with torch.inference_mode(): #evaluating model not training!
        for X, y in test_dataloader: #loop over batches of dataset
            X, y = X.to(device), y.to(device) #connect to gpu or cpu

            y_pred = model(X) #fwd pass

            #calculate losses
            DICE_loss = DICE_loss_function(y_pred, y)
            BCE_loss = BCE_loss_function(y_pred, y)
            loss = DICE_loss + BCE_loss

            #sum losses for given batch
            combined_val_loss[epoch] += loss
            DICE_val_loss[epoch] += DICE_loss
            BCE_val_loss[epoch] += BCE_loss

        #need to divide the loss by the datasetloader length
        combined_val_loss[epoch] /= len(test_dataloader)
        DICE_val_loss[epoch] /= len(test_dataloader)
        BCE_val_loss[epoch] /= len(test_dataloader)
    epoch_time = time.time()
    total_epoch_time = epoch_time-start_time
    print(f"Elapsed Time: {total_epoch_time}")


    print(f"Train loss: {combined_train_loss[epoch]:.5f}, Train Dice: {DICE_train_loss[epoch]:.5f}, Train BCE: {BCE_train_loss[epoch]:.5f}, Test loss: {combined_val_loss[epoch]:.5f}, Test Dice: {DICE_val_loss[epoch]:.5f}, Test BCE: {DICE_val_loss[epoch]:.5f}\n")

    # Save checkpoint after every epoch
    checkpoint = { "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch
    }

    path = f"model_epoch_{epoch+1}.pth.tar"
    checkpoint_saving(checkpoint, filename=path)

    #visualize samples of input images
    # if __name__ == "__main__": #print only if running this file alone
    #     if epoch % 10 == 0:
    #         plt.subplot(231)
    #         plt.imshow(X[0, 0].cpu().detach().numpy(),cmap='gray')
    #         plt.axis('off')
    #         plt.subplot(232)
    #         plt.imshow(y[0, 0].cpu().detach().numpy(),cmap='gray')
    #         plt.axis('off')
    #         plt.subplot(233)
    #         plt.imshow(y_pred[0, 0].cpu().detach().numpy(),cmap='gray')
    #         plt.axis('off')
    #         plt.subplot(234)
    #         plt.imshow(X[12, 0].cpu().detach().numpy(),cmap='gray')
    #         plt.axis('off')
    #         plt.subplot(235)
    #         plt.imshow(y[12, 0].cpu().detach().numpy(),cmap='gray')
    #         plt.axis('off')
    #         plt.subplot(236)
    #         plt.imshow(y_pred[12, 0].cpu().detach().numpy(),cmap='gray')
    #         plt.axis('off')
    #         plt.show()

#plot the training curve
plt.title("DICE Training Curve")
plt.plot(range(1 ,epochs + 1), DICE_train_loss, label="Train")
plt.plot(range(1 ,epochs + 1), DICE_val_loss, label="Validation")
plt.legend(loc='best')
plt.xlabel("Epochs")
plt.ylabel("DICE")
plt.show()

plt.title("BCE Training Curve")
plt.plot(range(1 ,epochs + 1), BCE_train_loss, label="Train")
plt.plot(range(1 ,epochs + 1), BCE_val_loss, label="Validation")
plt.legend(loc='best')
plt.xlabel("Epochs")
plt.ylabel("BCE")
plt.show()

plt.title("DICE & BCE Training Curve")
plt.plot(range(1 ,epochs + 1), combined_train_loss, label="Train")
plt.plot(range(1 ,epochs + 1), combined_val_loss, label="Validation")
plt.legend(loc='best')
plt.xlabel("Epochs")
plt.ylabel("Combined DICE and BCE")
plt.show()
    


#load the checkpoint
def load_checkpoint(checkpoint_file, model, optimizer, lr):
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    #Need to update the learning rate
    for group in optimizer.param_groups:
        group['lr'] = lr


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


def predict_mask(model, device, image):
    """
    Predict the segmentation mask.

    Inputs:
    - model: trained
    - device: CPU or GPU
    - image: input tensor

    #Output:
    - predicted mask
    """

    model.eval() #put model in evaluation mode

    image = image.to(device) #move image to device

    if image.dim() == 3:
        image = image.unsqueeze(0)

    with torch.no_grad(): #don't want to compute gradient!
        output = model(image) #get the model output
        probs = torch.sigmoid(output) #sigmiod since logits layer
        preds = (probs > 0.5).float() #probability is 0.5 for creating mask
    return preds


test_image_tensor = torch.from_numpy(test_image_flair[:,:,slice_number]).unsqueeze(0).unsqueeze(0).float()
prediction = predict_mask(model, device, test_image_tensor)

# Visualizing the prediction
plt.imshow(prediction.squeeze().cpu(), cmap='gray')
plt.title (f'Predicted Segmentation: (Slice Number: {slice_number})')
plt.show()
plt.imshow(test_mask[:,:,slice_number])
plt.title("Actual Segmentation Mask")
plt.show()
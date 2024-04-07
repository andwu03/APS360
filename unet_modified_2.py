# This is a neural network for brain tumor segmentation,
# implemented with the U-Net architecture. 
# We will be tuning and verifying this against the baseline model. 

# Code adapted from: 
# https://www.kaggle.com/code/ankruteearora/tumor-segmentation


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


TRAIN_DATASET_PATH = 'C:/Users/grace/OneDrive/Surface Laptop Desktop/UofT/APS360/Project/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
# TRAIN_DATASET_PATH = "/home/andrew/APS360_Project/Data/MICCAI_BraTS2020_TrainingData/"


class UNet(nn.Module):
    """
    NNet is a convolutional network with a contracting and expanding path.

    Double convolutions are applied to each layer of the convolution.

    """

    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        # Contracting path.
        # Each convolution is applied twice.
        self.encode_conv_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.encode_conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.encode_conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.encode_conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.encode_conv_5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )

        # Expanding path.
        #transpose convolutional laters and expanding path
        self.decode_trans_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512,
            kernel_size=2,
            stride=2)
        self.decode_conv_1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.decode_trans_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256,
            kernel_size=2,
            stride=2)
        self.decode_conv_2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.decode_trans_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128,
            kernel_size=2,
            stride=2)
        self.decode_conv_3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.decode_trans_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64,
            kernel_size=2,
            stride=2)
        self.decode_conv_4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Conv2d( #output is the number of channels
            in_channels=64, out_channels=num_classes,
            kernel_size=1
        )

    def forward(self, x):
        encode_1 = self.encode_conv_1(x)
        encode_2 = self.max_pool2d(encode_1)
        encode_3 = self.encode_conv_2(encode_2)
        encode_4 = self.max_pool2d(encode_3)
        encode_5 = self.encode_conv_3(encode_4)
        encode_6 = self.max_pool2d(encode_5)
        encode_7 = self.encode_conv_4(encode_6)
        encode_8 = self.max_pool2d(encode_7)
        encode_9 = self.encode_conv_5(encode_8)

        decode_1 = self.decode_trans_1(encode_9)
        decode_1 = torch.cat([encode_7, decode_1], 1)
        decode_2 = self.decode_conv_1(decode_1)
        decode_2 = self.decode_trans_2(decode_2)
        decode_2 = torch.cat([encode_5, decode_2], 1)
        decode_3 = self.decode_conv_2(decode_2)
        decode_3 = self.decode_trans_3(decode_3)
        decode_3 = torch.cat([encode_3, decode_3], 1)
        decode_4 = self.decode_conv_3(decode_3)
        decode_4 = self.decode_trans_4(decode_4)
        decode_4 = torch.cat([encode_1, decode_4], 1)
        decode_5 = self.decode_conv_4(decode_4)

        out = self.out(decode_5)

        return out

model = UNet(num_classes=1)

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

epochs = 3 #define number of epochs

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
    combined_train_loss /= len(train_dataloader)
    DICE_train_loss /= len(train_dataloader)
    BCE_train_loss /= len(train_dataloader)

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
        combined_val_loss/= len(test_dataloader)
        DICE_val_loss /= len(test_dataloader)
        BCE_val_loss /= len(test_dataloader)
    epoch_time = time.time()
    total_epoch_time = epoch_time-start_time
    print(f"Elapsed Time: {total_epoch_time}")


    print(f"Train loss: {combined_train_loss[epoch]:.5f}, Train Dice: {DICE_train_loss[epoch]:.5f}, Train BCE: {BCE_train_loss[epoch]:.5f}, Test loss: {combined_val_loss[epoch]:.5f}, Test Dice: {DICE_val_loss[epoch]:.5f}, Test BCE: {DICE_val_loss[epoch]:.5f}\n")

    # Save checkpoint after every epoch
    checkpoint = { "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch
    }

    path = f"model_epoch_{epoch+1}.pth.tar"
    checkpoint_saving(checkpoint, filename=path)

#load the checkpoint
def load_checkpoint(checkpoint_file, model, optimizer, lr):
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    #Need to update the learning rate
    for group in optimizer.param_groups:
        group['lr'] = lr

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

if __name__ == "__main__":
    # Visualizing the prediction
    plt.imshow(prediction.squeeze().cpu(), cmap='gray')
    plt.title (f'Predicted Segmentation: (Slice Number: {slice_number})')
    plt.show()
    plt.imshow(test_mask[:,:,slice_number])
    plt.title("Actual Segmentation Mask")
    plt.show()
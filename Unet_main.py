# Import all the necessary modules

from __future__ import print_function, division
import os, sys, glob
import random
import time

import numpy as np
import cv2
import math

import imgaug as ia
import imgaug.augmenters as iaa
import scipy.ndimage
from scipy.ndimage import gaussian_filter, map_coordinates

from skimage import io, transform
import PIL
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch.utils.data as d
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from tensorboardX import SummaryWriter

from sklearn.metrics import confusion_matrix

from utils import *


# specify if we should use a GPU (cuda) or only the CPU
if(torch.cuda.is_available()):
    print(torch.cuda.get_device_properties(gpuid))
    torch.cuda.set_device(gpuid)
    device = torch.device(f'cuda:{gpuid}')
else:
    print('no GPU available')
    device = torch.device(f'cpu')


# build the model according to the paramters specified above and copy it to the GPU.
# finally print out the number of trainable parameters
model = UNet(n_classes=n_classes, in_channels=in_channels, padding=padding,depth=depth,wf=wf, up_mode=up_mode,
             batch_norm=batch_norm).to(device)
print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")

# Set random seed so that we can reproduce same splitting in the future
seed = random.randrange(sys.maxsize)
random.seed(seed) # set the seed
print(f"random seed (note down for reproducibility): {seed}")

# Load the dataset without augmentations to compute the different weights of the classes
dataset = create_dataset(path_to_imgs, path_to_masks, transform=None, edge_weight=edge_weight,
                         input_only=None)

totals = np.zeros(
    (2, len(classes)))  # we can to keep counts of all the classes in for in particular training, since we
totals[0, :] = classes  # can later use this information to create better weights

for i in range(len(dataset)):
    (img, mask, mask_weight) = dataset[i]

    for j, key in enumerate(
            classes):  # sum the number of pixels, this is done pre-resize, the but proportions don't change which is really what we're after
        totals[1, j] += torch.sum(mask == key)
# print to see if everything worked as expected
print(totals)

# Specify the weights for each class
class_weights = totals[1,0:2]
# They suggest only summing pixels from training set, I don't think it makes a huge difference but smth to keep in mind
class_weights = torch.from_numpy(1-class_weights/class_weights.sum()).type('torch.FloatTensor')
#class_weight = torch.from_numpy(1-class_weight/class_weight.sum()).type('torch.FloatTensor').to(device)

# print because.. you know why
print(class_weights)

criterion = nn.CrossEntropyLoss(weight = class_weights, ignore_index = ignore_index, reduction='none') #reduce = False makes sure we get a 2D output instead of a 1D "summary" value

augs = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    #iaa.CropToFixedSize(width=patch_size, height=patch_size),
    iaa.Sometimes(0.3, iaa.SomeOf((1,3), [
        iaa.Affine(rotate=(-45, 45),
                   shear=(-16,16),
                   translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
        iaa.Crop(percent=(0, 0.5)),
        iaa.PerspectiveTransform(scale=(0.01, 0.2)),
        # iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25, name="elastic")
    ])),
    iaa.Sometimes(0.3, iaa.SomeOf((1,3), [
        iaa.Add((-40, 40), per_channel=0.5, name="color-jitter"),
        iaa.Grayscale(alpha=(0.0, 1.0), name="gray"),
        iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True, name="hue"),
        iaa.Sometimes(0.25,
                     iaa.OneOf([
                         iaa.GaussianBlur(sigma=1.0, name="gauss"),
                         iaa.AverageBlur(k=(2,7), name="avg"),
                         iaa.MedianBlur(k=(3,11), name="med"),
                         iaa.AdditiveGaussianNoise(scale=0.1*255, name="add")
                     ]))
    ]))
])

# Initialize the dataset with augmentations
dataset = create_dataset(path_to_imgs, path_to_masks, transform=augs, edge_weight=edge_weight,
                         input_only=['color-jitter', 'gray', 'hue', 'gauss', 'avg', 'med', 'add'])


# print(dataset[0][0]) --> first is index as specified by __getitem__, second is index of img = 0 or mask = 1

# Splits data into training and validation sets into the respective entry of the data dictionary
data = {}

data["train"], data["val"] = d.random_split(dataset, lengths=[int(np.floor(len(dataset)*(1-test_size))),
                                                      int(np.ceil(len(dataset)*test_size))])

print(len(data["train"])+len(data["val"]))

# img_type = {"img": 0, "mask": 1}

# Look at an example of your training set to see if everything works as expected
(img, mask, mask_weight) = data["val"][0] # img_idx = 0, mask_idx = 1

print(img.shape)

fig, ax = plt.subplots(1,3, figsize=(10,4))  # 1 row, 2 columns, firts pannel: original image, second pannel: respective mask

#build output showing original image, mask and weighting mask (all after augmentation, if specified as such)
ax[0].imshow(np.moveaxis(img.numpy(),0,-1))
ax[1].imshow(mask)
ax[2].imshow(mask_weight)

# print(dataset[0][0]) --> first is index as specified by __getitem__, second is index of img = 0 or mask = 1

dataLoader = {}
for phase in phases:  # now for each of the phases, we're creating the dataloader
    # interestingly, given the batch size, i've not seen any improvements from using a num_workers>0

    dataLoader[phase] = DataLoader(data[phase], batch_size=batch_size,
                                   shuffle=True, num_workers=0, pin_memory=True)  # With GPU usage num_workers=8

optim = torch.optim.Adam(model.parameters()) #adam is going to be the most robust, though perhaps not the best performing, typically a good place to start
# optim = torch.optim.SGD(model.parameters(),
#                           lr=.1,
#                           momentum=0.9,
#                           weight_decay=0.0005)

# writer=SummaryWriter() #open the tensorboard visualiser
best_loss_on_test = np.Infinity
edge_weight = torch.tensor(edge_weight).to(device)
# edge_weight=edge_weight.clone().detach().to(device)
start_time = time.time()
for epoch in range(num_epochs):
    # zero out epoch based performance variables
    all_acc = {key: 0 for key in phases}
    all_loss = {key: torch.zeros(0).to(device) for key in phases}
    cmatrix = {key: np.zeros((2, 2)) for key in phases}

    for phase in phases:  # iterate through both training and validation states

        if phase == 'train':
            model.train()  # Set model to training mode
        else:  # when in eval mode, we don't want parameters to be updated
            model.eval()  # Set model to evaluate mode

        for ii, (X, y, y_weight) in enumerate(dataLoader[phase]):  # for each of the batches
            X = X.to(device)  # [Nbatch, 3, H, W]
            y_weight = y_weight.type('torch.FloatTensor').to(device)
            y = y.type('torch.LongTensor').to(device)  # [Nbatch, H, W] with class indices (0, 1)

            with torch.set_grad_enabled(
                    phase == 'train'):  # dynamically set gradient computation, in case of validation, this isn't needed
                # disabling is good practice and improves inference time

                prediction = model(X)  # [N, Nclass, H, W]
                # print("Shape of y_pred is:", prediction.shape)
                loss_matrix = criterion(prediction, y)
                loss = (loss_matrix * (edge_weight ** y_weight)).mean()  # can skip if edge weight==1

                if phase == "train":  # in case we're in train mode, need to do back propogation
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    train_loss = loss

                all_loss[phase] = torch.cat((all_loss[phase], loss.detach().view(1, -1)))

                if phase in validation_phases:  # if this phase is part of validation, compute confusion matrix
                    p = prediction[:, :, :, :].detach().cpu().numpy()
                    cpredflat = np.argmax(p, axis=1).flatten()
                    yflat = y.cpu().numpy().flatten()

                    cmatrix[phase] = cmatrix[phase] + confusion_matrix(yflat, cpredflat, labels=range(n_classes))

        all_acc[phase] = (cmatrix[phase] / cmatrix[phase].sum()).trace()
        all_loss[phase] = all_loss[phase].cpu().numpy().mean()

        """
        #save metrics to tensorboard
        writer.add_scalar(f'{phase}/loss', all_loss[phase], epoch)
        if phase in validation_phases:
            writer.add_scalar(f'{phase}/acc', all_acc[phase], epoch)
            writer.add_scalar(f'{phase}/TN', cmatrix[phase][0,0], epoch)
            writer.add_scalar(f'{phase}/TP', cmatrix[phase][1,1], epoch)
            writer.add_scalar(f'{phase}/FP', cmatrix[phase][0,1], epoch)
            writer.add_scalar(f'{phase}/FN', cmatrix[phase][1,0], epoch)
            writer.add_scalar(f'{phase}/TNR', cmatrix[phase][0,0]/(cmatrix[phase][0,0]+cmatrix[phase][0,1]), epoch)
            writer.add_scalar(f'{phase}/TPR', cmatrix[phase][1,1]/(cmatrix[phase][1,1]+cmatrix[phase][1,0]), epoch)
        """

    print('%s ([%d/%d] %d%%), train loss: %.4f test loss: %.4f' % (timeSince(start_time, (epoch + 1) / num_epochs),
                                                                   epoch + 1, num_epochs,
                                                                   (epoch + 1) / num_epochs * 100, all_loss["train"],
                                                                   all_loss["val"]), end="")

    # if current loss is the best we've seen, save model state with all variables
    # necessary for recreation
    if all_loss["val"] < best_loss_on_test:
        best_loss_on_test = all_loss["val"]
        print("  **")
        state = {'epoch': epoch + 1,
                 'model_dict': model.state_dict(),
                 'optim_dict': optim.state_dict(),
                 'best_loss_on_test': all_loss,
                 'n_classes': n_classes,
                 'in_channels': in_channels,
                 'padding': padding,
                 'depth': depth,
                 'wf': wf,
                 'up_mode': up_mode, 'batch_norm': batch_norm}

        torch.save(state, f"{dataname}_unet_best_model.pth")
    else:
        print("")

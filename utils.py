import imgaug as ia
import imgaug.augmenters as iaa

import os, sys, glob
import random
import time
import cv2
import scipy.ndimage
import numpy as np
import math
from PIL import Image

import torch
from torch import nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch.utils.data as d
from torchvision import transforms, utils

from datetime import datetime

# Initalize paths

path_to_imgs_unprocessed = os.path.join('dataset', 'imgs_unprocessed')
path_to_masks_unprocessed = os.path.join('dataset', 'masks_unprocessed')
path_to_imgs = os.path.join('dataset', 'imgs_processed')
path_to_masks = os.path.join('dataset', 'masks_processed')

path_to_data_unprocessed = [path_to_imgs_unprocessed, path_to_masks_unprocessed]
path_to_data = [path_to_imgs, path_to_masks]

path_to_output = os.path.join('dataset', 'prediction_output')

Image.MAX_IMAGE_PIXELS = 1000000000
fixed_rows = 2000  # Set rows to be fixed for resizing
max_cols = 0  # Initialize maximum Columns
resized = 1024  # 2^10

# Set random seed so that we can reproduce same splitting in the future
seed = random.randrange(sys.maxsize)
random.seed(seed) # set the seed
print(f"random seed (note down for reproducibility): {seed}")

now = datetime.now()
date = now.strftime("%d-%b-%Y")

dataname = "pancreas"

"""
# If working on the server initialize path accordingly
path_to_masks = os.path.join('bitcel00@login.scicore.unibas.ch:', 'scicore', 'home', 'donath', 'bitcel00', 'LabRotation_nobackup', 'dataset_1024', 'masks')
# path_to_masks = os.path.join('dataset', 'masks', 'pancr_labeled')
path_to_imgs = os.path.join('bitcel00@login.scicore.unibas.ch:', 'scicore', 'home', 'donath', 'bitcel00', 'LabRotation_nobackup', 'dataset_1024', 'imgs')
"""


# --- unet params
# these parameters get fed directly into the UNET class, and more description of them can be discovered there
ignore_index = -100
classes = [0, 1]
n_classes = len(classes)  # number of classes in the data mask that we'll aim to predict
in_channels = 3  # input channel of the data, RGB = 3
padding = True   # should levels be padded
depth = 5       # depth of the network
wf = 2           # wf (int): number of filters in the first layer is 2**wf, was 6
up_mode = 'upconv'  # should we simply upsample the mask, or should we try and learn an interpolation
batch_norm = True  # should we use batch normalization between the layers

# --- training params
batch_size = 3
# patch_size = 256
# mirror_pad = 50
num_epochs = 100
test_size = 0.1
# edges tend to be the most poorly segmented given how little area they occupy in the training set, this paramter
# boosts their values along the lines of the original UNET paper
edge_weight = 1.1
phases = ["train", "val"]  # how many phases did we create databases for?
# when should we do valiation? note that validation is time consuming,
# so as opposed to doing for both training and validation, we do it only for vlaidation at the end of the epoch
validation_phases = ["val"]

"""
Takes image as pixel array and user defined row number for resized picture as input
and resizes image keeping the aspect ratio
(For later: check if resizing before adjusting is problematic)
"""
def resizeKeepRatio(img_arr, fixed_rows):
    rowPercent = (fixed_rows / float(img_arr.shape[0]))
    new_cols = int((float(img_arr.shape[1]) * float(rowPercent)))
    img = cv2.resize(img_arr, (new_cols, fixed_rows))
    return img

"""
Takes image pixel matrix and user defined padding number as input and adds 0s to the 
left and the right of the image to homogenize the data (same dimension for every image)
"""

# PAD IMAGE FOR RGB IMAGE
def adjustImagesRGB(img, addpad, horizontal=False, vertical=False):
    if horizontal:
        img = np.pad(img, ((0, 0), (int(np.floor(addpad/2)), int(np.ceil(addpad/2))), (0, 0)), 'constant')
    if vertical:
        img = np.pad(img, ((int(np.floor(addpad/2)), int(np.ceil(addpad/2))), (0, 0), (0, 0)), 'constant')

    return img



# PAD IMAGE FOR MASKS
def adjustImagesMask(img, addpad, horizontal=False, vertical=False):
    if horizontal:
        img = np.pad(img, ((0, 0), (int(np.floor(addpad/2)), int(np.ceil(addpad/2)))), 'constant',
                    constant_values=1)
    if vertical:
        img = np.pad(img, ((int(np.floor(addpad/2)), int(np.ceil(addpad/2))), (0, 0)), 'constant',
                    constant_values=1)
    return img


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=6,
        padding=False,
        batch_norm=False,
        up_mode='upconv',
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out

#helper function for pretty printing of current time and remaining time
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent+.00001)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# Class to costumize my own dataset
class create_dataset(d.Dataset):
    def __init__(self, path_to_imgs=None, path_to_masks=None, transform=None, edge_weight=edge_weight, input_only=None):
        # Initalize constructor arguments

        self.edge_weight = edge_weight
        self.input_only = input_only
        self.img_files = []
        self.path_to_imgs = path_to_imgs
        self.transform = transform

        # search recursively from the root directory for the raw images (end with 'RGB.tif')

        img_files = glob.glob('**/*_RGB.tif', recursive=True)

        for fname in img_files:
            self.img_files.append(fname)
            # print(fname)

        # Save the number of files as length in order to overwrite the __len__ module
        self.len = len(self.img_files)

        # Same for masks
        self.mask_files = []
        self.path_to_masks = path_to_masks

        # search recursively from the root directory for the masks (end with 'pancr_labeled.png')
        mask_files = glob.glob('**/*pancr_labeled.png', recursive=True)

        for fname in mask_files:
            self.mask_files.append(fname)
            # print(fname)

    def _activator_masks(self, images, augmenter, parents, default):
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default

    def __getitem__(self, index):
        # Save the respective image and mask accroding to the index parameter
        img = cv2.cvtColor(cv2.imread(self.img_files[index]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_files[index])

        # User can choose, if he/she wants to define edge_weights to the model
        if (self.edge_weight):
            weight = scipy.ndimage.morphology.binary_dilation(mask, iterations=2) & ~mask
        else:  # otherwise the edge weight is all ones and thus has no affect
            weight = np.ones(mask.shape, dtype=mask.dtype)

        img_new = img
        mask_new = mask
        weight_new = weight

        # If user chooses to transform images, this is initialized in this step
        # return images, masks and weight in this specific order
        if self.transform:
            det_tf = self.transform.to_deterministic()
            img_new = det_tf.augment_image(img_new)
            mask_new = det_tf.augment_image(
                mask_new,
                hooks=ia.HooksImages(activator=self._activator_masks))
            weight_new = det_tf.augment_image(
                weight_new,
                hooks=ia.HooksImages(activator=self._activator_masks))


        img_new = img_new[:, :, ::-1] - np.zeros_like(img_new)
        mask_new = mask_new[..., ::-1] - np.zeros_like(mask_new)
        mask_new = np.round(mask_new / 255, decimals=0)
        weight_new = weight_new[..., ::-1] - np.zeros_like(weight_new)


        return TF.to_tensor(img_new), torch.squeeze(TF.to_tensor(mask_new[:, :, 0])), torch.squeeze(
            TF.to_tensor(weight_new[:, :, 0]))

    def __len__(self):
        return self.len

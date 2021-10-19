import os
import random
import os
import glob

import numpy as np
from PIL import Image
import os
import cv2
from matplotlib import pyplot as plt

import torch
from torch import nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from utils import *

device = torch.device(f'cpu')

model = UNet(n_classes=n_classes, in_channels=in_channels, padding=padding,depth=depth,wf=wf,
             up_mode=up_mode, batch_norm=batch_norm).to(device)

dataname = "pancreas"

checkpoint = torch.load(f"{dataname}_unet_best_model.pth")
model.load_state_dict(checkpoint["model_dict"])

for filename in os.listdir(path_to_imgs_unprocessed):
    if filename.endswith('.tif'):
        print("Processed image is:", filename)

        horizontal = False
        vertical = False

        img = cv2.imread(os.path.join(path_to_imgs_unprocessed, filename))

        print("original image size is: ", img.shape)

        row_ori = img.shape[0]
        col_ori = img.shape[1]

        im_resized = resizeKeepRatio(img, fixed_rows)  # Resize images and fix number of rows
        print("dimensions of keep-ratio image is:", im_resized.shape)
        new_row = im_resized.shape[0] # = fixed_row = 2000
        print("new_row", new_row)
        new_col = im_resized.shape[1]
        print("new_col: ", new_col)

        if new_col > fixed_rows:
            vertical = True
        if new_col < fixed_rows:
            horizontal = True

        add_pad = abs(fixed_rows - im_resized.shape[1])

        im_pad = adjustImagesRGB(im_resized, abs(add_pad), horizontal=horizontal, vertical=vertical)
        print("dimensions of padded image:", filename, "is", im_pad.shape)

        im_resized = cv2.resize(im_pad, (resized, resized))
        print("dimensions of resized image:", filename, "is", im_resized.shape)

        if horizontal==True:
            percent = new_row / im_resized.shape[0]
        if vertical==True:
            percent = new_col / im_resized.shape[1]

        output = model(TF.to_tensor(im_resized)[None, ::].to(device))
        output = output.detach().squeeze().cpu().numpy()
        output = np.moveaxis(output, 0, -1)
        output.shape

        prediction = np.argmax(output, axis=2)
        print("dimension of predicted image is:", prediction.shape)

        #prediction_ori_size = cv2.pyrUp(prediction, prediction[new_col, new_col])

        prediction = np.uint8(prediction)
        upscaled_image = cv2.resize(prediction, None, fx=percent, fy=percent)

        print("upscaled image dimension: ", upscaled_image.shape)

        if horizontal==True:
            upscaled_nopad = upscaled_image[:, (int(np.floor(add_pad/2))):(int(new_row-np.ceil(add_pad/2)))]


        if vertical==True:
            upscaled_nopad = upscaled_image[(int(np.floor(add_pad/2))):(int(new_col-np.ceil(add_pad/2))), :]

        print("Dimension of upscaled nopad is:", upscaled_nopad.shape)


        ori_percent = row_ori / new_row
        print("ori_percent is:", ori_percent)

        # print("ori_col_percent is: ", ori_col_percent)
        #print("ori")
        reconstructed_image = cv2.resize(upscaled_nopad, None, fx=ori_percent, fy=ori_percent)

        print("reconstructed image size is:", reconstructed_image.shape)

        cv2.imwrite(os.path.join(path_to_output, os.path.basename(filename)).replace('(RGB).tif', 'predicted.png'),
                    reconstructed_image.astype('uint8') * 255)



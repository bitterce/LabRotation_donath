"""
-Script to resize images and add padding to homogenize the data
-Input data: Masks and raw images both in tif format
-Output data: Raw images in tif, masks in png
-author: Céline Bitter
-date: 30.09.19
-python 3.6 (needed to use tensorflow)
"""

import numpy as np
from PIL import Image
import os
import cv2
from matplotlib import pyplot as plt
from utils import *

print(path_to_data)
print(path_to_output)

# Loop that iterates through input folder
print(os.listdir(path_to_imgs_unprocessed))

for directory in path_to_data_unprocessed:
    for filename in os.listdir(directory):
        if filename.endswith(".tif"):  # We are only interested in tifs
            horizontal = False
            vertical = False
            im = cv2.imread(os.path.join(directory, filename))  # Save image into variable
            # only use when RBG image
            # im = normalizeImage(im)[:, :, :3]  # Normalize and drop 4th channel (alpha transparency) to recover true RGB img
            if directory == path_to_data_unprocessed[1]:
                im = im[:, :, 0] / 255

            im_resized = resizeKeepRatio(im, fixed_rows)  # Resize images and fix number of rows

            if im_resized.shape[1] > fixed_rows:
                vertical = True
            if im_resized.shape[1] < fixed_rows:
                horizontal = True

            if directory == path_to_data_unprocessed[1]:
                im_pad = adjustImagesMask(im_resized, abs(fixed_rows - im_resized.shape[1]), horizontal=horizontal,
                                          vertical=vertical)  # Add padding
                im_pad = 1 - im_pad  # Invert the image as NN takes 0 as background and 1 as pancreas
                im_resized = cv2.resize(im_pad, (resized, resized))
                print("dimensions of resized image:", filename, "is", im_resized.shape)
                cv2.imwrite(os.path.join(path_to_data[1], os.path.basename(filename).replace(".tif", ".png")),
                            im_resized.astype('uint8') * 255)  # Save image as png
            else:
                im_pad = adjustImagesRGB(im_resized, abs(fixed_rows - im_resized.shape[1]), horizontal=horizontal,
                                         vertical=vertical)
                #print("dimensions of padded image:", filename, "is", im_pad.shape)
                im_resized = cv2.resize(im_pad, (resized, resized))
                print("dimensions of resized image:", filename, "is", im_resized.shape)
                cv2.imwrite(os.path.join(path_to_data[0], os.path.basename(filename).replace("(RGB)", "RGB")),
                            im_resized)
        else:
            continue

print("Program ended!")

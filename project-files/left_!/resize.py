from skimage.io import imread
from skimage.io import imsave
from scipy import misc
import argparse
import os
from PIL import Image
import numpy as np

image_dir = os.getcwd() + "/path_png_resize/"
image_dir1 = os.getcwd() + "/path_png_resize_updated/"

def average(pixel):
    x = 0.299*pixel[1] + 0.587*pixel[2] + 0.114*pixel[3]
    #print(0.299*pixel[0] , 0.587*pixel[1], 0.114*pixel[2])
    return 1- (x / 3)

for f in os.listdir(image_dir):
    filename = os.fsdecode(f)
    print(filename)
    image = imread(image_dir+"/"+filename)
    grey = np.zeros((image.shape[0], image.shape[1]))
    for rownum in range(len(image)):
        for colnum in range(len(image[rownum])):
            grey[rownum][colnum] = average(image[rownum][colnum])
    misc.imsave(image_dir1+"/"+filename, grey)
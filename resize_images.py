#!/usr/bin/python
from PIL import Image
import os, sys

path = "./data/datasets/DIV2K/DIV2K_valid_HR/"
dirs = os.listdir(path)

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            # file, extention = os.path.splitext(path+item)
            imResize = im.resize((32,32), Image.BICUBIC)
            print(path +'resized/' + item)
            imResize.save(path + 'resized/' + item)

resize()
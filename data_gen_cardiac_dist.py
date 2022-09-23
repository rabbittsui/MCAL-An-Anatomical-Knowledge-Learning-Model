'''Load image/labels/boxes from an annotation file.

The list file is like:

    img.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
'''
from __future__ import print_function

import os
import sys
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt as distance

#from transform1 import randonm_resize, random_rotate, rotate_resize


class ListDataset(data.Dataset):
    def __init__(self, root, list_file, state, k):
        self.root = root
        self.state= state
        self.fnames = []
        if self.state == 'Train':
            with open(list_file) as f:
                lines = f.readlines()
            len_line=len(lines)
            self.fnames.extend(lines[:int((k % 10) * len_line / 10)] )
            self.fnames.extend(lines[int((k % 10 + 1) * len_line / 10):-1])
            self.num_samples = len(self.fnames)
        if self.state == 'Valid':
            with open(list_file) as f:
                lines = f.readlines()
            len_line = len(lines)
            self.fnames.extend(lines[int((k % 10) * len_line /10):int((k % 10 + 1) * len_line /10)])
            self.num_samples = len(self.fnames)
        if self.state == 'test':
            with open(list_file) as f:
                lines = f.readlines()
            self.fnames.extend(lines)
            self.num_samples = len(self.fnames)


    def __getitem__(self, idx):
        fname = self.fnames[idx][:-1] + '.png'
        mask_fname = self.fnames[idx][:-1] + '_gt.png'
        image_left = Image.open(os.path.join(self.root[0], fname))
        image_right = Image.open(os.path.join(self.root[1], mask_fname))
        resize = transforms.Resize(size=(512, 512))
        image_left = resize(image_left)
        image_right = resize(image_right)
        if self.state  == 'Train':
            # Resize
            '''
            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                image_left, output_size=(512, 512))

            image_left = TF.crop(image_left, i, j, h, w)
            image_right = TF.crop(image_right, i, j, h, w)
            '''
            # Random translation
            # ret = random.uniform(0.0, 0.3)
            translate = transforms.RandomAffine(0, translate=(0, 0.3), scale=None, shear=0, resample=False, fillcolor=0)

            image_left = translate(image_left)
            image_right = translate(image_right)
            # Random horizontal flipping
            if random.random() > 0.5:
                image_left = TF.hflip(image_left)
                image_right = TF.hflip(image_right)

            rotate = transforms.RandomAffine(20, translate=None, scale=None, shear=0, resample=False, fillcolor=0)
            image_left = rotate(image_left)
            image_right = rotate(image_right)
            # s = random.uniform(0.7,1.3)
            rscale = transforms.RandomAffine(0, translate=None, scale=(0.7, 1.3), shear=0, resample=False, fillcolor=0)
            image_left = rscale(image_left)
            image_right = rscale(image_right)

            '''
            # Random vertical flipping
            if random.random() > 0.5:
                image_left = TF.vflip(image_left)
                image_right = TF.vflip(image_right)
            '''
        dm_img = self.one_hot2dist(image_right)
        image_left = TF.to_tensor(image_left)
        image_right = TF.to_tensor(image_right)
        dm_img = TF.to_tensor(dm_img)
        return image_left, image_right,dm_img

    def one_hot2dist(self, seg):
        seg = np.where(np.array(seg) == 2, 1, 0)
        res = np.zeros_like(seg)
        posmask = seg.astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            res = distance(negmask) * negmask - distance(posmask-1)  * posmask#boundary loss
            #res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
        return res

    def __len__(self):
        return self.num_samples
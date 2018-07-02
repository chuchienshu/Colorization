# from __future__ import division
import torch
import torch.utils.data as data
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize, rescale

import os, math, random
from os.path import *
import numpy as np

from glob import glob
from utils import frame_utils
from scipy.misc import imread, imresize
from utils.img_transforms import *
# import natsort
from torchvision.transforms import RandomRotation, Resize, Compose

class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw),:]

class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[int((self.h-self.th)//2):int((self.h+self.th)//2), int((self.w-self.tw)//2):int((self.w+self.tw)//2),:]
'''
comp = Compose([
    Scale([286, 286]),
    RandomCrop([224,224]),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    # RandomRotate(11)
])
'''
class Image_from_folder(data.Dataset):

    def __init__(self, args ):
        super().__init__()
        self.replicates = args['replicates']
        self.render_size = []
        self.gt_images = []
        self.rr = Random_Rotate(9)

        self.train = args['train']
        self.gt_images = glob(args['file'])

        if not self.train:
            #only choose 300 images for validation
            self.gt_images = random.sample( self.gt_images, 300)
        self.size = len(self.gt_images)

        self.frame_size = frame_utils.read_gen(self.gt_images[0]).shape

        if  (self.frame_size[0]%64) or (self.frame_size[1]%64):

            self.render_size.append( ((self.frame_size[0])//64)  *64)
            self.render_size.append( ( (self.frame_size[1])//64)  * 64)
        else:
            self.render_size.append( self.frame_size[0])
            self.render_size.append( self.frame_size[1])

    def __getitem__(self, index):

        index = index % self.size
        img = frame_utils.read_gen(self.gt_images[index])

        if self.train:
            img = resize(img ,(224, 224)) 
        else:
            img = resize(img, self.render_size)

        img = rgb2lab(img)   

        img = np.array(img).transpose(2,0,1)
        img = torch.from_numpy(img.astype(np.float32))

        return   img

    def __len__(self):
        return self.size * self.replicates



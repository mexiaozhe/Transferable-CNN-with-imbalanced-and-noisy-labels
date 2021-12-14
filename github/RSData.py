import torch
import os, sys
from osgeo import gdal
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms as T

class RSData(data.Dataset):
    def __init__(self, data_dir, transforms=True, mode='train', suffix='.bmp',
                            mean=[94.96,87.30,88.19,109.74,115.21,99.27], 
                            std=[27.91,25.40,27.31,25.93,27.67,28.84],test_base='clip'):
        self.mode = mode
        self.transforms = transforms
        self.images={}
        self.labels={}
        if mode == 'test':
            self.images[mode] = self.recursive_glob(rootdir=test_base, suffix=suffix)
            if not self.images[mode]:
                raise Exception("No files for [%s] found in %s" % (mode, self.images_base))
            print("Found %d %s images" % (len(self.images[mode]), mode))
            self.file_count = len(self.images[mode])
        elif mode == 'train':
            images_base = os.path.join(data_dir, 'image_train')
            labels_base = os.path.join(data_dir, 'label_train')
            self.images[mode] = self.recursive_glob(rootdir=images_base, suffix=suffix)
            self.labels[mode] = self.recursive_glob(rootdir=labels_base, suffix=suffix)        
            if not self.images[mode]:
                raise Exception("No files for [%s] found in %s" % (mode, self.images_base))
            print("Found %d %s images" % (len(self.images[mode]), mode))
            if not self.labels[mode]:
                raise Exception("No files for [%s] found in %s" % (mode, self.labels_base))
            print("Found %d %s labels" % (len(self.labels[mode]), mode))
            self.file_count = len(self.images[mode])
        else:
            images_base = os.path.join(data_dir, 'image_val')
            labels_base = os.path.join(data_dir, 'label_val')
            self.images[mode] = self.recursive_glob(rootdir=images_base, suffix=suffix)
            self.labels[mode] = self.recursive_glob(rootdir=labels_base, suffix=suffix)        
            if not self.images[mode]:
                raise Exception("No files for [%s] found in %s" % (mode, self.images_base))
            print("Found %d %s images" % (len(self.images[mode]), mode))
            if not self.labels[mode]:
                raise Exception("No files for [%s] found in %s" % (mode, self.labels_base))
            print("Found %d %s labels" % (len(self.labels[mode]), mode))
            self.file_count = len(self.images[mode])

        
        self.transforms = T.Compose([T.ToTensor(),
                                     T.Normalize(mean, std)])

        self.transform = T.Compose([T.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.1, hue=0)])

        self.augmentation = T.Compose([T.RandomHorizontalFlip(),
                                       T.RandomVerticalFlip()])

    def aug(self, img):
        alpha = np.random.rand(1)
        beta = int(255*np.random.rand(1))
        num_band = int(img.shape[0]*np.random.rand(1))
        band=torch.zeros(num_band)
        for i in range(num_band):
            band=int(img.shape[0]*np.random.rand(1))
            img[band,:,:] = alpha*img[band,:,:]+beta
            img[img>255] = -img[img>255]
            img[img<0] = -img[img<0]
        return img

    def __getitem__(self, index):
        mode = self.mode
        
        if mode == 'test':
            img_path = self.images[mode][index]
            img = gdal.Open(img_path)
            img = img.ReadAsArray()
            img = img.transpose(1, 2, 0)
            if self.transforms:
                img = self.transforms(img)
            return img.float()
        else:
            img_path = self.images[mode][index]
            lbl_path = self.labels[mode][index]
            img = gdal.Open(img_path)
            lbl = gdal.Open(lbl_path)
            img = img.ReadAsArray()
            lbl = lbl.ReadAsArray()
            if np.random.rand(1)>0.75:
                img = self.aug(img)


            img = img.transpose(1, 2, 0)
            if self.transforms:
                img = self.transforms(img)       
            return img, lbl

    
    def __len__(self):
        return self.file_count

    def recursive_glob(self, rootdir='.', suffix=''):
        return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]
            
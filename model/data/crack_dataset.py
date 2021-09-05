import os
import numpy as np
import cv2
from PIL import Image
from copy import copy
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch

from torch.utils.data import DataLoader, Dataset
from model.data.transforms.transforms import ToTensor

class CrackDataSet(Dataset):
    def __init__(self, cfg, image_dir , seg_dir, transforms=None, sr_transforms=None):      
        print("image_dir", image_dir)
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.fnames = [path.name for path in Path(image_dir).glob('*.jpg')]
        self.img_transforms = transforms
        self.sr_transforms = sr_transforms

    def __getitem__(self, i):
        fname = self.fnames[i]
        fpath = os.path.join(self.image_dir, fname)
        img = np.array(Image.open(fpath)) #Image.open(fpath)
        spath = os.path.join(self.seg_dir, fname)

        seg_target = np.array(Image.open(spath))[:, :, np.newaxis] # HxWxC
        img, seg_target = self.img_transforms(img, seg_target)
        sr_target = copy(img)
        img = self.sr_transforms(img)

        return img, sr_target, seg_target

    def __len__(self):
        return len(self.fnames)


class CrackDataSetTest(Dataset):
    def __init__(self, cfg, image_dir , seg_dir, transforms=None, sr_transforms=None):
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.fnames = [path.name for path in Path(image_dir).glob('*.jpg')]
        self.img_transforms = transforms
        self.sr_transforms = sr_transforms

    def __getitem__(self, i):
        fname = self.fnames[i]
        fpath = os.path.join(self.image_dir, fname)
        img = np.array(Image.open(fpath)) #Image.open(fpath)
        spath = os.path.join(self.seg_dir, fname)

        seg_target = np.array(Image.open(spath))[:, :, np.newaxis] # HxWxC
        img, seg_target = self.img_transforms(img, seg_target)
        sr_target = copy(img)
        img = self.sr_transforms(img)
    
        return img, sr_target, seg_target, fname

    def __len__(self):
        return len(self.fnames)

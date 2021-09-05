import os
import sys
import math
import torch
import numpy as np
import cv2

import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

import torch
from torch import nn

from scipy.ndimage.morphology import distance_transform_edt as edt


class IoU:
    """Intersection over Union
    output and target have range [0, 1]"""

    def __init__(self, th=0.5):
        self.name = "IoU"
        self.th = 0.5

    def __call__(self, output, target):
        smooth = 1e-5

        if torch.is_tensor(output):
            output = output.data.cpu().numpy()
        if torch.is_tensor(target):
            target = target.data.cpu().numpy()
        output = output > self.th
        target = target > self.th
        intersection = (output & target).sum(axis=(1,2,3))
        union = (output | target).sum(axis=(1,2,3))

        # print("iouuuuuuuu", (intersection + smooth) / (union + smooth))
        return (intersection + smooth) / (union + smooth)


# https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/metrics/psnr.py#L4

class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2, [1,2,3]) # batch dim retained
        # print("mseeeeeeee", mse)  
        return (10 * torch.log10(1 / mse.to('cpu')).detach().numpy().copy())
        # return (20 * torch.log10(255.0 / torch.sqrt(mse)).to('cpu')).detach().numpy().copy()


# https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        # print("ssimmmmm", ssim_map.mean(1).mean(1).mean(1))
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = False):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average).to('cpu').detach().numpy().copy()

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


# https://github.com/PatRyg99/HausdorffLoss/blob/master/hausdorff_metric.py
class HausdorffDistance:
    def hd_distance(self, x: np.ndarray, y: np.ndarray):

        if np.count_nonzero(x) == 0 or np.count_nonzero(y) == 0:
            return np.array([np.Inf])

        indexes = np.nonzero(x)
        distances = edt(np.logical_not(y))

        return np.max(distances[indexes])

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
        assert (
            pred.shape[1] == 1 and target.shape[1] == 1
        ), "Only binary channel supported"

        pred = pred.byte()
        target = target.byte()
        result_hd = np.empty((pred.shape[0], 2))

        for batch_idx in range(pred.shape[0]):
            result_hd[batch_idx, 0] = self.hd_distance(pred[batch_idx].cpu().numpy(), target[batch_idx].cpu().numpy())
            result_hd[batch_idx, 1] = self.hd_distance(target[batch_idx].cpu().numpy(), pred[batch_idx].cpu().numpy())

        result_hd = np.max(result_hd, axis=1)    
        # print(result_hd)

        return result_hd  # np.max(right_hd, left_hd)
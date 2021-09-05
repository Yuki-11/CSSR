from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import torch
import torch.nn as nn
import numpy as np

"""
original code: https://github.com/LIVIAETS/boundary-loss
refarence from: https://github.com/JunMa11/SegWithDistMap

"""

class BoundaryLoss(nn.Module):
    """
    compute boundary loss for binary segmentation
    input: outputs_soft: sigmoid results,  shape=(b,c,x,y)
           gt_sdf: sdf of ground truth (can be original or normalized sdf); shape=(b,c,x,y)
    output: boundary_loss; sclar
    """
    
    def __init__(self, class_idx=1):
        super(BoundaryLoss, self).__init__()
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.class_idx = class_idx

    def __call__(self, outputs_soft, label_batch):  
        # print(outputs_soft.shape)
        gt_sdf_npy = compute_sdf1_1(label_batch.cpu().numpy(), outputs_soft.shape)
        gt_sdf = torch.from_numpy(gt_sdf_npy).float().cuda(outputs_soft.device.index)
        pc = outputs_soft.type(torch.float32)
        dc = gt_sdf.type(torch.float32)

        multipled = torch.einsum("bcwh, bcwh->bcwh", pc, dc)
        loss = multipled.mean(dim=(1,2,3))

        return loss

def compute_sdf1_1(img_gt, out_shape):
    """
    compute the normalized signed distance map of binary mask
    input: segmentation, shape = (batch_size, c, x, y)
    output: the Signed Distance Map (SDM) 
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1, 1]
    """

    img_gt = img_gt.astype(np.uint8)

    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
            # ignore background
        posmask = img_gt[b].astype(np.bool)
        for c in range(1, out_shape[1]):
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                sdf[boundary==1] = 0
                normalized_sdf[b][c] = sdf

    return normalized_sdf

def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM) 
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    """

    img_gt = img_gt.astype(np.uint8)

    gt_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        posmask = img_gt[b].astype(np.bool)
        for c in range(1, out_shape[1]):
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = negdis - posdis
                sdf[boundary==1] = 0
                gt_sdf[b][c] = sdf

    return gt_sdf




                
            
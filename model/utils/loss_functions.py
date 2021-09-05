import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from model.utils.boundary_loss import BoundaryLoss

class BoundaryComboLoss(nn.Module):  
    def __init__(self, per_epoch, smooth=10 ** -8, reduction='none', pos_weight=[1, 1], loss_weight=[1, 1], alpha_min=0.01, decrease_ratio=1):
        super(BoundaryComboLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.alpha = 1.0
        self.alpha_min = alpha_min
        self.fix_alpha = False
        self.decrease_ratio = decrease_ratio
        self.per_epoch = per_epoch 
        self.iter = 1
        
        self.wbce_dice_loss = BCE_DiceLoss(smooth=smooth, pos_weight=pos_weight, loss_weight=loss_weight).to('cuda') 
        self.bd_loss = BoundaryLoss().to('cuda')

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.clamp(min=self.smooth)
        # target = target.contiguous().view(target.shape[0], -1)
        
        wbce_dice_loss = self.wbce_dice_loss(predict, target)
        bd_loss = self.bd_loss(predict, target)

        loss = self.alpha*wbce_dice_loss + (1 - self.alpha) * bd_loss
        
        if self.iter % self.per_epoch == 0 and self.alpha > self.alpha_min and not self.fix_alpha:
            self.alpha -= 0.01 * self.decrease_ratio
            
        self.iter += 1

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class GeneralizedBoundaryComboLoss(nn.Module):  
    def __init__(self, per_epoch, smooth=10 ** -8, reduction='none', pos_weight=[1, 1], loss_weight=[1, 1], alpha_min=0.01, decrease_ratio=1):
        super(GeneralizedBoundaryComboLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.alpha = 1.0
        self.alpha_min = alpha_min
        self.fix_alpha = False
        self.decrease_ratio = decrease_ratio
        self.per_epoch = per_epoch 
        self.iter = 1
        
        self.wbce_dice_loss = BCE_DiceLoss(smooth=smooth, pos_weight=pos_weight, loss_weight=loss_weight, gdice=True).to('cuda') 
        self.bd_loss = BoundaryLoss().to('cuda')

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.clamp(min=self.smooth)
        # target = target.contiguous().view(target.shape[0], -1)
        
        wbce_dice_loss = self.wbce_dice_loss(predict, target)
        bd_loss = self.bd_loss(predict, target)

        loss = self.alpha*wbce_dice_loss + (1 - self.alpha) * bd_loss
        
        if self.iter % self.per_epoch == 0 and self.alpha > self.alpha_min and not self.fix_alpha:
            self.alpha -= 0.01 * self.decrease_ratio
            
        self.iter += 1

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class Boundary_GDiceLoss(nn.Module):  
    def __init__(self, per_epoch, resume_iter=0, smooth=10 ** -8, reduction='none', alpha_min=0.01, decrease_ratio=1):
        super(Boundary_GDiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.alpha = 1.0 - (resume_iter // per_epoch) * 0.01
        self.alpha_min = alpha_min
        self.fix_alpha = False
        self.decrease_ratio = decrease_ratio
        self.per_epoch = int(per_epoch) 
        self.iter = 1
        
        self.gdice_loss = GDiceLoss().to('cuda') 
        self.bd_loss = BoundaryLoss().to('cuda')

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.clamp(min=self.smooth)
        # target = target.contiguous().view(target.shape[0], -1)
        
        gdice_loss = self.gdice_loss(predict, target)
        bd_loss = self.bd_loss(predict, target)

        loss = self.alpha*gdice_loss + (1 - self.alpha) * bd_loss
        
        if self.iter % self.per_epoch == 0 and self.alpha > self.alpha_min and not self.fix_alpha:
            self.alpha -= 0.01 * self.decrease_ratio
            
        self.iter += 1

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
        

class WeightedBCELoss(nn.Module):
    def __init__(self, smooth=10 ** -8, reduction='mean', pos_weight=[1, 1]):
        super(WeightedBCELoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], f"predict & target batch size don't match. predict.shape={predict.shape}"
        predict = predict.clamp(min=self.smooth)
        # target = target.contiguous().view(target.shape[0], -1)
        
        loss =  - (self.pos_weight[0]*target*torch.log(predict+self.smooth) + self.pos_weight[1]*(1-target)*torch.log(1-predict+self.smooth))/sum(self.pos_weight)

        if self.reduction == 'mean':
            return loss.mean(dim=(1,2,3))
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

# https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/dice_loss.py
class GDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Generalized Dice;
        Copy from: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        shp_x = net_output.shape # (batch size,class_num,x,y,z)
        shp_y = gt.shape # (batch size,1,x,y,z)
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)

        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)
    
        # copy from https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        w: torch.Tensor = 1 / (torch.einsum("bcxy->bc", y_onehot).type(torch.float32) + 1e-10)**2
        intersection: torch.Tensor = w * torch.einsum("bcxy, bcxy->bc", net_output, y_onehot)
        union: torch.Tensor = w * (torch.einsum("bcxy->bc", net_output) + torch.einsum("bcxy->bc", y_onehot))
        gdc: torch.Tensor = 1 - 2 * (torch.einsum("bc->b", intersection) + self.smooth) / (torch.einsum("bc->b", union) + self.smooth)
        # gdc = divided.mean()

        return gdc
    

# https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py
class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction

    paper:
        Milletari, F., Navab, N., Ahmadi, S.A.: V-Net: Fully Convolutional Neural Networks for 
        Volumetric Medical Image Segmentation. In: 2016 Fourth International Conference on 3D 
        Vision (3DV). pp. 565–571. IEEE (oct 2016)
    """
    def __init__(self, smooth=1e-6, p=2, reduction='none'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num =  2 * torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 -num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]

class BCE_DiceLoss(nn.Module):

    def __init__(self, smooth=1, p=2, reduction='none', pos_weight=[1, 1], loss_weight = [1, 1], gdice=False):
        super(BCE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        self.bce_loss = WeightedBCELoss(pos_weight=pos_weight).to('cuda')
        if gdice:
            self.dice_loss =  GDiceLoss().to('cuda')
        else:
            self.dice_loss = BinaryDiceLoss().to('cuda')
        self.loss_weight = loss_weight

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"

        loss = (self.loss_weight[0] * self.bce_loss(predict, target) + self.loss_weight[1] * self.dice_loss(predict, target)) / sum(self.loss_weight)
        # print("loss", loss.mean())

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

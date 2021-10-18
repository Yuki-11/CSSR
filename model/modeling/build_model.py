import os
from copy import copy

import torch
import torch.nn as nn

# from .resnet import get_pose_net
from .dbpn import *
from .unet import *
from model.modeling.pspnet_pytorch.pspnet import PSPNet
from model.utils.loss_functions import BinaryDiceLoss, WeightedBCELoss, BoundaryComboLoss, Boundary_GDiceLoss, GeneralizedBoundaryComboLoss
from model.utils.misc import fix_model_state_dict, chop_forward
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

_MODEL = {
    2: Net_2,
    4: Net_4,
    6: Net_6,
    7: Net_7,
    8: Net_8,
    10: Net_10,
}


class ModelWithLoss(nn.Module):
    def __init__(self, cfg, num_train_ds, resume_iter):
        super(ModelWithLoss, self).__init__()

        self.scale_factor = cfg.MODEL.SCALE_FACTOR
        self.sr_model = set_sr_model(cfg)
        self.sr_loss_fn = set_sr_loss(cfg)
        
        # size = cfg.INPUT.IMAGE_SIZE
        self.segmentation_model = set_ss_model(cfg)
        self.ss_loss_fn = set_ss_loss(num_train_ds, resume_iter, cfg)
        
        self.norm_sr_output = cfg.SOLVER.NORM_SR_OUTPUT
        self.mean = cfg.INPUT.MEAN
        self.std = cfg.INPUT.STD
        self.seg_model_name = cfg.MODEL.DETECTOR_TYPE

    def forward(self, x, sr_targets=None, segment_targets=None):
        x = x.to('cuda')
        sr_targets = sr_targets.to('cuda')
        segment_targets = segment_targets.to('cuda').to(torch.float)

        # Super Resolution task
        if self.sr_model is None:
            sr_preds = sr_targets.clone()
            sr_loss = None
        elif self.sr_model == 'bicubic':
            upsampling_size =  [ i * self.scale_factor for i in x.size()[2:]]
            transform = transforms.Resize(upsampling_size, InterpolationMode.BICUBIC)
            sr_preds = transform(x)
            sr_loss = None
        else:
            sr_preds = self.sr_model(x)
            sr_loss = self.sr_loss_fn(sr_preds, sr_targets).mean(dim=(1,2,3))

        sr_preds_norm = normalize(sr_preds, self.mean, self.std, self.norm_sr_output)

        # Semantic segmentation task
        if self.seg_model_name == 'PSPNet':
            alpha = 0.4 
            segment_preds, aux_segment_preds = self.segmentation_model(sr_preds_norm)
            segment_loss = self.ss_loss_fn(segment_preds, segment_targets)
            segment_loss += self.ss_loss_fn(aux_segment_preds, segment_targets) * alpha
        else:
            segment_preds = self.segmentation_model(sr_preds_norm) 
            segment_loss = self.ss_loss_fn(segment_preds, segment_targets)

        return segment_loss, sr_loss, segment_preds, sr_preds


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()

        print('Building model...')
        self.scale_factor = cfg.MODEL.SCALE_FACTOR
        self.sr_model = set_sr_model(cfg)
   
        self.segmentation_model = set_ss_model(cfg)

        self.mean = cfg.INPUT.MEAN
        self.std = cfg.INPUT.STD
        self.norm_sr_output = cfg.SOLVER.NORM_SR_OUTPUT
        self.seg_model_name = cfg.MODEL.DETECTOR_TYPE

    def forward(self, x):
        x = x.to('cuda')

        # Super Resolution task
        if self.sr_model is None:
            sr_preds = x
        elif self.sr_model == 'bicubic':
            upsampling_size =  [ i * self.scale_factor for i in x.size()[2:]]
            transform = transforms.Resize(upsampling_size, InterpolationMode.BICUBIC)
            sr_preds = transform(x)
        else:
            sr_preds = self.sr_model(x)
            
        sr_preds[sr_preds>1] = 1 # clipping
        sr_preds[sr_preds<0] = 0 # clipping       
        sr_preds_norm = normalize(sr_preds, self.mean, self.std, self.norm_sr_output)

        # Semantic segmentation task
        if self.seg_model_name == 'PSPNet':
            segment_preds, _ = self.segmentation_model(sr_preds_norm)
        else:
            segment_preds = self.segmentation_model(sr_preds_norm) 

        return sr_preds, segment_preds


class InvModelWithLoss(nn.Module):
    def __init__(self, cfg, num_train_ds, resume_iter, sr_transforms):
        super(InvModelWithLoss, self).__init__()

        # size = cfg.INPUT.IMAGE_SIZE
        self.segmentation_model = set_ss_model(cfg)
        self.ss_loss_fn = set_ss_loss(num_train_ds, resume_iter, cfg)

        self.scale_factor = cfg.MODEL.SCALE_FACTOR
        self.sr_model = set_sr_model(cfg)
        self.sr_loss_fn = set_sr_loss(cfg)
               
        self.norm_sr_output = cfg.SOLVER.NORM_SR_OUTPUT
        self.mean = cfg.INPUT.MEAN
        self.std = cfg.INPUT.STD
        self.seg_model_name = cfg.MODEL.DETECTOR_TYPE
        
        self.sr_transforms = sr_transforms

    def forward(self, x, sr_targets=None, segment_targets=None):
        x = x.to('cuda')
        hr_segment_targets = segment_targets.to('cuda').to(torch.float)
        lr_segment_targets = self.sr_transforms(hr_segment_targets)

        # Semantic segmentation task
        if self.seg_model_name == 'PSPNet':
            alpha = 0.4 
            lr_segment_preds, lr_aux_segment_preds = self.segmentation_model(x)
            segment_loss = self.ss_loss_fn(lr_segment_preds, lr_segment_targets)
            segment_loss += self.ss_loss_fn(lr_aux_segment_preds, lr_segment_targets) * alpha
        else:
            lr_segment_preds = self.segmentation_model(x)
            segment_loss = self.ss_loss_fn(lr_segment_preds, lr_segment_targets)

        # Super Resolution task
        if self.sr_model is None:
            sr_segment_preds = lr_segment_preds.clone()
            sr_loss = None
        elif self.sr_model == 'bicubic':
            upsampling_size =  [ i * self.scale_factor for i in lr_segment_preds.size()[2:]]
            sr_segment_preds = F.interpolate(lr_segment_preds, upsampling_size, mode='bicubic')
            sr_loss = None
        else:
            sr_segment_preds = self.sr_model(lr_segment_preds)
            sr_loss = self.sr_loss_fn(sr_segment_preds, hr_segment_targets).mean(dim=(1,2,3))

        return segment_loss, sr_loss, sr_segment_preds, None


class InvModel(nn.Module):
    def __init__(self, cfg):
        super(InvModel, self).__init__()

        print('Building model...')
        self.segmentation_model = set_ss_model(cfg)

        self.scale_factor = cfg.MODEL.SCALE_FACTOR
        self.sr_model = set_sr_model(cfg)

        self.mean = cfg.INPUT.MEAN
        self.std = cfg.INPUT.STD
        self.norm_sr_output = cfg.SOLVER.NORM_SR_OUTPUT
        self.seg_model_name = cfg.MODEL.DETECTOR_TYPE

    def forward(self, x):
        x = x.to('cuda')

        # Semantic segmentation task
        if self.seg_model_name == 'PSPNet':
            lr_segment_preds, _ = self.segmentation_model(x)
        else:
            lr_segment_preds = self.segmentation_model(x)

        # Super Resolution task
        if self.sr_model is None:
            sr_segment_preds = lr_segment_preds.clone()
        elif self.sr_model == 'bicubic':
            upsampling_size =  [ i * self.scale_factor for i in lr_segment_preds.size()[2:]]
            sr_segment_preds = F.interpolate(lr_segment_preds, upsampling_size, mode='bicubic')
        else:
            sr_segment_preds = self.sr_model(lr_segment_preds)
            
        sr_segment_preds[sr_segment_preds>1] = 1 # clipping
        sr_segment_preds[sr_segment_preds<0] = 0 # clipping       

        return None, sr_segment_preds

def set_sr_model(cfg):
    if cfg.MODEL.SR_SEG_INV:
        num_channels = 1
    else:
        num_channels = 3

    if cfg.MODEL.SCALE_FACTOR == 1: 
        sr_model = None
    elif cfg.MODEL.SR == 'bicubic':
        sr_model = 'bicubic'
    elif cfg.MODEL.SR == 'DBPN':
        sr_model = _MODEL[cfg.MODEL.NUM_STAGES](cfg.MODEL.SCALE_FACTOR, num_channels)
        pretrained_model_path = os.path.join('weights', 'sr_pretrain_x{}_stage{}.pth'.format(cfg.MODEL.SCALE_FACTOR, cfg.MODEL.NUM_STAGES))
        sr_model.load_state_dict(fix_model_state_dict(torch.load(pretrained_model_path)), strict=False) # data parallelの関係で読み込めないとき
        print('SR pretrained model was loaded from {}'.format(pretrained_model_path))

    return sr_model

def set_sr_loss(cfg):
    if cfg.SOLVER.SR_LOSS_FUNC == "L1":
        sr_loss_fn = nn.L1Loss(reduction="none")

    else:
        raise NotImplementedError(cfg.SOLVER.SR_LOSS_FUNC)

    return sr_loss_fn

def set_ss_model(cfg):
    num_classes = cfg.MODEL.NUM_CLASSES

    if cfg.MODEL.DETECTOR_TYPE == 'u-net16':
        segmentation_model = UNet16(num_classes=num_classes, pretrained=True, up_sampling_method=cfg.MODEL.UP_SAMPLE_METHOD)        
    elif cfg.MODEL.DETECTOR_TYPE == 'PSPNet':
        # Main loss and auxiliary loss conform to segmentation loss function
        segmentation_model = PSPNet(n_classes=num_classes, pretrained=True)
    else:
        raise NotImplementedError(cfg.MODEL.DETECTOR_TYPE)

    return segmentation_model

def set_ss_loss(num_train_ds, resume_iter, cfg):
    if cfg.SOLVER.SEG_LOSS_FUNC == "BCE":
        ss_loss_fn = nn.BCELoss().to('cuda')
    elif cfg.SOLVER.SEG_LOSS_FUNC == "WeightedBCE":
        pos_weight = cfg.SOLVER.BCELOSS_WEIGHT
        ss_loss_fn = WeightedBCELoss(pos_weight=pos_weight).to('cuda')
    elif cfg.SOLVER.SEG_LOSS_FUNC == "Dice":
        ss_loss_fn = BinaryDiceLoss().to('cuda')
    elif cfg.SOLVER.SEG_LOSS_FUNC == "BoundaryCombo": # old: Boundary
        pos_weight = cfg.SOLVER.BCELOSS_WEIGHT
        loss_weight = cfg.SOLVER.WB_AND_D_WEIGHT
        per_epoch = num_train_ds // cfg.SOLVER.BATCH_SIZE + 1
        ss_loss_fn = BoundaryComboLoss(per_epoch, pos_weight=pos_weight, loss_weight=loss_weight, decrease_ratio=cfg.SOLVER.BOUNDARY_DEC_RATIO).to('cuda')
    elif cfg.SOLVER.SEG_LOSS_FUNC == "Boundary_GDice":
        per_epoch = num_train_ds // cfg.SOLVER.BATCH_SIZE + 1
        ss_loss_fn = Boundary_GDiceLoss(per_epoch, resume_iter=resume_iter).to('cuda')
    elif cfg.SOLVER.SEG_LOSS_FUNC == "GeneralizedBoundaryCombo":
        pos_weight = cfg.SOLVER.BCELOSS_WEIGHT
        loss_weight = cfg.SOLVER.WB_AND_D_WEIGHT
        per_epoch = num_train_ds // cfg.SOLVER.BATCH_SIZE + 1
        ss_loss_fn = GeneralizedBoundaryComboLoss(per_epoch, pos_weight=pos_weight, loss_weight=loss_weight).to('cuda')
    else:
        raise NotImplementedError(cfg.SOLVER.SEG_LOSS_FUNC)

    return ss_loss_fn

def normalize(sr_images, mean, std, norm_method):
    if norm_method == "all":
        _mean = torch.empty(sr_images.shape).to('cuda')
        _std = torch.empty(sr_images.shape).to('cuda')
        num_channels = 3
        
        for i in range(num_channels):
            _mean[:, i, :, :] = self.mean[i]
            _std[:, i, :, :] = self.std[i]
        sr_images_norm = (sr_images - _mean) / _std
    
    elif norm_method == "instance":
        norm = nn.InstanceNorm2d(3)
        sr_images_norm  = norm(sr_images)
        
    return sr_images_norm








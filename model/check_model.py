import os
from copy import copy

import torch
import torch.nn as nn
from torchsummary import summary

# from .resnet import get_pose_net
from modeling.dbpn import *


print("test")
_MODEL = {
    2: Net_2,
    4: Net_4,
    6: Net_6,
    7: Net_7,
    8: Net_8,
    10: Net_10,
}

sf = 4
stage = 4

sr_model = _MODEL[stage](sf)
pretrained_model_path = os.path.join('weights', 'sr_pretrain_x{}_stage{}.pth'.format(sf, stage))
pretrained_model_path = os.path.join('weights', 'DBPN_x4.pth')
# sr_model.load_state_dict(fix_model_state_dict(torch.load(pretrained_model_path))) # data parallelの関係で読み込めないとき
#sr_model.load_state_dict(torch.load(pretrained_model_path)) # 通常
model = torch.load(pretrained_model_path)
summary(model, (1, 512, 512))
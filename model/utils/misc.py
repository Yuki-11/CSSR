import torch
from collections import OrderedDict
import cv2
import os
import requests

from model.data.transforms.transforms import *

def _sigmoid(x):
	y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
	return y

def send_line_notify(notification_message):
    line_notify_token = ''
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'message: {notification_message}'}
    requests.post(line_notify_api, headers = headers, data = data)


def str2bool(s):
	return s.lower() in ('true', '1')


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict


def overlay_heatmap(image, heatmap):
    jet_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed_image = cv2.addWeighted(jet_heatmap, 0.4, image, 0.6, 0)
    
    return overlayed_image


def init_layer(layer):
    for m in layer:
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('ConvTranspose2d') != -1:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

class SaveImage:
    def __init__(self, cfg):
        self.transforms = Compose([
            ToNumpy(),
            # Denormalize(cfg),
        ])

    def __call__(self, image, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image, _, _ = self.transforms(image)
        image = image*255
        image = image.clip(0, 255)
        # print(image.min(), image.max())

        cv2.imwrite(save_path, image)

def chop_forward(x, model, scale, shave=8, min_size=160000, nGPUs=1):
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, nGPUs):
            with torch.no_grad():
                input_batch = torch.cat(inputlist[i:(i + nGPUs)], dim=0)
                output_batch = model(input_batch)
            outputlist.extend(output_batch.chunk(nGPUs, dim=0))
    else:
        outputlist = [
            chop_forward(patch, model, scale, shave, min_size, nGPUs) \
            for patch in inputlist]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    with torch.no_grad():
        output = x.data.new(b, c, h, w)

    output[:, :, 0:h_half, 0:w_half] \
        = outputlist[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output
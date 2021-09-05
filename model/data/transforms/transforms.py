import numpy as np
import cv2
import random
import math
import time
from itertools import product
from math import sqrt
import torchvision.transforms as torch_transforms
from torchvision.transforms import InterpolationMode
import torch
import torch.nn as nn


from inspect import currentframe

def chkprint(*args):
    names = {id(v):k for k,v in currentframe().f_back.f_locals.items()}
    print(', '.join(names.get(id(arg),'???')+' = '+repr(arg) for arg in args))

def image_clipping(image):
    # image[image > 255] = 255
    # image[image < 0] = 0
    image = np.clip(image, 0, 255)

    return image

def multiple_list(list1, list2):
    assert len(list1) == len(list2)
    output_list = [None for _ in range(len(list1))]
    for i in range(len(output_list)):
        output_list[i] = list1[i] * list2[i]

    return output_list

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  
  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[:2]
    
  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
  return heatmap


def extract_valid_boxes(boxes, scale_range):
    box_sizes = boxes[:, 2:] - boxes[:, :2] # [[w, h]]
    box_sizes = box_sizes.max(1)

    m1 = box_sizes > scale_range[0]
    m2 = box_sizes < scale_range[1]
    
    # print(box_sizes, m1*m2)

    return m1 * m2

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask=None):
        i = 1

        for t in self.transforms:
            # chkprint(t)
            img, mask = t(img, mask)
            i += 1

        return img, mask


class RandomResize(object):
    def __init__(self):
        self.interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        self.resize_range = (0.6, 1.4)

    def __call__(self, image, boxes=None, labels=None):
        interp_method = self.interp_methods[random.randint(0, 4)]
        fx, fy = random.uniform(self.resize_range[0], self.resize_range[1]), random.uniform(self.resize_range[0], self.resize_range[1])
        height, width, _ = image.shape
        size = (int(width*fx), int(height*fy))
        image = cv2.resize(image, size, interpolation=interp_method)

        if boxes is not None:
            boxes[:, 1::2] = boxes[:, 1::2] * fy
            boxes[:, 0::2] = boxes[:, 0::2] * fx

        return image, boxes, labels


class Resize(object):
    def __init__(self, size=(2048, 1024)):
        self.size = size
        self.interp_method = cv2.INTER_CUBIC

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, self.size, interpolation=self.interp_method)

        return image, boxes, labels


class ToTensor(object):
    def __call__(self, image, mask=None):
        if mask is None:
            return torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1), mask
        else:
            return torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1), torch.from_numpy(mask.astype(np.float32)).permute(2, 0, 1)

class ToNumpy(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.numpy().transpose((1, 2, 0)), boxes, labels


class ConvertFromInts(object):
    def __call__(self, image, mask):

        if mask is not None:
            return image.astype(np.float32), mask.astype(np.float32)
        else:
            return image.astype(np.float32), None

class ConvertToInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.uint8), boxes, labels


class SubtractMeans(object):
    def __init__(self, mean=0):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class Normalize(object):
    def __init__(self, cfg):
        self.mean = cfg.INPUT.MEAN
        self.std = cfg.INPUT.STD

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)/255
        image -= self.mean
        image /= self.std

        return image.astype(np.float32), boxes, labels


class Denormalize(object):
    def __init__(self, cfg):
        self.mean = cfg.INPUT.MEAN
        self.std = cfg.INPUT.STD

    def __call__(self, image, boxes=None, labels=None):
        image *= self.std
        image += self.mean
        image = image*255

        return image, boxes, labels

class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, mask):
        if np.random.randint(2):
            image[:, :, 1] *= np.random.uniform(self.lower, self.upper)

        return image, mask


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, mask):
        if np.random.randint(2):
            image[:, :, 0] += np.random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, mask


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image):
        if np.random.randint(2):
            swap = self.perms[np.random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image


class ConvertColor(object):
    def __init__(self, current, transform):
        self.transform = transform
        self.current = current

    def __call__(self, image, mask):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'BGR' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.transform == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, mask


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, mask):
        if np.random.randint(2):
            alpha = np.random.uniform(self.lower, self.upper)
            image *= alpha
        return image_clipping(image), mask


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image):
        if np.random.randint(2):
            delta = np.random.uniform(-self.delta, self.delta)
            image += delta

        return image_clipping(image)


class RandomMirror(object):
    def __call__(self, image, mask):
        if np.random.randint(2):
            image = image[:, ::-1]
            mask = mask[:, ::-1]

        return image, mask


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pmd = [
            RandomContrast(),  # RGB
            ConvertColor(current="RGB", transform='HSV'),  # HSV
            RandomSaturation(),  # HSV
            RandomHue(),  # HSV
            ConvertColor(current='HSV', transform='RGB'),  # RGB
            RandomContrast()  # RGB
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, mask):

        im = image.copy()
        im = self.rand_brightness(im)

        if np.random.randint(2):
            distort = Compose(self.pmd[:-1])
        else:
            distort = Compose(self.pmd[1:])
        im, mask = distort(im, mask)

        return self.rand_light_noise(im), mask


class Clamp(object):
    def __init__(self, min=0., max=255.):
        self.min = min
        self.max = max

    def __call__(self, image):
        return torch.clamp(image, min=self.min, max=self.max)


class CenterCrop(object):
    def __init__(self, crop_size=64):
        self.crop_size = crop_size

    def __call__(self, hr_img, lr_img=None):
        height, width, _ = hr_img.shape
        crop_area = [(height - self.crop_size)/2, (height + self.crop_size)/2, (width - self.crop_size)/2, (width + self.crop_size)/2]
        crop_area = [int(i) for i in crop_area]
        
        return hr_img[crop_area[0]:crop_area[1], crop_area[2]:crop_area[3]], lr_img


class ConstantPadding(object):
    def __init__(self, image_size=640):
        self.image_size = image_size

    def __call__(self, image, boxes=None, labels=None):
        height, width, channel = image.shape
        padded_image = np.zeros((self.image_size, self.image_size, channel))

        padded_image[:height, :width, :] = image
    
        return padded_image, boxes, labels

class MakeHeatmap(object):
    def __init__(self, cfg, down_ratio=4 ,eval=False):
        # self.input_size = cfg.INPUT.IMAGE_SIZE
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.max_objects = cfg.INPUT.MAX_OBJECTS

        self.down_ratio = down_ratio
        self.eval = eval

    def __call__(self, image, boxes, labels):
        output_height, output_width, _ = image.shape
        # _, output_height, output_width = image.shape
        num_objects = min(len(boxes), self.max_objects)
        output_height = int(output_height / self.down_ratio)
        output_width = int(output_width / self.down_ratio)
        for i in range(len(boxes)):
            for j in range(len(boxes[i])):
                boxes[i][j] /= self.down_ratio


        heatmap = np.zeros((self.num_classes, output_height, output_width), dtype=np.float32)
        wh = np.zeros((self.max_objects, 2), dtype=np.float32)
        reg = np.zeros((self.max_objects, 2), dtype=np.float32)
        ind = np.zeros((self.max_objects), dtype=np.int64)
        reg_mask = np.zeros((self.max_objects), dtype=np.uint8)

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, output_width - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, output_height - 1)

        for i in range(num_objects):
            box = boxes[i]
            label = labels[i]
            class_id = label - 1 # remove __background__ label

            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            center = (box[:2] + box[2:]) / 2
            center_int = center.astype(np.uint8)
            radius = max(0, int(gaussian_radius((math.ceil(box_height), math.ceil(box_width)))))

            heatmap[class_id] = draw_umich_gaussian(heatmap[class_id], center.astype(np.uint8), radius)
            wh[i] = 1. * box_width, 1. * box_height
            ind[i] = int(center_int[1] * output_width + center_int[0])
            reg[i] = center - center_int
            reg_mask[i] = 1

        

        ret = {'hm': heatmap, 'reg': reg, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}

        return image, ret


class FactorResize(object):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, image):
        height, width = image.shape[-2:]
        transform = torch_transforms.Resize((int(height/self.factor), int(width/self.factor)))
        image = transform(image)

        return image


# class RandomCrop(object):
#     def __init__(self, crop_size):
#         self.crop_size = crop_size

#     def __call__(self, image, boxes=None, labels=None):
#         height, width, _ = image.shape

#         image_center = np.zeros(2)
#         image_center[0] = np.random.randint(width)
#         image_center[1] = np.random.randint(height)

#         left_pad = int(np.clip(self.crop_size/2 - image_center[0], 0, None))
#         right_pad = int(np.clip(self.crop_size/2 - (width - image_center[0]), 0, None))
#         upper_pad = int(np.clip(self.crop_size/2 - image_center[1], 0, None))
#         under_pad = int(np.clip(self.crop_size/2 - (height - image_center[1]), 0, None))

#         image_center += np.array([left_pad, upper_pad]).astype(np.uint16)

#         crop_area = np.array([image_center[0] - self.crop_size/2, image_center[1] - self.crop_size/2,
#                               image_center[0] + self.crop_size/2, image_center[1] + self.crop_size/2]).astype(np.uint16)

#         image = np.pad(image, [(upper_pad, under_pad), (left_pad, right_pad), (0, 0)], 'constant')       
#         image = image[crop_area[1]:crop_area[3], crop_area[0]:crop_area[2], :]
#         boxes[:, :2] = boxes[:, :2] + np.array([left_pad, upper_pad])
#         boxes[:, 2:] = boxes[:, 2:] + np.array([left_pad, upper_pad])
#         boxes[:, :2] = boxes[:, :2] - crop_area[:2]
#         boxes[:, 2:] = boxes[:, 2:] - crop_area[:2]

#         boxes = np.clip(boxes, 0, self.crop_size)
#         boxes = np.clip(boxes, 0, self.crop_size)
#         box_sizes = boxes[:, 2:] - boxes[:, :2]

#         mask = box_sizes > 0
#         mask = mask.all(1)

#         boxes = boxes[mask]
#         labels = labels[mask]

#         return image, boxes, labels


class PriorBox(nn.Module):
    def __init__(self, cfg, feature_maps):
        super(PriorBox, self).__init__()
        self.image_size = cfg.INPUT.IMAGE_SIZE
        prior_config = cfg.MODEL.PRIORS
        # self.feature_maps = prior_config.FEATURE_MAPS
        self.feature_maps = feature_maps
        self.min_sizes = prior_config.MIN_SIZES
        self.max_sizes = prior_config.MAX_SIZES
        self.strides = prior_config.STRIDES
        self.aspect_ratios = prior_config.ASPECT_RATIOS
        self.clip = prior_config.CLIP

    def forward(self):
        """Generate SSD Prior Boxes.
            It returns the center, height and width of the priors. The values are relative to the image size
            Returns:
                priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
                    are relative to the image size.
        """
        priors = []
        for k, f in enumerate(self.feature_maps):
            scale = self.image_size / self.strides[k]
            # for i, j in product(range(f), repeat=2):
            for i in range(f[0]):
                for j in range(f[1]):
                    # print(i, j)
                    # unit center x,y
                    cx = (j + 0.5) / scale
                    cy = (i + 0.5) / scale

                    # small sized square box
                    size = self.min_sizes[k]
                    h = w = size / self.image_size
                    priors.append([cx, cy, w, h])

                    # big sized square box
                    size = sqrt(self.min_sizes[k] * self.max_sizes[k])
                    h = w = size / self.image_size
                    priors.append([cx, cy, w, h])

                    # change h/w ratio of the small sized box
                    size = self.min_sizes[k]
                    h = w = size / self.image_size
                    for ratio in self.aspect_ratios[k]:
                        ratio = sqrt(ratio)
                        priors.append([cx, cy, w * ratio, h / ratio])
                        priors.append([cx, cy, w / ratio, h * ratio])

        priors = torch.Tensor(priors)
        if self.clip:
            priors.clamp_(max=1, min=0)
        return priors


class RandomResizedCrop(object):
    def __init__(self, cfg=None, scale=(0.5,1.0)):
        self.rrc = torch_transforms.RandomResizedCrop(cfg.INPUT.IMAGE_SIZE)
        self.size = cfg.INPUT.IMAGE_SIZE

    def __call__(self, image, mask=None):
        # seed = random.randint(0, 2**32)
        params = self.rrc.get_params(image, scale=(0.6, 1.0), ratio=(1.0, 1.0))
        # img_crop = transforms.functional.crop(image, *params)
        image = torch_transforms.functional.resized_crop(image, size=self.size, *params)
         
        if mask is None:
            return image

        mask = torch_transforms.functional.resized_crop(mask, size=self.size, *params)

        return image, mask

class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.rvf = torch_transforms.RandomVerticalFlip(p=1.0)
        self.p = p

    def __call__(self, image, mask):
        if self.p <= np.random.rand():
            image = self.rvf(image)
            mask = self.rvf(mask)

        return image, mask

class RandomGrayscale(object):
    def __init__(self, p=0.5):
        self.rgs = torch_transforms.RandomGrayscale(p=1.0)
        self.p = p

    def __call__(self, image, mask):
        if self.p <= np.random.rand():
            image = self.rgs(image)

        return image, mask
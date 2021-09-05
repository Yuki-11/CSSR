from .transforms import *

class TrainTransforms:
    def __init__(self, cfg):
        self.augment = Compose([
            ConvertFromInts(),
            RandomMirror(),
            PhotometricDistort(),
            # Normalize(cfg),
            ToTensor(),
            # RandomGrayscale(p=0.25),
            # RandomVerticalFlip(p=0.25),
            RandomResizedCrop(cfg=cfg),
        ])

    def __call__(self, image, mask):
        image, mask = self.augment(image, mask)
        return image/255, mask/255

class TestTransforms:
    def __init__(self, cfg):
        self.augment = Compose([
            ConvertFromInts(),
            ToTensor(),
        ])

    def __call__(self, image, mask):
        image, mask = self.augment(image, mask)
        if mask is not None:
            return image/255, mask/255
        
        return image/255, None

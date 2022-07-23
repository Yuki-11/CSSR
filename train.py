import numpy as np
import argparse
import os
import random
import shutil
import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, BatchSampler

from model.utils.sync_batchnorm import convert_model
from model.config import cfg
from model.engine.trainer import do_train
import torchvision.transforms as transforms
from model.data.transforms.data_preprocess import TrainTransforms, TestTransforms
from model.data.transforms.transforms import FactorResize
from model.modeling.build_model import ModelWithLoss, InvModelWithLoss
from model.data.crack_dataset import CrackDataSet
from model.utils.misc import str2bool, fix_model_state_dict
from model.data import samplers
from model.utils.lr_scheduler import WarmupMultiStepLR
from torch.multiprocessing import Pool, Process, set_start_method

def train(args, cfg):
    device = torch.device(cfg.DEVICE)

    print('Loading Datasets...')
    train_transforms = TrainTransforms(cfg)
    sr_transforms = FactorResize(cfg.MODEL.SCALE_FACTOR)
    trainval_dataset = CrackDataSet(cfg, cfg.DATASET.TRAIN_IMAGE_DIR, cfg.DATASET.TRAIN_MASK_DIR, transforms=train_transforms, sr_transforms=sr_transforms)

    n_samples = len(trainval_dataset) 
    train_size = int(len(trainval_dataset) * cfg.SOLVER.TRAIN_DATASET_RATIO) 
    val_size = n_samples - train_size
    print(f"Train dataset size: {train_size}, Validation dataset size: {val_size}")
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])

    sampler = torch.utils.data.RandomSampler(train_dataset)
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=cfg.SOLVER.BATCH_SIZE, drop_last=True)
    batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, num_iterations=cfg.SOLVER.MAX_ITER)
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_sampler=batch_sampler, pin_memory=True)

    eval_sampler = SequentialSampler(val_dataset)
    eval_batch_sampler = BatchSampler(sampler=eval_sampler, batch_size=cfg.SOLVER.BATCH_SIZE, drop_last=True)
    eval_loader = DataLoader(val_dataset, num_workers=args.num_workers, batch_sampler=eval_batch_sampler, pin_memory=True)

    print('Building model...')
    if cfg.MODEL.SR_SEG_INV:
        model = InvModelWithLoss(cfg, num_train_ds=train_size, resume_iter=args.resume_iter, sr_transforms=sr_transforms).to(device)
        print(f'------------Model Architecture-------------\n\n<Network SS>\n{model.segmentation_model}\n\n<Network SR>\n{model.sr_model}')
    else:
        model = ModelWithLoss(cfg, num_train_ds=train_size, resume_iter=args.resume_iter).to(device)
        print(f'------------Model Architecture-------------\n\n<Network SR>\n{model.sr_model}\n\n<Network SS>\n{model.segmentation_model}')

    if cfg.MODEL.OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=cfg.SOLVER.LR)
    elif cfg.MODEL.OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad, model.parameters()), lr=cfg.SOLVER.LR, momentum=0.9, weight_decay=5e-4)
  
    milestones = [step for step in cfg.SOLVER.LR_STEPS]
    scheduler = WarmupMultiStepLR(cfg, optimizer=optimizer, milestones=milestones, gamma=cfg.SOLVER.GAMMA, warmup_factor=cfg.SOLVER.WARMUP_FACTOR, warmup_iters=cfg.SOLVER.WARMUP_ITERS)

    if args.resume_iter > 0:
        print('Resume from {}'.format(os.path.join(cfg.OUTPUT_DIR, 'model', 'iteration_{}.pth'.format(args.resume_iter))))
        model.load_state_dict(fix_model_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, 'model', 'iteration_{}.pth'.format(args.resume_iter)))))
        optimizer.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, 'optimizer', 'iteration_{}.pth'.format(args.resume_iter))))
    
    if cfg.SOLVER.SYNC_BATCHNORM:
        model = convert_model(model).to(device)
    
    if args.num_gpus > 1:
        device_ids = list(range(args.num_gpus))
        # device_ids.insert(0, device_ids.pop(cfg.DEVICE_NUM))
        print("device_ids:",device_ids)
        model = torch.nn.DataParallel(model, device_ids=device_ids)  # primaly gpu is last device.
    
    do_train(args, cfg, model, optimizer, scheduler, train_loader, eval_loader)

def main():
    parser = argparse.ArgumentParser(description='Crack Segmentation with Super Resolution(CSSR)')
    parser.add_argument('--config_file', type=str, default='./config/configs_train.yaml', metavar='FILE', help='path to config file')
    parser.add_argument('--output_dirname', type=str, default='', help='')
    parser.add_argument('--num_workers', type=int, default=2, help='')
    parser.add_argument('--log_step', type=int, default=50, help='')
    parser.add_argument('--save_step', type=int, default=2000)
    parser.add_argument('--eval_step', type=int, default=250)
    parser.add_argument('--num_gpus', type=int, default=6)
    parser.add_argument('--mixed_precision', type=str2bool, default=False)
    parser.add_argument('--wandb_flag', type=str2bool, default=True)
    parser.add_argument('--resume_iter', type=int, default=0)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--wandb_prj_name', type=str, default="CSSR_train")

    args = parser.parse_args()

    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    cuda = torch.cuda.is_available()
    if cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(cfg.SEED)

    if len(args.config_file) > 0:
        print('Configration file is loaded from {}'.format(args.config_file))
        cfg.merge_from_file(args.config_file)
    
    if "_ds_" in cfg.DATASET.TRAIN_IMAGE_DIR:
        cfg.INPUT.IMAGE_SIZE = int(cfg.INPUT.IMAGE_SIZE / cfg.MODEL.SCALE_FACTOR )

    cfg.freeze()

    if not args.debug and args.resume_iter == 0:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        shutil.copy2(args.config_file, os.path.join(cfg.OUTPUT_DIR, 'config.yaml'))

    train(args, cfg)

if __name__ == '__main__':
    set_start_method('spawn')
    main()

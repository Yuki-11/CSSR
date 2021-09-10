import argparse
import datetime
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, BatchSampler

from model.config import cfg
from model.modeling.build_model import Model, InvModel
from model.data.transforms.data_preprocess import TestTransforms
from model.data.crack_dataset import CrackDataSetTest
from model.engine.inference import inference_for_ss
from model.utils.misc import fix_model_state_dict, send_line_notify
from model.data.transforms.transforms import FactorResize
from torch.multiprocessing import Pool, Process, set_start_method

def test(args, cfg):
    device = torch.device(cfg.DEVICE)
    # model = Model(cfg).to(device)
    if cfg.MODEL.SR_SEG_INV:
        model = InvModel(cfg).to(device)
        print(f'------------Model Architecture-------------\n\n<Network SS>\n{model.segmentation_model}\n\n<Network SR>\n{model.sr_model}')
    else:
        model = Model(cfg).to(device)
        print(f'------------Model Architecture-------------\n\n<Network SR>\n{model.sr_model}\n\n<Network SS>\n{model.segmentation_model}')

    model.load_state_dict(fix_model_state_dict(torch.load(args.trained_model, map_location=lambda storage, loc:storage)))
    model.eval()
  
    print('Loading Datasets...')
    test_transforms = TestTransforms(cfg)
    sr_transforms = FactorResize(cfg.MODEL.SCALE_FACTOR)
    test_dataset = CrackDataSetTest(cfg, cfg.DATASET.TEST_IMAGE_DIR, cfg.DATASET.TEST_MASK_DIR, transforms=test_transforms, sr_transforms=sr_transforms)
    sampler = SequentialSampler(test_dataset)
    batch_sampler = BatchSampler(sampler=sampler, batch_size=args.batch_size, drop_last=False)
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers, batch_sampler=batch_sampler)

    if args.num_gpus > 1:
        # for k in models.keys():
        #     device_ids = list(range(args.num_gpus))
        #     print("device_ids:",device_ids)
        #     # models[k] = torch.nn.DataParallel(models[k], device_ids=device_ids)
        device_ids = list(range(args.num_gpus))
        print("device_ids:",device_ids)
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    with torch.no_grad():  
        inference_for_ss(args, cfg, model, test_loader)

def main():
    parser = argparse.ArgumentParser(description='Crack Segmentation with Super Resolution(CSSR)')
    parser.add_argument('test_dir', type=str, default=None)
    parser.add_argument('iteration', type=int, default=None)

    parser.add_argument('--output_dirname', type=str, default=None)
    parser.add_argument('--config_file', type=str, default=None, metavar='FILE')    
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_gpus', type=int, default=6)
    parser.add_argument('--test_aiu', type=bool, default=True)
    parser.add_argument('--trained_model', type=str, default=None)
    parser.add_argument('--wandb_flag', type=bool, default=True)
    parser.add_argument('--wandb_prj_name', type=str, default="CSSR_kyoken_test_blur")
    
    args = parser.parse_args()

    check_args = [('config_file', f'{args.test_dir}config.yaml'), 
     ('output_dirname', f'{args.test_dir}eval_AIU/iter_{args.iteration}'),
     ('trained_model', f'{args.test_dir}model/iteration_{args.iteration}.pth'), 
    ]

    for check_arg in check_args:
        arg_name = f'args.{check_arg[0]}'
        if exec(arg_name) == None:
            exec(f'{arg_name} = "{check_arg[1]}"')

    cuda = torch.cuda.is_available()
    if cuda:
        torch.backends.cudnn.benchmark = True

    if len(args.config_file) > 0:
        print('Configration file is loaded from {}'.format(args.config_file))
        cfg.merge_from_file(args.config_file)
    
    cfg.OUTPUT_DIR = args.output_dirname
    
    cfg.freeze()

    print('Running with config:\n{}'.format(cfg))

    test(args, cfg)


if __name__ == '__main__':
    set_start_method('spawn')
    main()

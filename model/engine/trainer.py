import time
import os
import datetime
from copy import copy
import numpy as np
from tqdm import tqdm

import torch

from model.utils.misc import SaveImage
from model.utils.estimate_metrics import PSNR, SSIM, IoU 
from model.engine.inference import save_img
import torchvision.transforms as transforms

import wandb

def do_train(args, cfg, model, optimizer, scheduler, train_loader, eval_loader):

    max_iter = len(train_loader) + args.resume_iter
    trained_time = 0
    tic = time.time()
    end = time.time()

    psnr = PSNR()
    ssim = SSIM()
    iou = IoU()

    logging_sr_loss = 0
    logging_segment_loss = 0

    free_1st_stage_model_params = True

    ## --- wandb setting https://docs.wandb.ai/integrations/pytorch --- ##
    if args.wandb_flag:
        wandb.init(config=cfg, project=args.wandb_prj_name)
        wandb.config.update(args)
        # Magic
        wandb.watch(model, log='all')        
        wandb.run.name = cfg.OUTPUT_DIR.replace("output/", "")

    print('Training Starts')
    fix_2nd_stage_model_params(cfg, model)
    for iteration, (imgs, sr_targets, segment_targets) in enumerate(train_loader, args.resume_iter+1):

        free_1st_stage_model_params = fix_1st_stage_model_params(cfg, model, free_1st_stage_model_params, iteration)

        model.train()
        optimizer.zero_grad()
        
        segment_loss, sr_loss, _, _ = model(imgs, sr_targets=sr_targets, segment_targets=segment_targets)
        loss, logging_segment_loss, logging_sr_loss = calc_loss(segment_loss, logging_segment_loss, sr_loss, logging_sr_loss, iteration, cfg)

        loss.backward()
        optimizer.step()   
        scheduler.step()
        
        trained_time += time.time() - end
        end = time.time()

        # ============================== debug ===================================
        # 
        # fname = [f'image{iteration}_{i}.png' for i in range(6)]
        # save_img(cfg.OUTPUT_DIR, imgs, fname) # debug

        # fname = [f'hr{iteration}_{i}.png' for i in range(6)]
        # save_img(cfg.OUTPUT_DIR, sr_targets, fname) # debug

        # fname = [f'seg{iteration}_{i}.png' for i in range(6)]
        # for batch_num in range(segment_targets.size()[0]):
        #     # print(segment_targets.size())
        #     ss_pred = transforms.ToPILImage(mode='L')(segment_targets[batch_num])
        #     os.makedirs(os.path.dirname(cfg.OUTPUT_DIR+f"/masks/"), exist_ok=True)
        #     fpath = os.path.join(cfg.OUTPUT_DIR+f"/images/", f"{fname[batch_num]}")
        #     ss_pred.save(fpath)

        # =======================================================================

        del loss, segment_loss, sr_loss, imgs, sr_targets, segment_targets

        if iteration % args.log_step == 0:
            logging_segment_loss /= args.log_step
            logging_sr_loss /= args.log_step
            logging_tot_loss = logging_sr_loss + cfg.SOLVER.TASK_LOSS_WEIGHT * logging_segment_loss

            eta_seconds = int((trained_time / iteration) * (max_iter - iteration))
            print('===> Iter: {:07d}, LR: {:.5f}, Cost: {:.2f}s, Eta: {}, Segment_Loss({}): {:.6f}, SR_Loss({}): {:.6f}'.format(iteration, optimizer.param_groups[0]['lr'], time.time() - tic, str(datetime.timedelta(seconds=eta_seconds)), cfg.SOLVER.SEG_LOSS_FUNC, logging_segment_loss, cfg.SOLVER.SR_LOSS_FUNC, logging_sr_loss))

            if args.wandb_flag:
                # wandb
                wandb.log({"loss": logging_tot_loss, 
                        f"segment_loss({cfg.SOLVER.SEG_LOSS_FUNC})":logging_segment_loss, 
                        f"sr_loss({cfg.SOLVER.SR_LOSS_FUNC})": logging_sr_loss,
                        'lr': optimizer.param_groups[0]['lr'],
                        'Iteration': iteration,
                        })

            logging_segment_loss = logging_sr_loss = logging_tot_loss = 0

            tic = time.time()

        if iteration % args.save_step == 0 and not args.debug:
            model_path = os.path.join(cfg.OUTPUT_DIR, 'model', 'iteration_{}.pth'.format(iteration))
            optimizer_path = os.path.join(cfg.OUTPUT_DIR, 'optimizer', 'iteration_{}.pth'.format(iteration))

            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            os.makedirs(os.path.dirname(optimizer_path), exist_ok=True)

            if args.num_gpus > 1:
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)

            torch.save(optimizer.state_dict(), optimizer_path)

            print('=====> Save Checkpoint to {}'.format(model_path))

        if iteration % args.eval_step == 0:
            with torch.no_grad():
                model.eval() # add
                eval_sr_loss = 0
                eval_segment_loss = 0
                data_len = len(eval_loader)

                psnr_scores = np.array([])
                ssim_scores = np.array([])
                iou_scores = np.array([])

                for imgs, sr_targets, segment_targets in tqdm(eval_loader):
                    segment_loss, sr_loss, segment_preds, sr_preds = model(imgs, sr_targets=sr_targets, segment_targets=segment_targets)

                    if sr_loss is None:
                        # sr_loss = model.sr_loss_fn(sr_images, sr_targets).mean(dim=(1,2,3))
                        loss = segment_loss.mean()
                        eval_segment_loss += loss.item()
                    else:
                        segment_loss, sr_loss = segment_loss.mean(), sr_loss.mean()
                        eval_segment_loss += segment_loss.item()
                        eval_sr_loss += sr_loss.item()

                    if not cfg.MODEL.SR_SEG_INV and cfg.MODEL.SCALE_FACTOR != 1:
                        sr_preds[sr_preds>1] = 1 # clipping
                        sr_preds[sr_preds<0] = 0 # clipping
                        psnr_scores = np.append(psnr_scores, psnr(sr_preds, sr_targets.to("cuda")))
                        ssim_scores = np.append(ssim_scores,ssim(sr_preds, sr_targets.to("cuda")))
                    else:
                        psnr_scores = np.append(psnr_scores, 0)
                        ssim_scores = np.append(ssim_scores, 0)

                    segment_preds = (segment_preds.to("cuda") >=  torch.Tensor([0.5]).to("cuda")).float()
                    iou_scores = np.append(iou_scores, iou(segment_preds, segment_targets.to("cuda")))

                eval_segment_loss /= data_len
                eval_sr_loss /= data_len

                print(f"\nestimation result (iter={iteration}):")
                print(f'=====> Segment_Loss({cfg.SOLVER.SEG_LOSS_FUNC}): {eval_segment_loss:.6f}, SR_Loss({cfg.SOLVER.SR_LOSS_FUNC}): {eval_sr_loss:.6f} PSNR:{sum(psnr_scores)/len(psnr_scores):.4f} SSIM:{sum(ssim_scores)/len(ssim_scores):.4f} IoU:{sum(iou_scores)/len(iou_scores):.4f}')
                # print(f"PSNR:{sum(psnr_scores)/len(psnr_scores):.4f}  SSIM:{sum(ssim_scores)/len(ssim_scores):.4f}  IoU:{sum(iou_scores)/len(iou_scores):.4f}\n")

                if args.wandb_flag:
                    wandb.log({"loss": logging_tot_loss,
                            f"segment_loss_eval({cfg.SOLVER.SEG_LOSS_FUNC})":eval_segment_loss,
                            f"sr_loss_eval({cfg.SOLVER.SR_LOSS_FUNC})": eval_sr_loss,
                            'Iteration': iteration,
                            "PSNR_eval":sum(psnr_scores)/len(psnr_scores),
                            "SSIM_eval":sum(ssim_scores)/len(ssim_scores),
                            "IoU_eval":sum(iou_scores)/len(iou_scores)
                            })
                    
def calc_loss(segment_loss, logging_segment_loss, sr_loss, logging_sr_loss, iteration, cfg):
    segment_loss = segment_loss.mean()
    logging_segment_loss += segment_loss.item()

    if sr_loss != None:
        sr_loss = sr_loss.mean()
        logging_sr_loss += sr_loss.item()

    if cfg.MODEL.SCALE_FACTOR == 1 or cfg.MODEL.SR == "bicubic":
        loss = segment_loss
    elif cfg.MODEL.JOINT_LEARNING:
        loss = (1 - cfg.SOLVER.TASK_LOSS_WEIGHT) * sr_loss + cfg.SOLVER.TASK_LOSS_WEIGHT * segment_loss
        loss = calc_pretrain_loss(loss, segment_loss, sr_loss, iteration, cfg)
    elif not cfg.MODEL.JOINT_LEARNING:
        if not cfg.MODEL.SR_SEG_INV:
            loss = segment_loss
        else:
            loss = sr_loss
        loss = calc_pretrain_loss(loss, segment_loss, sr_loss, iteration, cfg)

    return loss, logging_segment_loss, logging_sr_loss

def calc_pretrain_loss(loss, segment_loss, sr_loss, iteration, cfg):
    if iteration < cfg.SOLVER.SR_PRETRAIN_ITER:
        loss = sr_loss
    if iteration < cfg.SOLVER.SEG_PRETRAIN_ITER:
        loss = segment_loss

    return loss

def fix_1st_stage_model_params(cfg, model, free_1st_stage_model_params, iteration):
    if not cfg.MODEL.JOINT_LEARNING and free_1st_stage_model_params and cfg.MODEL.SR != "bicubic" and cfg.MODEL.SCALE_FACTOR != 1:
        if not cfg.MODEL.SR_SEG_INV and iteration >= cfg.SOLVER.SR_PRETRAIN_ITER:
            print('+++++++ Fix parameters of SR model(1st stage model). +++++++') 
            print('+++++++ Update parameters of segmentation model(2nd stage model). +++++++') 
            for param in model.module.sr_model.parameters():
                param.requires_grad = False
            for param in model.module.segmentation_model.parameters():
                param.requires_grad = True
            free_1st_stage_model_params = False

            if "Boundary" in cfg.SOLVER.SEG_LOSS_FUNC:
                model.module.fix_alpha, model.module.iter = False, 1
        elif cfg.MODEL.SR_SEG_INV and iteration >= cfg.SOLVER.SEG_PRETRAIN_ITER:
            print('+++++++ Fix parameters of segmentation model(1st stage model). +++++++') 
            print('+++++++ Update parameters of SR model(2nd stage model). +++++++') 
            for param in model.module.segmentation_model.parameters():
                param.requires_grad = False
            for param in model.module.sr_model.parameters():
                param.requires_grad = True

            free_1st_stage_model_params = False
    
    return free_1st_stage_model_params

def fix_2nd_stage_model_params(cfg, model):
    if not cfg.MODEL.JOINT_LEARNING and cfg.MODEL.SR != "bicubic" and cfg.MODEL.SCALE_FACTOR != 1:
        if not cfg.MODEL.SR_SEG_INV:
            print('+++++++ Fix parameters of segmentation model(2nd stage model). +++++++') 
            print('+++++++ Update parameters of SR model(1st stage model). +++++++') 
            for param in model.module.segmentation_model.parameters():
                param.requires_grad = False

            if "Boundary" in cfg.SOLVER.SEG_LOSS_FUNC:
                model.module.fix_alpha = True
        else:
            print('+++++++ Fix parameters of SR model(2nd stage model). +++++++') 
            print('+++++++ Update parameters of segmentation model(1st stage model). +++++++') 
            for param in model.module.sr_model.parameters():
                param.requires_grad = False




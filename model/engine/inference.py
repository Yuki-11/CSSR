import os
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import wandb
from model.utils.estimate_metrics import PSNR, SSIM, IoU




def inference_for_ss(args, cfg, model, test_loader):
    """
    aiu_scoures : test_case(=len(test_loader)) x threshold_case(=99)
    """
    fnames = []
    max_iter = len(test_loader)

    psnr_scores = np.array([])
    ssim_scores = np.array([])

    psnr = PSNR()
    ssim = SSIM()
    iou = IoU()
    
    os.makedirs(os.path.dirname(os.path.join(args.output_dirname, "images")), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.join(args.output_dirname, "masks")), exist_ok=True)
    
    if args.test_aiu:
        thresholds = [i*0.01 for i in range(1, 100)]
        iou_mode = "AIU"
        
    else:
        thresholds = [0.5]
        iou_mode = "IoU"

    if args.wandb_flag:
        # --- wandb setting https://docs.wandb.ai/integrations/pytorch --- #
        wandb.init(config=cfg, project=args.wandb_prj_name)
        # if args.dataset_cfg == "":
        #     wandb.init(config=cfg, project=args.wandb_prj_name)
        # else:
        #     wandb.init(config=cfg, project= f"{args.wandb_prj_name}_{args.dataset_cfg}")
        
        wandb.config.update(args)
        wandb.run.name = cfg.OUTPUT_DIR.replace("output/", "")
        # Magic
        wandb.watch(model, log='all')

    print('Evaluation Starts')
    print(f'Number of test dataset : {len(test_loader) * args.batch_size}')

    model.eval()
    for iteration, (imgs, sr_targets, masks, fname) in enumerate(test_loader, 1):
        fnames += list(fname)
        sr_preds, segment_preds = model(imgs)

        # SR evaluation
        if not cfg.MODEL.SR_SEG_INV and cfg.MODEL.SCALE_FACTOR != 1:
            sr_preds[sr_preds>1] = 1 # clipping
            sr_preds[sr_preds<0] = 0 # clipping        
            psnr_scores = np.append(psnr_scores, psnr(sr_preds, sr_targets.to("cuda"))) 
            ssim_scores = np.append(ssim_scores, ssim(sr_preds, sr_targets.to("cuda"))) 
            save_img(args.output_dirname, sr_preds, fname)
        else:
            psnr_scores = np.append(psnr_scores, 0)
            ssim_scores = np.append(ssim_scores, 0)
        
        # Segmentation evaluation
        for iou_th in tqdm(thresholds):
            segment_preds_bi = (segment_preds.to("cuda") >=  torch.Tensor([iou_th]).to("cuda")).float()    
        
            # segment_preds_bi, segment_preds_bi_down = up_scale(segment_preds_bi, cfg)
            if iou_th * 100 % 10 == 0 or iou_th == 0.01 or iou_th == 0.99:
                save_mask(args, segment_preds_bi, fname, iou_th)

            if 'iou_scores' in locals():
                iou_scores = np.append(iou_scores, iou(segment_preds_bi, masks.to("cuda"))[:, np.newaxis], axis=1)
            else:
                # print(iou(segment_preds_bi, masks.to("cuda")).shape)
                iou_scores = np.copy(iou(segment_preds_bi, masks.to("cuda"))[:, np.newaxis])

        if 'aiu_scores' in locals():
            aiu_scores = np.append(aiu_scores, iou_scores, axis=0)
        else:
            aiu_scores = np.copy(iou_scores)
        
        if args.wandb_flag:
        # wandb
            wandb.log({"PSNR_score": psnr_scores[-1], 
                    "SSIM_score":ssim_scores[-1],
                    f"{iou_mode}_scores": np.mean(iou_scores), 
                    })

        del iou_scores
        if iteration % 10 == 0:
            print(f"estimation {iteration/max_iter*100:.4f} % finish!")
            print(f"PSNR_mean:{np.mean(psnr_scores):.4f}  SSIM_mean:{np.mean(ssim_scores):.4f} {iou_mode}_mean:{np.mean(aiu_scores):.4f}")
        
    print(f"estimation finish!!")
    print(f"PSNR_mean:{np.mean(psnr_scores):.4f}  SSIM_mean:{np.mean(ssim_scores):.4f} {iou_mode}_mean:{np.mean(aiu_scores):.4f} ")

    if args.wandb_flag:
        wandb.log({"PSNR_score_mean": np.mean(psnr_scores), 
            "SSIM_score_mean":np.mean(ssim_scores), 
            f"{iou_mode}_scores_mean": np.mean(aiu_scores),
            })
        
        if args.test_aiu:
            plot_metrics_th(aiu_scores, thresholds, "IoU")

    # save_iou_log(aiu_scores, thresholds, fnames, args.output_dirname) # Output IoU scores as csv file.

def save_img(dirname, sr_preds, fname):
    # print(fpath)
    for batch_num in range(sr_preds.size()[0]):
        if sr_preds.shape[1] == 3:
            sr_pred = transforms.ToPILImage(mode='RGB')(sr_preds[batch_num])
        elif sr_preds.shape[1] == 1:
            sr_pred = transforms.ToPILImage(mode='L')(sr_preds[batch_num])
            
        os.makedirs(os.path.dirname(dirname+f"/images/"), exist_ok=True)
        fpath = os.path.join(dirname+f"/images/", f"{fname[batch_num]}")

        sr_pred.save(fpath)
        
        
def save_mask(args, segment_predss, fname, iou_th, add_path=""):
    # print(segment_predss.shape)
    for batch_num in range(segment_predss.size()[0]):
        th_name = f"th_{iou_th:.2f}"
        segment_predss = segment_predss.to("cpu")
        segment_preds = transforms.ToPILImage()(segment_predss[batch_num])
        
        os.makedirs(os.path.dirname(os.path.join(args.output_dirname+f"/masks{add_path}/{th_name}/")), exist_ok=True) 
        mpath = os.path.join(args.output_dirname+f"/masks{add_path}/{th_name}/", f"{fname[batch_num]}".replace("jpg", 'png'))

        segment_preds.save(mpath)     

def plot_metrics_th(metrics_scores, thresholds, metrics):
    metrics_scores = np.mean(metrics_scores, axis=0)
    for iou, th in zip(metrics_scores, thresholds):
        wandb.log({f"{metrics}(thresholds)": iou, 
                "thresholds":th, 
                })

def save_iou_log(aiu_scores, thresholds, fnames, output_dir):
    df = pd.DataFrame(aiu_scores, columns=thresholds, index=fnames)
    df.to_csv(os.path.join(output_dir, 'iou_log.csv'))
    print('IoU log saved!!')
    print(df)
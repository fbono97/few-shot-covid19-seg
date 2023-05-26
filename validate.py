import os
import time
import argparse
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from util import utils
from dataloading.data_utils import DATASET_INFO
from dataloading.datasets import Covid19_Dataset
from dataloading.samplers import InferenceSampler
from models.model_utils import get_model_by_name, ModelLoss



def main(args, cfg):

    device = torch.device('cuda')

    # Set-up logging to console and TensorBoard writer
    logger = utils.get_logger("__main__")
    writer = SummaryWriter(os.path.join(args.logs_path, args.run_id))


    logger.info("====== Parameters and Arguments ======")
    logger.info(args)
    logger.info(cfg)


    logger.info("====== Model ======")
    model = get_model_by_name(cfg["model_name"], device, 'valid', cfg)
    logger.info(f"Model class name: {model.__class__.__name__}")
    
    model = model.to(device)
    model.eval()


    # Load the state dictionary of the trained model      
    try:
        # Try loading checkpoint at the last epoch
        ckpt_id = f"{args.run_id}_ep{cfg['max_epoch']}"
        logger.info(f"Loading checkpoint from {args.checkpoints_path}/{ckpt_id}.pt ...")
        checkpoint = torch.load(os.path.join(args.checkpoints_path, f"{ckpt_id}.pt"))
        model.load_state_dict(checkpoint["model_state_dict"])

    except FileNotFoundError:
        # If specified checkpoint is not available, load latest one
        ckpt_list = sorted(Path(args.checkpoints_path).glob("*.pt"), key=os.path.getmtime)

        if ckpt_list:
            last_ckpt_path = str(ckpt_list[-1])
            logger.info(f"Loading checkpoint from {last_ckpt_path} ...")
            checkpoint = torch.load(last_ckpt_path)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            raise ValueError(f"No available checkpoints at {args.checkpoints_path}")


    # Set up data loader for inference
    logger.info("====== Dataloading ======")
    
    valid_set = Covid19_Dataset(data_path = cfg["data_path"],
                                data_info_path = cfg["data_info_path"],
                                fold = args.fold,
                                mode = "valid",
                                preload = cfg["preload"],
                                seg_masks_union = cfg["seg_masks_union"],
                                norm_level = cfg["norm_level"],
                                norm_type = cfg["norm_type"],
                                repeat_ch = cfg["repeat_ch"],
                                custom_transforms = None)
                                
    valid_sampler = InferenceSampler(valid_set,
                                     fold = args.fold,
                                     k_shot = cfg["k_shot"])
    
    valid_loader = DataLoader(valid_set,
                              batch_sampler = valid_sampler,
                              num_workers = cfg["num_workers"],
                              pin_memory = True,
                              collate_fn = valid_sampler.inference_collate_fn)
     
    
    logger.info("====== Validation started ======")
  
    class_dice = {}
    class_prec = {}
    class_rec = {}

    weights = torch.FloatTensor([cfg["bg_weight"], 1.0]).to(device)
    criterion = ModelLoss(cfg["model_name"], weights)
    logger.info(f"Query loss criterion: {criterion.criterion}")
    
    with torch.no_grad():

        # Loop through each class separately, so inference as 1-way-k-shot segmentation
        for label_name, label in DATASET_INFO["seg_labels"].items():

            logger.info(f"Current label: {label_name}")
            valid_sampler.set_current_label(label)

            query_loss, model_time, patients_dice, patients_prec, patients_rec = inference(
                args, cfg, valid_loader, valid_sampler, label_name, model, criterion, logger, writer, device
            )

            # Log class-wise results averaged across queries
            class_dice_avg, class_dice_std = np.mean(patients_dice), np.std(patients_dice)
            class_prec_avg, class_prec_std = np.mean(patients_prec), np.std(patients_prec)
            class_rec_avg, class_rec_std = np.mean(patients_rec), np.std(patients_rec)

            class_dice[label_name] = f"{class_dice_avg:.4f}+-{class_dice_std:.4f}"
            class_prec[label_name] = f"{class_prec_avg:.4f}+-{class_prec_std:.4f}"
            class_rec[label_name] = f"{class_rec_avg:.4f}+-{class_rec_std:.4f}"

            logger.info(f"Mean query loss: {query_loss:.4f}  "
                        f"Avg inference time: {model_time:.4f}")
            logger.info(
                f"Mean class Dice: {class_dice_avg:.4f}+-{class_dice_std:.4f}  "
                f"Mean class Precision: {class_prec_avg:.4f}+-{class_prec_std:.4f}  "
                f"Mean class Recall: {class_rec_avg:.4f}+-{class_rec_std:.4f}"
            )

    logger.info("====== Inference finished ======")

    # Log final evaluation results
    logger.info(f"Mean Dice: {class_dice}")
    logger.info(f"Mean Precision: {class_prec}")
    logger.info(f"Mean Recall: {class_rec}")

    writer.close()



def inference(args, cfg, valid_loader, valid_sampler, label_name, model, criterion, logger, writer, device):

    query_loss = utils.AverageMeter()
    model_time = utils.AverageMeter()
    eval_metrics = utils.Metrics()

    for n, (supp_img, supp_mask, qry_img, qry_mask) in enumerate(valid_loader):

        supp_img = supp_img.to(device, non_blocking=True)
        supp_mask = supp_mask.to(device, non_blocking=True)
        qry_img = qry_img.to(device, non_blocking=True)
        qry_mask = qry_mask.long().to(device, non_blocking=True)

        # The query volume is divided into chunks to decrease memory usage 
        z = 0
        qry_pred = []
        start_time = time.time()  
        
        while z < qry_mask.shape[0]:
            # Each query chunk is segmented individually with the same support
            qry_chunk = qry_img[:, z : z+cfg["query_chunk_size"]]
            qry_pred_chunk, _, _ = model(supp_img, supp_mask, qry_chunk)
            qry_pred.append(qry_pred_chunk)
            z += cfg["query_chunk_size"]

        model_time.update(time.time() - start_time)

        # The separate chunks are combined to form the whole segmented query volume
        qry_pred = torch.cat(qry_pred)

        q_loss = criterion(qry_pred, qry_mask)
        query_loss.update(q_loss.item())

        # At inference, segmentation is 1-way so predicted mask is already binary
        qry_pred = qry_pred.argmax(axis=1).cpu()
        dice, prec, rec = eval_metrics.get_patient_scores(qry_pred, qry_mask.cpu())

        # Save predictions
        file_name = f"{valid_sampler.query_ids[n]}_{label_name}.pt"
        torch.save(qry_pred.type(torch.uint8),
                   os.path.join(args.img_save_path, file_name))


        logger.info(f"Query volume [{n+1}/{len(valid_loader)}]:  "
                    f"Dice {dice:.4f}  "
                    f"Precision {prec:.4f}  "
                    f"Recall {rec:.4f}  "
                    f"Cumulative mean query loss {query_loss.val:.4f}")          

        writer.add_scalar(f"{label_name}/valid_query_loss", q_loss.item(), n+1)
        writer.add_scalar(f"{label_name}/Dice", dice, n+1)
        writer.add_scalar(f"{label_name}/Precision", prec, n+1)
        writer.add_scalar(f"{label_name}/Recall", rec, n+1)

    return (query_loss.avg,
            model_time.avg,
            eval_metrics.patients_dice,
            eval_metrics.patients_prec,
            eval_metrics.patients_rec)



def build_args():
    
    parser = argparse.ArgumentParser(
        description="Few-shot semantic segmentation of COVID-19-CT-Seg dataset"
    )
    parser.add_argument("--fold", type=int,
                        help="Which dataset's fold to use for cross-validation")
    parser.add_argument("--run_id", type=str,
                        help="File identifier for tensorboard logging and checkpointing")
    parser.add_argument("--config_file", type=str,
                        help="Path to yaml configuration file")
    parser.add_argument("--logs_path", type=str,
                        help="Path to dir containing TensorBoard logs")
    parser.add_argument("--checkpoints_path", type=str,
                        help="Path to dir containing model checkpoints")
    parser.add_argument("--img_save_path", type=str,
                        help="Path to dir containing model predictions")

    args = parser.parse_args()
    cfg = utils.load_config(args.config_file)

    utils.check_mkdir(args.logs_path)
    utils.check_mkdir(args.img_save_path)

    return args, cfg



def run_main():
    args, cfg = build_args()
    main(args, cfg)

if __name__ == "__main__":
    run_main()
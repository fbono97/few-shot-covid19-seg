import os
import time
import random
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from util import utils
from dataloading.datasets import Covid19_Dataset
from dataloading.samplers import FewShotSampler
from models.model_utils import get_model_by_name, ModelLoss



def main(args, cfg):

    device = torch.device('cuda')

    # Set global seed and deterministic options for reproducibility across runs
    if args.seed is not None:
        utils.set_global_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Set-up logging to console and TensorBoard writer
    logger = utils.get_logger("__main__")
    writer = SummaryWriter(os.path.join(args.logs_path, args.run_id))


    logger.info("====== Parameters and Arguments ======")
    logger.info(args)
    logger.info(cfg)

    
    logger.info("====== Model ======")
    model = get_model_by_name(cfg["model_name"], device, 'train', cfg)
    logger.info(f"Model class name: {model.__class__.__name__}")

    train_p, nontrain_p, total_p = utils.get_model_params(model)
    logger.info("Number of model parameters: "
                f"{train_p} trainable, {nontrain_p} non-trainable, {total_p} total.")
    
    model = model.to(device)


    # Set up data loader with few-shot sampler
    train_transforms = None   # Default augment transforms will be applied

    if args.seed:   # Preserve reproducibility in multi-process data loading
        seed_worker = lambda worker_id: random.seed(args.seed + worker_id)
    else:
        seed_worker = None


    logger.info("====== Dataloading ======")
    
    train_set = Covid19_Dataset(data_path = cfg["data_path"],
                                data_info_path = cfg["data_info_path"],
                                fold = args.fold,
                                mode = "train",
                                preload = cfg["preload"],
                                seg_masks_union = cfg["seg_masks_union"],
                                norm_level = cfg["norm_level"],
                                norm_type = cfg["norm_type"],
                                repeat_ch = cfg["repeat_ch"],
                                custom_transforms = train_transforms)
                                
    train_sampler = FewShotSampler(train_set,
                                   n_way = cfg["n_way"],
                                   k_shot = cfg["k_shot"],
                                   n_query = cfg["n_query"],
                                   batch_size = cfg["batch_size"],
                                   n_tasks = cfg["n_tasks_per_epoch"])
    
    train_loader = DataLoader(train_set,
                              batch_sampler = train_sampler,
                              num_workers = cfg["num_workers"],
                              pin_memory = True,
                              collate_fn = train_sampler.episodic_collate_fn,
                              worker_init_fn = seed_worker)


    # Training specs
    start_epoch = 0

    weights = torch.FloatTensor([cfg["bg_weight"]] + [1.0]*cfg["n_way"]).to(device)
    criterion = ModelLoss(cfg["model_name"], weights)
    logger.info(f"Query loss criterion: {criterion.criterion}")

    optimizer = torch.optim.SGD(model.parameters(),
                                lr = cfg["base_lr"],
                                momentum = cfg["momentum"],
                                weight_decay = cfg["weight_decay"])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=cfg["max_epoch"] * len(train_loader),
                                                           eta_min=cfg["eta_min"])
    

    # Load model checkpoint to resume training if applicable
    if args.resume_from_epoch:       
        try:
            # Try loading checkpoint at specified epoch
            ckpt_id = f"{args.run_id}_ep{args.resume_from_epoch}"
            logger.info(f"Loading checkpoint from {args.checkpoints_path}/{ckpt_id}.pt ...")
            checkpoint = torch.load(os.path.join(args.checkpoints_path, f"{ckpt_id}.pt"))

            start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        except FileNotFoundError:
            # If specified checkpoint is not available, load latest one
            ckpt_list = sorted(Path(args.checkpoints_path).glob("*.pt"), key=os.path.getmtime)

            if ckpt_list:
                last_ckpt_path = str(ckpt_list[-1])
                logger.info(f"Loading checkpoint from {last_ckpt_path} ...")
                checkpoint = torch.load(last_ckpt_path)

                start_epoch = checkpoint["epoch"]
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            else:
                raise ValueError(f"No available checkpoints at {args.checkpoints_path}")
        
    
    logger.info("====== Training started ======")
    model.train()

    for epoch in range(start_epoch, cfg['max_epoch']):

        if args.seed:  # For reproducibility after loading from checkpoint
            utils.set_global_seed(args.seed + epoch)

        query_loss, align_loss, threshold_loss, total_loss = train(
            cfg, train_loader, model, optimizer, scheduler, criterion, epoch, logger, writer, device
        )

        writer.add_scalar('Loss_epoch/query_loss', query_loss, epoch+1)
        writer.add_scalar('Loss_epoch/align_loss', align_loss, epoch+1)
        writer.add_scalar('Loss_epoch/threshold_loss', threshold_loss, epoch+1)
        writer.add_scalar('Loss_epoch/total_loss', total_loss, epoch+1)

        # Save model checkpoint
        if ((epoch+1) % cfg["save_ckpt_freq"]) == 0:

            save_path = os.path.join(args.checkpoints_path, f"{args.run_id}_ep{epoch+1}.pt")
            logger.info(f"Saving checkpoint at {save_path}")
            torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 
                        'scheduler_state_dict': scheduler.state_dict(),
                        }, save_path)
            
        if epoch+1 == args.checkpoint_epoch:
            logger.info(f"Checkpoint reached at epoch {args.checkpoint_epoch}. "
                        "You may resume training by loading the latest checkpoint.")
            break

            
    logger.info("====== Training finished ======")
    save_path = os.path.join(args.checkpoints_path, f"{args.run_id}_ep{epoch+1}.pt")

    logger.info(f"Saving final checkpoint at {save_path}")
    torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                }, save_path)

    writer.close()



def train(cfg, train_loader, model, optimizer, scheduler, criterion, epoch, logger, writer, device):

    # Keep track of losses and training time as training progresses
    query_loss = utils.AverageMeter()
    align_loss = utils.AverageMeter()
    threshold_loss = utils.AverageMeter() 
    total_loss = utils.AverageMeter()
    iter_time = utils.AverageMeter()

    max_iter = cfg['max_epoch'] * len(train_loader)
    end_time = time.time()

    for i_iter, (supp_img, supp_mask, qry_img, qry_mask) in enumerate(train_loader):

        current_iter = epoch * len(train_loader) + i_iter + 1
        
        # Training section
        supp_img = supp_img.to(device, non_blocking=True)
        supp_mask = supp_mask.to(device, non_blocking=True)
        qry_img = qry_img.to(device, non_blocking=True)
        qry_mask = qry_mask.long().to(device, non_blocking=True)

        optimizer.zero_grad()
        qry_pred, a_loss, t_loss = model(supp_img, supp_mask, qry_img)
        
        q_loss = criterion(qry_pred, qry_mask)
        loss = q_loss + a_loss * cfg["align_loss_weight"] + t_loss
    
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update all loss values and training time left
        query_loss.update(q_loss.item())
        align_loss.update(a_loss.item())
        threshold_loss.update(t_loss.item())
        total_loss.update(loss.item())

        iter_time.update(time.time() - end_time)
        remain_time = utils.get_remaining_train_time(max_iter,
                                                     current_iter,
                                                     iter_time.avg)
        end_time = time.time()
        
        # Log results and training info
        if ((i_iter+1) % cfg["print_freq"]) == 0:
            logger.info(f"Epoch [{epoch+1}/{cfg['max_epoch']}][{i_iter+1}/{len(train_loader)}]:  "
                        f"Remain {remain_time}  "
                        f"Query loss {query_loss.val:.4f}  "
                        f"Align loss {align_loss.val:.4f}  "
                        f"Threshold loss {threshold_loss.val:.4f}  "                     
                        f"Total loss {total_loss.val:.4f}")
            
        writer.add_scalar('Loss_iter/query_loss', q_loss.item(), current_iter)
        writer.add_scalar('Loss_iter/align_loss', a_loss.item(), current_iter)
        writer.add_scalar('Loss_iter/threshold_loss', t_loss.item(), current_iter)
        writer.add_scalar('Loss_iter/total_loss', loss.item(), current_iter) 

    return query_loss.avg, align_loss.avg, threshold_loss.avg, total_loss.avg




def build_args():
    
    parser = argparse.ArgumentParser(
        description="Few-shot semantic segmentation of COVID-19-CT-Seg dataset"
    )
    parser.add_argument("--seed", type=int,
                        help="Starting random seed")
    parser.add_argument("--fold", type=int,
                        help="Which dataset's fold to use for cross-validation")
    parser.add_argument("--run_id", type=str,
                        help="File identifier for tensorboard logging and checkpointing")
    parser.add_argument("--checkpoint_epoch", type=int, nargs='?', default=None,
                        help="If not None, epoch at which to pause training")
    parser.add_argument("--resume_from_epoch", type=int, nargs='?', default=None,
                        help="If not None, epoch from which to resume training")
    parser.add_argument("--config_file", type=str,
                        help="Path to yaml configuration file")
    parser.add_argument("--logs_path", type=str,
                        help="Path to dir containing TensorBoard logs")
    parser.add_argument("--checkpoints_path", type=str,
                        help="Path to dir containing model checkpoints")

    args = parser.parse_args()
    cfg = utils.load_config(args.config_file)

    utils.check_mkdir(args.logs_path)
    utils.check_mkdir(args.checkpoints_path)

    return args, cfg



def run_main():
    args, cfg = build_args()
    main(args, cfg)

if __name__ == "__main__":
    run_main()

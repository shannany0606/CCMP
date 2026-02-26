from datetime import datetime
import numpy as np 
import torch
import torch.nn.functional as F
from tqdm import tqdm
import gc

import os
from torchvision.utils import save_image

import sys
sys.path.append("../../evaluation") 
import losses
import metrics

from torch.cuda.amp import autocast
from sklearn.metrics import balanced_accuracy_score

def overlay_mask_on_image(image, mask, color=(1, 0, 0), alpha=0.5):
    """
    Overlay a binary mask on an RGB image, only altering masked areas.

    Args:
        image (Tensor): RGB image tensor (3, H, W), values in [0, 1]
        mask (Tensor): Binary mask tensor (1, H, W) or (H, W), values in {0, 1}
        color (tuple): RGB overlay color, values in [0, 1]
        alpha (float): Transparency of overlay on masked region

    Returns:
        Tensor: Overlay image (3, H, W)
    """
    if mask.dim() == 3:
        mask = mask.squeeze(0)  # (H, W)

    # If mask is all zeros, return original image
    if mask.sum() == 0:
        return image.clone()

    masked = mask.bool()
    overlay = image.clone()
    
    # Create a (3, N) tensor for color broadcasting
    num_pixels = masked.sum().item()
    color_tensor = torch.tensor(color, device=image.device).view(3, 1).expand(3, num_pixels)

    overlay[:, masked] = (
        image[:, masked] * (1 - alpha) + color_tensor * alpha
    )

    return overlay

def log_image_predictions(writer, T1, T2, FM1, output, target, epoch, tag_prefix=''):
    """
    Logs input images and predicted/ground truth masks to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        T1, T2, FM1: Input tensors (B, 3, H, W), normalized with ImageNet stats
        output: Predicted mask tensor (B, 1, H, W)
        target: Ground truth mask tensor (B, 1, H, W)
        epoch: Current epoch
        tag_prefix: Optional string prefix for tag names
    """
    def denorm(img):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(img.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(img.device)
        return img * std + mean

    def clip(img):
        return torch.clamp(img, 0, 1)

    with torch.no_grad():
        pred_mask = (output > 0.5).float()

        num_vis = min(2, T1.size(0))
        for vis_idx in range(num_vis):
            T1_vis = denorm(T1[vis_idx].cpu())
            T2_vis = denorm(T2[vis_idx].cpu())
            FM1_vis = FM1[vis_idx].cpu()
            pred_vis = pred_mask[vis_idx].cpu()
            target_vis = target[vis_idx].cpu()

            # Overlay only in masked area
            T2_with_pred = overlay_mask_on_image(T2_vis, pred_vis > 0.5, color=(1, 0, 0))
            T2_with_gt = overlay_mask_on_image(T2_vis, target_vis > 0.5, color=(0, 1, 0))
            T1_with_gt = overlay_mask_on_image(T1_vis, FM1_vis > 0.5, color=(0, 1, 0))
            
            writer.add_image(f'{tag_prefix}T1_{vis_idx}', clip(T1_vis), epoch)
            writer.add_image(f'{tag_prefix}T2_{vis_idx}', clip(T2_vis), epoch)
            writer.add_image(f'{tag_prefix}FM1_{vis_idx}', clip(FM1_vis), epoch)
            writer.add_image(f'{tag_prefix}GT Mask_{vis_idx}', target_vis, epoch)
            writer.add_image(f'{tag_prefix}Pred Mask_{vis_idx}', pred_vis, epoch)
            writer.add_image(f'{tag_prefix}T1_GTMask_{vis_idx}', clip(T1_with_gt), epoch)
            writer.add_image(f'{tag_prefix}T2_GTMask_{vis_idx}', clip(T2_with_gt), epoch)
            writer.add_image(f'{tag_prefix}T2_PredMask_{vis_idx}', clip(T2_with_pred), epoch)

def save_visualization_to_disk(T1, T2, FM1, output, target, Reversed_output, pth1, pth2, output_dir, sample_offset=0):
    def denorm(img):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(img.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(img.device)
        return img * std + mean

    def clip(img):
        return torch.clamp(img, 0, 1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        pred_mask = (output > 0.5).float()
        Reversed_pred_mask = (Reversed_output > 0.5).float()
        batch_size = T1.size(0)
        
        for i in range(batch_size):
            sample_idx = i + sample_offset
            
            T1_vis = denorm(T1[i].cpu())
            FM1_vis = FM1[i].cpu()
            T2_vis = denorm(T2[i].cpu())
            pred_vis = pred_mask[i].cpu()
            target_vis = target[i].cpu()
            Reversed_pred_vis = Reversed_pred_mask[i].cpu()

            T1_with_gt = overlay_mask_on_image(T1_vis, FM1_vis > 0.5, color=(0, 1, 0))
            T1_with_Reversed_pred = overlay_mask_on_image(T1_vis, Reversed_pred_vis > 0.5, color=(1, 0, 0))
            T2_with_pred = overlay_mask_on_image(T2_vis, pred_vis > 0.5, color=(1, 0, 0))
            T2_with_gt = overlay_mask_on_image(T2_vis, target_vis > 0.5, color=(0, 1, 0))
            
            save_image(clip(T1_vis), f"{output_dir}/{sample_idx:03d}_T1.png")
            save_image(clip(T1_with_Reversed_pred), f"{output_dir}/{sample_idx:03d}_T1_ReversedPredMask.png")
            save_image(clip(T1_with_gt), f"{output_dir}/{sample_idx:03d}_T1_GTMask.png")
            save_image(clip(T2_vis), f"{output_dir}/{sample_idx:03d}_T2.png")
            save_image(clip(T2_with_pred), f"{output_dir}/{sample_idx:03d}_T2_PredMask.png")
            save_image(clip(T2_with_gt), f"{output_dir}/{sample_idx:03d}_T2_GTMask.png")
            # save_image(clip(T2_with_gt), f"{output_dir}/{sample_idx:03d}_T2_GTMask_{pth2[i]}.png")

def getIoU(gt_mask, pred_mask, thresh=0.5): 
    gt_mask = gt_mask > 0.5
    pred_mask_bin = pred_mask > thresh
    intersection = torch.logical_and(gt_mask, pred_mask_bin).sum(dim=(1, 2, 3)).float()
    union = torch.logical_or(gt_mask, pred_mask_bin).sum(dim=(1, 2, 3)).float()
    eps = 1e-6  # avoid division by zero
    return (intersection / (union + eps)).mean()

def fg_recall(gt_mask, pred_mask, thresh=0.5):
    gt_mask_bin = gt_mask > 0.5
    pred_mask_bin = pred_mask > thresh
    correct_foreground = torch.logical_and(gt_mask_bin, pred_mask_bin).sum(dim=(1, 2, 3)).float()
    total_foreground = gt_mask_bin.sum(dim=(1, 2, 3)).float()
    eps = 1e-6  # avoid division by zero
    acc = correct_foreground / (total_foreground + eps)
    return acc.mean()

def loss_calculation(loss_mask_list, dice_loss_list, cls_loss_list, dice_weight, cls_weight, aux_weight, n_aux_layers):
    loss = loss_mask_list[-1] + dice_weight * dice_loss_list[-1] + cls_weight * cls_loss_list[-1]
    if abs(aux_weight) < 1e-6:
        return loss, torch.tensor(0.0)

    aux_loss = 0
    for i in range(n_aux_layers):
        aux_loss += loss_mask_list[i] + dice_weight * dice_loss_list[i] + cls_weight * cls_loss_list[i]
    aux_loss = aux_weight * aux_loss
    return aux_loss + loss, aux_loss

def trainEpoch(trainLoader,
               netEncoder,
               optimizer,
               scaler,
               lr_scheduler,
               history,
               Loss,
               logger,
               iter_epoch,
               epoch,
               writer,
               dice_weight,
               consistency_dice_weight,
               cls_weight,
               aux_weight,
               n_aux_layers,
               consistency_weight,
               accumulation_steps=16,
               warmup=False,
               use_amp=False,
               check_data=False) : 

    netEncoder.train()

    if not hasattr(trainEpoch, 'has_saved_data'):
        trainEpoch.has_saved_data = not check_data
    if not hasattr(trainEpoch, 'iter_counter'):
        trainEpoch.iter_counter = 0
    
    save_dir = "check_train_data"
    if not trainEpoch.has_saved_data:
        sample_count = 0
    
    loss_log, loss_mask_log, loss_dice_log, loss_cls_log, loss_aux_log, loss_consistency_log, acc_log, miou_log = [], [], [], [], [], [], [], []
    trainLoader_iter = iter(trainLoader)
    
    optimizer.zero_grad()
    
    gt_exist_list, pred_exist_list = [], []

    current_lr = lr_scheduler.update_lr(trainEpoch.iter_counter)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    
    for batch_id in tqdm(range(iter_epoch)):

        try:
            batch = next(trainLoader_iter)
        except:
            trainLoader_iter = iter(trainLoader)
            batch = next(trainLoader_iter)

        T1, T2, FM1, target, pth1, pth2, exist = batch['T1'].cuda(), batch['T2'].cuda(), batch['FM1'].cuda(), batch['target2'].cuda(), batch['pth1'], batch['pth2'], batch['exist'].cuda()
        gt_exist_list.extend([int(x) for x in exist])
        with autocast(enabled=use_amp):
            output_list, cls_score_list = netEncoder(T1, FM1, T2)
            
            # BCEWithLogitsLoss
            loss_mask_list = [Loss(output_i, target) for output_i in output_list]

            dice_loss_list = [losses.dice_loss_with_logits(output_i, target).mean() for output_i in output_list]
             #sum()->mean(): prevent the loss being influenced by the batch size

            cls_loss_list = [Loss(cls_score_i, exist) for cls_score_i in cls_score_list]
            pred_exist_list.extend([int(torch.sigmoid(cls_score_i) > 0.5) for cls_score_i in cls_score_list[-1]])

            loss, aux_loss = loss_calculation(loss_mask_list, dice_loss_list, cls_loss_list, dice_weight, cls_weight, aux_weight, n_aux_layers)

            output = torch.sigmoid(output_list[-1])

            # consistency loss (self-supervised)
            if abs(consistency_weight) > 1e-6:
                FM2 = output.type(torch.FloatTensor).cuda()
                Reversed_output_list, _ = netEncoder(T2, FM2, T1)
                if abs(consistency_dice_weight) > 1e-6:
                    consistency_loss = Loss(Reversed_output_list[-1], FM1) + consistency_dice_weight * losses.dice_loss_with_logits(Reversed_output_list[-1], FM1).mean()          
                else:
                    consistency_loss = Loss(Reversed_output_list[-1], FM1)
                loss += consistency_weight * consistency_loss
            else:
                consistency_loss = torch.tensor(0.0)

        scaler.scale(loss / accumulation_steps).backward()
        
        if (batch_id + 1) % accumulation_steps == 0 or (batch_id + 1) == iter_epoch:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            trainEpoch.iter_counter += 1
            
            current_lr = lr_scheduler.update_lr(trainEpoch.iter_counter)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

        loss_log.append(loss.item())
        loss_mask_log.append(loss_mask_list[-1].item())
        loss_dice_log.append(dice_loss_list[-1].item())
        loss_cls_log.append(cls_loss_list[-1].item())
        loss_aux_log.append(aux_loss.item())
        loss_consistency_log.append(consistency_loss.item())
        with torch.no_grad() :
            acc = fg_recall(target, output)
            miou = getIoU(target, output)
            acc_log.append(acc.item())
            miou_log.append(miou.item()) 
            
        if batch_id % 100 == 99: 
            for g in optimizer.param_groups:
                lr_print = g['lr']
                break
            msg = '{} Batch id {:d}, Lr {:.6f}; \t | Loss : {:.3f}, Mask : {:.3f}, Dice : {:.3f}, cls : {:.3f}, aux : {:.3f}, consistency : {:.3f} |  Acc : {:.3f}%, mIoU : {:.3f} \t '.format(datetime.now().time(), batch_id + 1, lr_print, np.mean(loss_log), np.mean(loss_mask_log), np.mean(loss_dice_log), np.mean(loss_cls_log), np.mean(loss_aux_log), np.mean(loss_consistency_log), np.mean(acc_log) * 100, np.mean(miou_log))
            logger.info(msg)
            msg = f"Balanced accuracy: {balanced_accuracy_score(gt_exist_list, pred_exist_list):.3f} | gt_negative: {len(gt_exist_list) - sum(gt_exist_list)}/{len(gt_exist_list)} | pred_negative: {len(pred_exist_list) - sum(pred_exist_list)}/{len(pred_exist_list)}"
            logger.info(msg)

            torch.cuda.empty_cache()

        # save visualization data only at the first run
        if not trainEpoch.has_saved_data:
            batch_size = T1.size(0)
            save_visualization_to_disk(
                T1, T2, FM1, output, target, Reversed_output_list[-1], pth1, pth2,  
                save_dir, sample_offset=sample_count
            )
            sample_count += batch_size
            if sample_count >= 200:
                trainEpoch.has_saved_data = True
        
    history['trainLoss'].append(np.mean(loss_log))
    history['trainLossMask'].append(np.mean(loss_mask_log))
    history['trainLossDice'].append(np.mean(loss_dice_log))
    history['trainLossCls'].append(np.mean(loss_cls_log))
    history['trainLossAux'].append(np.mean(loss_aux_log))
    history['trainLossConsistency'].append(np.mean(loss_consistency_log))
    history['trainAcc'].append(np.mean(acc_log))
    history['trainMIoU'].append(np.mean(miou_log))

    if not warmup:
        writer.add_scalar('train_loss', history['trainLoss'][-1], epoch * iter_epoch + batch_id)
        writer.add_scalar('train_mask_loss', history['trainLossMask'][-1], epoch * iter_epoch + batch_id)
        writer.add_scalar('train_dice_loss', history['trainLossDice'][-1], epoch * iter_epoch + batch_id)
        writer.add_scalar('train_cls_loss', history['trainLossCls'][-1], epoch * iter_epoch + batch_id)
        writer.add_scalar('train_aux_loss', history['trainLossAux'][-1], epoch * iter_epoch + batch_id)
        writer.add_scalar('train_consistency_loss', history['trainLossConsistency'][-1], epoch * iter_epoch + batch_id)
        writer.add_scalar('train_acc', history['trainAcc'][-1], epoch * iter_epoch + batch_id)
        writer.add_scalar('train_mIoU', history['trainMIoU'][-1], epoch * iter_epoch + batch_id)
        log_image_predictions(writer, T1, T2, FM1, output, target, epoch * iter_epoch + batch_id, tag_prefix='Train_Vis_')
    
    gc.collect()
    torch.cuda.empty_cache()

    return netEncoder, optimizer, history

def posttrainEpoch(trainLoader,
                    netEncoder,
                    optimizer,
                    scaler,
                    lr_scheduler,
                    history,
                    Loss,
                    logger,
                    iter_epoch,
                    epoch,
                    writer,
                    aux_weight,
                    n_aux_layers,
                    accumulation_steps=16,
                    warmup=False,
                    use_amp=False) : 

    netEncoder.eval()
    netEncoder.cls_branch.train()

    if not hasattr(posttrainEpoch, 'iter_counter'):
        posttrainEpoch.iter_counter = 0
    
    loss_log, VA_log = [], []
    trainLoader_iter = iter(trainLoader)
    
    optimizer.zero_grad()
    
    gt_exist_list, pred_exist_list = [], []

    current_lr = lr_scheduler.update_lr(posttrainEpoch.iter_counter)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    
    for batch_id in tqdm(range(iter_epoch)):

        try:
            batch = next(trainLoader_iter)
        except:
            trainLoader_iter = iter(trainLoader)
            batch = next(trainLoader_iter)

        T1, T2, FM1, exist = batch['T1'].cuda(), batch['T2'].cuda(), batch['FM1'].cuda(), batch['exist'].cuda()
        gt_exist_list.extend([int(x) for x in exist])

        with autocast(enabled=use_amp):
            _, cls_score_list = netEncoder(T1, FM1, T2)
            
            cls_loss_list = [Loss(cls_score_i, exist) for cls_score_i in cls_score_list]
            pred_exist_list.extend([int(torch.sigmoid(cls_score_i) > 0.5) for cls_score_i in cls_score_list[-1]])

            loss = cls_loss_list[-1]
            if abs(aux_weight) > 1e-6:
                aux_loss = 0
                for i in range(n_aux_layers):
                    aux_loss += cls_loss_list[i]
                aux_loss = aux_weight * aux_loss
                loss += aux_loss   

        scaler.scale(loss / accumulation_steps).backward()
        
        if (batch_id + 1) % accumulation_steps == 0 or (batch_id + 1) == iter_epoch:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            posttrainEpoch.iter_counter += 1
            
            current_lr = lr_scheduler.update_lr(posttrainEpoch.iter_counter)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

        loss_log.append(loss.item())
            
        if batch_id % 100 == 99: 
            for g in optimizer.param_groups:
                lr_print = g['lr']
                break
            VA = balanced_accuracy_score(gt_exist_list, pred_exist_list)
            VA_log.append(VA)
            msg = f'{datetime.now().time()} Batch id {batch_id + 1}, Lr {lr_print:.6f}; \t | Loss: {np.mean(loss_log):.3f} | Balanced accuracy: {VA:.3f} | gt_negative: {len(gt_exist_list) - sum(gt_exist_list)}/{len(gt_exist_list)} | pred_negative: {len(pred_exist_list) - sum(pred_exist_list)}/{len(pred_exist_list)}  \t '
            logger.info(msg)

            torch.cuda.empty_cache()
        
    history['posttrainLoss'].append(np.mean(loss_log))
    history['posttrainVA'].append(np.mean(VA_log))

    if not warmup:
        writer.add_scalar('posttrain_loss', history['posttrainLoss'][-1], epoch * iter_epoch + batch_id)
        writer.add_scalar('posttrain_va', history['posttrainVA'][-1], epoch * iter_epoch + batch_id)
    
    gc.collect()
    torch.cuda.empty_cache()

    return netEncoder, optimizer, history

@torch.no_grad()
def evalEpoch(valLoader,
               netEncoder,
               history,
               Loss,
               iter_epoch,
               epoch,
               writer,
               n_aux_layers,
               use_amp=False,
               check_data=False) : 

    if not hasattr(evalEpoch, 'has_saved_data'):
        evalEpoch.has_saved_data = not check_data
    
    save_dir = "check_val_data"
    if not evalEpoch.has_saved_data:
        sample_count = 0
    
    netEncoder.eval()
    miou_log, miou_480_log = [], []
    valLoader_iter = iter(valLoader)
    
    for batch_id in tqdm(range(iter_epoch)):   
        
        try:
            batch = next(valLoader_iter)
        except:
            valLoader_iter = iter(valLoader)
            batch = next(valLoader_iter)

        T1, T2, FM1, target2, target2_480, pth1, pth2, exist = batch['T1'].cuda(), batch['T2'].cuda(), batch['FM1'].cuda(), batch['target2'].cuda(), batch['target2_480'].cuda(), batch['pth1'], batch['pth2'], batch['exist'].cuda()

        with autocast(enabled=use_amp):
            output_list, _ = netEncoder(T1, FM1, T2)
            output = torch.sigmoid(output_list[-1])
        
        miou= getIoU(target2, output)

        output_480 = F.interpolate(output, size=(480, 480), mode='bilinear', align_corners=False)
        miou_480 = getIoU(target2_480, output_480)

        miou_log.append(miou.item())
        miou_480_log.append(miou_480.item())
        
        # save visualization data only at the first run
        if not evalEpoch.has_saved_data:
            batch_size = T1.size(0)
            # save_visualization_to_disk(
            #     T1, T2, FM1, output, target2, pth1, pth2,  
            #     save_dir, sample_offset=sample_count
            # )
            sample_count += batch_size
            if sample_count >= 200:
                evalEpoch.has_saved_data = True

    history['valMIoU'].append(np.mean(miou_log))
    history['valMIoU_480'].append(np.mean(miou_480_log))

    writer.add_scalar('val_mIoU', history['valMIoU'][-1], epoch * iter_epoch + batch_id)
    writer.add_scalar('val_mIoU_480', history['valMIoU_480'][-1], epoch * iter_epoch + batch_id)
    log_image_predictions(writer, T1, T2, FM1, output, target2, epoch * iter_epoch + batch_id, tag_prefix='Val_Vis_')
    
    gc.collect()
    torch.cuda.empty_cache()
    netEncoder.train()
    return netEncoder, history

@torch.no_grad()
def testEpoch(testLoader, netEncoder, history, iter_epoch, epoch, writer, warmup=False, use_amp=False) : 
    
    netEncoder.eval()
    
    iou_log, iou_log_480 = [], []
    testLoader_iter = iter(testLoader)
    
    for batch_id in tqdm(range(iter_epoch)):   
        
        try:
            batch = next(testLoader_iter)
        except:
            testLoader_iter = iter(testLoader)
            batch = next(testLoader_iter)

        ## put all into cuda
        T1, T2, FM1, target2, target2_480 = batch['T1'].cuda(), batch['T2'].cuda(), batch['FM1'].cuda(), batch['target2'].cuda(), batch['target2_480'].cuda()
        
        with autocast(enabled=use_amp):
            O2_pred_list, _ = netEncoder(T1, FM1, T2)
            O2_pred = torch.sigmoid(O2_pred_list[-1])

        for i in range(len(O2_pred)):
            pred_mask = (O2_pred[i] > 0.5).float()
            gt_mask = target2[i]
            
            iou = metrics.db_eval_iou(gt_mask.cpu().numpy(), pred_mask.cpu().numpy())
            iou_log.append(iou)
        
        O2_pred_480 = F.interpolate(O2_pred, size=(480, 480), mode='bilinear', align_corners=False)
        for i in range(len(O2_pred_480)):
            pred_mask = (O2_pred_480[i] > 0.5).float()
            gt_mask = target2_480[i]
            
            iou = metrics.db_eval_iou(gt_mask.cpu().numpy(), pred_mask.cpu().numpy())
            iou_log_480.append(iou)
    
    mean_iou = np.mean(iou_log)
    history['testMIoU'].append(mean_iou)
    writer.add_scalar('test_mIoU', mean_iou, epoch * iter_epoch + batch_id)

    mean_iou_480 = np.mean(iou_log_480)
    history['testMIoU_480'].append(mean_iou_480)
    writer.add_scalar('test_mIoU_480', mean_iou_480, epoch * iter_epoch + batch_id)

    netEncoder.train()
    return netEncoder, history
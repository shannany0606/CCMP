# coding=utf-8
import os 
import sys 
sys.path.append('model/')
import numpy as np 
from PIL import Image
import cv2
import json
import copy
import time

import torch 
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.utils as vutils
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from train.csegmentor import ConditionalSegmentationModel

import tqdm

from pycocotools import mask as mask_utils
import utils
import train.losses as losses

from torch.cuda.amp import autocast, GradScaler

MASKThresh = 0.5

def overlay_mask_on_image(image, mask, color=(1, 0, 0), alpha=0.5):
    """
    Overlay a binary mask on an RGB image, only altering masked areas.

    Args:
        image (numpy array): RGB image (H, W, 3), values in [0, 1]
        mask (numpy array): Binary mask (H, W), values in {0, 1}
        color (tuple): RGB overlay color, values in [0, 1]
        alpha (float): Transparency of overlay on masked region

    Returns:
        numpy array: Overlay image (H, W, 3)
    """
    # Ensure mask and image share the same height/width
    h, w = image.shape[:2]
    
    if len(mask.shape) == 3 and mask.shape[2] == 1:
        mask = mask[..., 0]  # (H, W)
    
    # Ensure mask and image sizes match; use reshape_img_war
    if mask.shape[:2] != (h, w):
        mask = reshape_img_war(mask, (w, h))
    
    # If mask is all zeros, return original image
    if mask.sum() == 0:
        return image.copy()

    masked = mask > 0.5
    overlay = image.copy()
    
    overlay[masked] = overlay[masked] * (1 - alpha) + np.array(color) * alpha
    
    return overlay

def save_visualization(image1, image2, mask1, pred_mask_before, pred_mask_after, save_path, prefix):
    """
    Save visualization of images and masks before and after TTT.
    
    Args:
        image1: Source image (H, W, 3)
        image2: Target image (H, W, 3)
        mask1: Source mask (H, W)
        pred_mask_before: Predicted mask before TTT (H, W)
        pred_mask_after: Predicted mask after TTT (H, W)
        save_path: Directory to save visualization
        prefix: Filename prefix
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Get image shapes
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    # Ensure masks match corresponding image shapes; use reshape_img_war
    if mask1.shape[:2] != (h1, w1):
        print(f"Warning: mask1 shape {mask1.shape[:2]} does not match image1 shape {(h1, w1)}. Resizing.")
        mask1 = reshape_img_war(mask1, (w1, h1))
    
    if pred_mask_before.shape[:2] != (h2, w2):
        print(f"Warning: pred_mask_before shape {pred_mask_before.shape[:2]} does not match image2 shape {(h2, w2)}. Resizing.")
        pred_mask_before = reshape_img_war(pred_mask_before, (w2, h2))
    
    if pred_mask_after.shape[:2] != (h2, w2):
        print(f"Warning: pred_mask_after shape {pred_mask_after.shape[:2]} does not match image2 shape {(h2, w2)}. Resizing.")
        pred_mask_after = reshape_img_war(pred_mask_after, (w2, h2))
    
    # Original images
    cv2.imwrite(f"{save_path}/{prefix}_img1.jpg", image1)
    cv2.imwrite(f"{save_path}/{prefix}_img2.jpg", image2)
    
    # Image1 with mask - overlay_mask_on_image also checks shapes internally
    img1_with_mask = overlay_mask_on_image(image1/255.0, mask1, color=(0, 1, 0)) * 255
    cv2.imwrite(f"{save_path}/{prefix}_img1_with_mask.jpg", img1_with_mask.astype(np.uint8))
    
    # Image2 with predicted mask before TTT
    img2_with_mask_before = overlay_mask_on_image(image2/255.0, pred_mask_before, color=(0, 0, 1)) * 255
    cv2.imwrite(f"{save_path}/{prefix}_img2_pred_before.jpg", img2_with_mask_before.astype(np.uint8))
    
    # Image2 with predicted mask after TTT
    img2_with_mask_after = overlay_mask_on_image(image2/255.0, pred_mask_after, color=(0, 0, 1)) * 255
    cv2.imwrite(f"{save_path}/{prefix}_img2_pred_after.jpg", img2_with_mask_after.astype(np.uint8))

def compute_iou(mask1, mask2):
    """Compute IoU between two binary masks"""
    mask1_bin = mask1 > 0.5
    mask2_bin = mask2 > 0.5
    
    intersection = np.logical_and(mask1_bin, mask2_bin).sum()
    union = np.logical_or(mask1_bin, mask2_bin).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union

def reshape_img_war(img, size):
    C = 1
    if len(img.shape) == 2:
        H, W = img.shape
        img = img[..., None]
    else:
        H, W, C = img.shape
    
    temp = np.zeros((max(H, W), max(H, W), C), dtype=img.dtype)

    if H > W:
        L = (H - W) // 2
        temp[:, L:-L] = img
    elif W > H:
        L = (W - H) // 2
        temp[L:-L] = img
    else:
        temp = img

    if img.dtype == np.uint8:
        temp = cv2.resize(temp, size, interpolation=cv2.INTER_NEAREST)
    else:
        temp = cv2.resize(temp, size, interpolation=cv2.INTER_LINEAR)

    return temp

def get_model(model_path, backbone_size='large', backbone_type='dinov2', image_size=518, extractor_type='cn2_base', device=None):

    if device is None:
        device = torch.device('cuda')

    netEncoder = ConditionalSegmentationModel(
        feat_extractor=extractor_type,  
        extractor_depth=2, 
        backbone_size=backbone_size,
        image_size=image_size,
        upsampler='bilinear',
        backbone_type=backbone_type,
        num_register_tokens=4
    )

    print('Loading net weight from {}'.format(model_path))
    # Avoid loading weights onto the default cuda:0:
    # first load to CPU, then move to the current rank's device.
    param = torch.load(model_path, map_location='cpu', weights_only=False)
    netEncoder.load_state_dict(param['encoder'], strict=True)
    del param
    netEncoder.eval()
    netEncoder.to(device)

    return netEncoder 

def get_tensors(I1np, I2np, M1np, device):
    I1 = I1np
    I2 = I2np

    norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transformINet = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])
    tensor1 = transformINet(I1).unsqueeze(0).to(device, non_blocking=True)
    tensor2 = transformINet(I2).unsqueeze(0).to(device, non_blocking=True)
    tensor3 = torch.from_numpy(M1np).unsqueeze(0).unsqueeze(0).float().to(device, non_blocking=True)

    return tensor1, tensor2, tensor3

def forward_pass(netEncoder, tensor1, tensor2, tensor3, image_size, use_amp=False):
    with torch.no_grad():
        with autocast(enabled=use_amp):
            output, confidence = netEncoder(tensor1, tensor3, tensor2)
            output = torch.sigmoid(output[-1])
            confidence = torch.sigmoid(confidence[-1]).item()
        
        m2_final = (output > MASKThresh).float().cpu().numpy()[0, 0]
        if image_size > 480:
            m2_final = cv2.resize(m2_final, (480, 480), interpolation=cv2.INTER_NEAREST)

    return m2_final, confidence

def test_time_training(netEncoder, tensor1, tensor2, tensor3, lr=0.0001, iterations=5, use_amp=False, ttt_layers=2, dice_weight=0.0):

    trainable_blocks = netEncoder.backbone.blocks[-ttt_layers:]
    # Back up initial weights to CPU to avoid duplicated GPU memory usage
    initial_states = {}
    for i, block in enumerate(trainable_blocks):
        state_cpu = {k: v.detach().cpu() for k, v in block.state_dict().items()}
        initial_states[i] = state_cpu
    
    trainable_blocks.train()
    optimizer = optim.AdamW(trainable_blocks.parameters(), lr=lr, weight_decay=1e-3)
    scaler = GradScaler(enabled=use_amp)
    
    for _ in range(iterations):
        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            output_list, _ = netEncoder(tensor1, tensor3, tensor2)
            FM2 = torch.sigmoid(output_list[-1])
            Reversed_output_list, _ = netEncoder(tensor2, FM2, tensor1)
            loss = torch.nn.BCEWithLogitsLoss()(Reversed_output_list[-1], tensor3)
            
            if abs(dice_weight) > 1e-6:
                dice_loss = losses.dice_loss_with_logits(Reversed_output_list[-1], tensor3).mean()
                loss += dice_weight * dice_loss
   
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    netEncoder.eval()
    
    return netEncoder, initial_states, trainable_blocks

def load_frame(path, frame_idx, image_size=518, ret_size=False, ret_original=False):
    img_path = os.path.join(path, f'{frame_idx}.jpg')
    img_original = cv2.imread(img_path)
    img = img_original[..., ::-1]  # BGR to RGB
    orig_size = img.shape[:-1]
    img_resized = reshape_img_war(img, (image_size, image_size))
    img_pil = Image.fromarray(img_resized)
    
    if ret_original and ret_size:
        return img_pil, orig_size, img_original
    elif ret_original:
        return img_pil, img_original
    elif ret_size:
        return img_pil, orig_size
    return img_pil

def egoexo(netEncoder, annotations, ego, exo, obj, take, anno_path, pred_json, image_size, device, vis_path=None, iou_threshold=0.1, 
          ttt_enable=False, ttt_lr=0.0001, ttt_iter=5, ttt_layers=2, dice_weight=0.0, use_amp=False, max_iou_improvement=None):

    pred_json['masks'][obj][f'{ego}_{exo}'] = {}
    os.makedirs(vis_path, exist_ok=True) if vis_path else None
    
    pair_count = 0
    total_time = 0.0
    
    for idx in annotations['masks'][obj][ego].keys():
        pair_start_time = time.time()
        # Check whether GT exists for the target frame
        has_gt = False
        if idx in annotations['masks'][obj].get(exo, {}):
            has_gt = True

        # Load source (ego) frame and mask
        ego_frame, ego_original = load_frame(path=f'{anno_path}/{take}/{ego}/', frame_idx=idx, image_size=image_size, ret_original=True)
        ego_mask = mask_utils.decode(annotations['masks'][obj][ego][idx])
        ego_mask_resized = reshape_img_war(ego_mask, (image_size, image_size))

        # Load target (exo) frame
        exo_frame, exo_size, exo_original = load_frame(path=f'{anno_path}/{take}/{exo}/', frame_idx=idx, image_size=image_size, ret_size=True, ret_original=True)

        # If GT exists, load GT mask
        gt_mask = None
        if has_gt:
            gt_mask = mask_utils.decode(annotations['masks'][obj][exo][idx])
            # 480x480 for visualization / IoU comparison
            gt_mask_480 = reshape_img_war(gt_mask, (480, 480))

        # Convert to tensors
        tensor1, tensor2, tensor3 = get_tensors(ego_frame, exo_frame, ego_mask_resized, device)
        
        # Forward pass before TTT
        m2_bin_before, confidence_before = forward_pass(netEncoder, tensor1, tensor2, tensor3, image_size, use_amp)
        
        # Compute IoU
        iou_before = None
        if has_gt:
            m2_before_480 = reshape_img_war(m2_bin_before, (480, 480))
            iou_before = compute_iou(gt_mask_480, m2_before_480)
        
        # Apply test-time training if enabled
        initial_states = None
        trainable_blocks = None
        
        if ttt_enable:
            netEncoder, initial_states, trainable_blocks = test_time_training(
                netEncoder, tensor1, tensor2, tensor3,
                lr=ttt_lr, iterations=ttt_iter, use_amp=use_amp, ttt_layers=ttt_layers, dice_weight=dice_weight
            )
            
        # Forward pass after TTT
        m2_bin_after, confidence_after = forward_pass(netEncoder, tensor1, tensor2, tensor3, image_size, use_amp)

        # Reset model state if TTT was applied
        if ttt_enable and initial_states is not None and trainable_blocks is not None:
            for i, block in enumerate(trainable_blocks):
                block.load_state_dict(initial_states[i])
            netEncoder.eval()

        # Prepare mask for result recording - remove padding back to original resolution
        y_step = (m2_bin_after > MASKThresh)
        y_step = utils.remove_pad(y_step, orig_size=exo_size)
        exo_pred = mask_utils.encode(np.asfortranarray(y_step.astype(np.uint8)))
        exo_pred['counts'] = exo_pred['counts'].decode('ascii')
        pred_json['masks'][obj][f'{ego}_{exo}'][idx] = {'pred_mask': exo_pred, 'confidence': confidence_after}
        
        # Compute IoU after TTT and compare with GT
        if has_gt and ttt_enable and vis_path:
            m2_after_480 = reshape_img_war(m2_bin_after, (480, 480))
            iou_after = compute_iou(gt_mask_480, m2_after_480)
            iou_improvement = iou_after - iou_before
            
                # Save visualization only if IoU change exceeds threshold and
                # is better than the current best improvement for this take.
            if (iou_improvement > iou_threshold or iou_improvement < -iou_threshold) and (max_iou_improvement is None or iou_improvement > max_iou_improvement[take]):
                # Update best IoU improvement for this take
                if max_iou_improvement is not None:
                    max_iou_improvement[take] = iou_improvement
                
                # Resize to 480x480 for visualization
                ego_original_480 = reshape_img_war(ego_original, (480, 480))
                exo_original_480 = reshape_img_war(exo_original, (480, 480))
                ego_mask_480 = reshape_img_war(ego_mask, (480, 480))
                
                # Create visualization at 480x480
                exo_with_gt = overlay_mask_on_image(exo_original_480/255.0, gt_mask_480, color=(0, 1, 0)) * 255
                cv2.imwrite(f"{vis_path}/{take}_{ego}_{exo}_{idx}_img2_with_gt.jpg", exo_with_gt.astype(np.uint8))
                
                save_visualization(
                    ego_original_480, exo_original_480, 
                    ego_mask_480, m2_before_480, m2_after_480,
                    vis_path, f"{take}_{ego}_{exo}_{idx}_iou_{iou_before:.3f}_to_{iou_after:.3f}"
                )
                print(f"NEW BEST! take: {take}, exo: {exo}, ego: {ego}, idx: {idx}, iou before: {iou_before:.3f}, iou after: {iou_after:.3f}, improvement: {iou_improvement:.3f}")
        
        # Record per-pair inference time
        pair_end_time = time.time()
        total_time += (pair_end_time - pair_start_time)
        pair_count += 1
    
    return pair_count, total_time

def exoego(netEncoder, annotations, ego, exo, obj, take, anno_path, pred_json, image_size, device, vis_path=None, iou_threshold=0.1, 
          ttt_enable=False, ttt_lr=0.0001, ttt_iter=5, ttt_layers=2, dice_weight=0.0, use_amp=False, max_iou_improvement=None):

    pred_json['masks'][obj][f'{exo}_{ego}'] = {}
    os.makedirs(vis_path, exist_ok=True) if vis_path else None
    
    pair_count = 0
    total_time = 0.0
    
    for idx in annotations['masks'][obj][exo].keys():
        pair_start_time = time.time()
        # Check whether GT exists for the target frame
        has_gt = False
        if idx in annotations['masks'][obj].get(ego, {}):
            has_gt = True

        # Load source (exo) frame and mask
        exo_frame, exo_original = load_frame(path=f'{anno_path}/{take}/{exo}/', frame_idx=idx, image_size=image_size, ret_original=True)
        exo_mask = mask_utils.decode(annotations['masks'][obj][exo][idx])
        exo_mask_resized = reshape_img_war(exo_mask, (image_size, image_size))

        # Load target (ego) frame
        ego_frame, ego_size, ego_original = load_frame(path=f'{anno_path}/{take}/{ego}/', frame_idx=idx, image_size=image_size, ret_size=True, ret_original=True)

        # If GT exists, load GT mask
        gt_mask = None
        if has_gt:
            gt_mask = mask_utils.decode(annotations['masks'][obj][ego][idx])
            # 480x480 for visualization / IoU comparison
            gt_mask_480 = reshape_img_war(gt_mask, (480, 480))

        # Convert to tensors
        tensor1, tensor2, tensor3 = get_tensors(exo_frame, ego_frame, exo_mask_resized, device)
        
        # Forward pass before TTT
        m2_bin_before, confidence_before = forward_pass(netEncoder, tensor1, tensor2, tensor3, image_size, use_amp)
        
        # Compute IoU
        iou_before = None
        if has_gt:
            m2_before_480 = reshape_img_war(m2_bin_before, (480, 480))
            iou_before = compute_iou(gt_mask_480, m2_before_480)
        
        # Apply test-time training if enabled
        initial_states = None
        trainable_blocks = None
        
        if ttt_enable:
            netEncoder, initial_states, trainable_blocks = test_time_training(
                netEncoder, tensor1, tensor2, tensor3,
                lr=ttt_lr, iterations=ttt_iter, use_amp=use_amp, ttt_layers=ttt_layers, dice_weight=dice_weight
            )
            
        # Forward pass after TTT
        m2_bin_after, confidence_after = forward_pass(netEncoder, tensor1, tensor2, tensor3, image_size, use_amp)

        # Reset model state if TTT was applied
        if ttt_enable and initial_states is not None and trainable_blocks is not None:
            for i, block in enumerate(trainable_blocks):
                block.load_state_dict(initial_states[i])
            netEncoder.eval()

        # Prepare mask for result recording - remove padding back to original resolution
        y_step = (m2_bin_after > MASKThresh)
        y_step = utils.remove_pad(y_step, orig_size=ego_size)
        ego_pred = mask_utils.encode(np.asfortranarray(y_step.astype(np.uint8)))
        ego_pred['counts'] = ego_pred['counts'].decode('ascii')
        pred_json['masks'][obj][f'{exo}_{ego}'][idx] = {'pred_mask': ego_pred, 'confidence': confidence_after}
        
        # Compute IoU after TTT and compare with GT
        if has_gt and ttt_enable and vis_path:
            m2_after_480 = reshape_img_war(m2_bin_after, (480, 480))
            iou_after = compute_iou(gt_mask_480, m2_after_480)
            iou_improvement = iou_after - iou_before
            
            # Save visualization only if IoU change exceeds threshold and
            # is better than the current best improvement for this take.
            if (iou_improvement > iou_threshold or iou_improvement < -iou_threshold) and (max_iou_improvement is None or iou_improvement > max_iou_improvement[take]):
                # Update best IoU improvement for this take
                if max_iou_improvement is not None:
                    max_iou_improvement[take] = iou_improvement
                
                # Resize to 480x480 for visualization
                exo_original_480 = reshape_img_war(exo_original, (480, 480))
                ego_original_480 = reshape_img_war(ego_original, (480, 480))
                exo_mask_480 = reshape_img_war(exo_mask, (480, 480))
                
                # Create visualization at 480x480
                ego_with_gt = overlay_mask_on_image(ego_original_480/255.0, gt_mask_480, color=(0, 1, 0)) * 255
                cv2.imwrite(f"{vis_path}/{take}_{exo}_{ego}_{idx}_img2_with_gt.jpg", ego_with_gt.astype(np.uint8))
                
                save_visualization(
                    exo_original_480, ego_original_480, 
                    exo_mask_480, m2_before_480, m2_after_480,
                    vis_path, f"{take}_{exo}_{ego}_{idx}_iou_{iou_before:.3f}_to_{iou_after:.3f}"
                )
                print(f"NEW BEST! take: {take}, exo: {exo}, ego: {ego}, idx: {idx}, iou before: {iou_before:.3f}, iou after: {iou_after:.3f}, improvement: {iou_improvement:.3f}")
        
        # Record per-pair inference time
        pair_end_time = time.time()
        total_time += (pair_end_time - pair_start_time)
        pair_count += 1
    
    return pair_count, total_time

def main(model_path, takes, anno_path, out_path, setting='ego-exo', save_inter=False, image_size=518, use_amp=False,
        ttt_enable=False, ttt_lr=0.0001, ttt_iter=5, ttt_layers=2, iou_threshold=0.1, backbone_size='large', backbone_type='dinov2',
        extractor_type='cn2_base', device=None, dice_weight=0.0, save_vis=False,
        rank=0, world_size=1):
    
    is_distributed = dist.is_available() and dist.is_initialized() and world_size > 1
    if rank == 0:
        print('TOTAL TAKES: ', len(takes))

    # set device per-rank
    device = torch.device('cuda')
    if torch.cuda.is_available():
        if is_distributed:
            local_rank = int(os.environ.get('LOCAL_RANK', str(rank)))
            num_visible = torch.cuda.device_count()
            if num_visible == 0:
                raise RuntimeError('CUDA is available but no GPUs are visible. Check CUDA_VISIBLE_DEVICES.')
            if local_rank >= num_visible:
                raise RuntimeError(f'LOCAL_RANK {local_rank} >= visible GPU count {num_visible}.\n'
                                   f'Fix by setting CUDA_VISIBLE_DEVICES to at least {local_rank+1} GPUs, '
                                   f'or reduce --nproc_per_node to {num_visible}.')
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cuda')

    netEncoder = get_model(model_path, backbone_size=backbone_size, backbone_type=backbone_type, image_size=image_size, extractor_type=extractor_type, device=device)
    
    # Only create visualization directory when TTT is enabled and save_vis is True
    if ttt_enable and save_vis:
        vis_path = f"{out_path}/vis_ttt_compare_{setting}"
        os.makedirs(vis_path, exist_ok=True)
        if rank == 0:
            print(f"Visualization will be saved to {vis_path}")
    else:
        vis_path = None
        if rank == 0 and ttt_enable and not save_vis:
            print("Visualization is disabled (use --save_vis to enable)")
    
    # Shard takes by rank if distributed
    def shard_list_by_rank(items, rank, world_size):
        if world_size <= 1:
            return items
        return items[rank::world_size]

    assigned_takes = shard_list_by_rank(takes, rank, world_size) if is_distributed else takes

    results = {}
    # Track best IoU improvement per take
    max_iou_improvements = {}
    
    # Track inference time and pair count
    total_pairs = 0
    total_inference_time = 0.0
    
    for take in tqdm.tqdm(assigned_takes, disable=(rank != 0)):

        with open(f'{anno_path}/{take}/annotation.json', 'r') as fp:
            annotations = json.load(fp)

        pred_json = {'masks': {}, 'subsample_idx': annotations['subsample_idx']}
        # Initialize best IoU improvement record for this take
        max_iou_improvements[take] = 0.0

        for obj in annotations['masks']:

            pred_json['masks'][obj] = {}

            cams = annotations['masks'][obj].keys()

            exo_cams = [x for x in cams if 'aria' not in x]
            ego_cams = [x for x in cams if 'aria' in x]

            for ego in ego_cams:
                for exo in exo_cams:
                    # ego -> exo
                    if setting == 'ego-exo':
                        pair_count, pair_time = egoexo(netEncoder=netEncoder, annotations=annotations,
                              ego=ego, exo=exo, obj=obj, take=take, anno_path=anno_path, pred_json=pred_json, 
                              image_size=image_size, device=device, vis_path=vis_path, iou_threshold=iou_threshold,
                              ttt_enable=ttt_enable, ttt_lr=ttt_lr, 
                              ttt_iter=ttt_iter, ttt_layers=ttt_layers, dice_weight=dice_weight, use_amp=use_amp,
                              max_iou_improvement=max_iou_improvements)
                        total_pairs += pair_count
                        total_inference_time += pair_time
                    elif setting == 'exo-ego':
                        pair_count, pair_time = exoego(netEncoder=netEncoder, annotations=annotations,
                              ego=ego, exo=exo, obj=obj, take=take, anno_path=anno_path, pred_json=pred_json, 
                              image_size=image_size, device=device, vis_path=vis_path, iou_threshold=iou_threshold,
                              ttt_enable=ttt_enable, ttt_lr=ttt_lr, 
                              ttt_iter=ttt_iter, ttt_layers=ttt_layers, dice_weight=dice_weight, use_amp=use_amp,
                              max_iou_improvement=max_iou_improvements)
                        total_pairs += pair_count
                        total_inference_time += pair_time
                    else:
                        raise Exception(f"Setting {setting} not recognized.")

        results[take] = pred_json

        if save_inter:
            os.makedirs(f'{out_path}/{take}', exist_ok=True)
            with open(f'{out_path}/{take}/pred_annotations.json', 'w') as fp:
                json.dump(pred_json, fp)

    # Compute average inference time
    avg_time_per_pair = total_inference_time / total_pairs if total_pairs > 0 else 0.0
    
    # Print statistics
    if rank == 0:
        print("\n" + "="*60)
        print(f"Inference statistics (Rank {rank}):")
        print(f"  Total pairs: {total_pairs}")
        print(f"  Total inference time: {total_inference_time:.2f} s")
        print(f"  Avg time per pair: {avg_time_per_pair:.4f} s")
        print("="*60 + "\n")
    
    return results, {'total_pairs': total_pairs, 'total_time': total_inference_time, 'avg_time': avg_time_per_pair}

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--splits_path', type=str, required=True, help="Path to json of take splits")
    parser.add_argument('--split', type=str, required=True, help="Split to evaluate on")
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--setting', required=True, choices=['ego-exo', 'exo-ego'], help="ego-exo or exo-ego")
    parser.add_argument('--save_inter', action='store_true', help="Store intermediate take wise results")
    parser.add_argument('--use_amp', action='store_true', help="Use AMP")
    parser.add_argument('--image_size', type=int, default=518, help="Image size")
    
    parser.add_argument('--ttt_enable', action='store_true', help="Enable test-time training")
    parser.add_argument('--ttt_lr', type=float, default=5e-6, help="Learning rate for test-time training")
    parser.add_argument('--ttt_iter', type=int, default=2, help="Number of iterations for test-time training")
    parser.add_argument('--ttt_layers', type=int, default=4, help="Number of backbone layers to fine-tune during test-time training")
    parser.add_argument('--iou_threshold', type=float, default=0.3, help="IoU difference threshold for visualization")
    parser.add_argument('--dice_weight', type=float, default=0.0, help="Weight for dice loss in test-time training")
    parser.add_argument('--save_vis', action='store_true', help="Save visualization results (only works when ttt_enable is True)")

    parser.add_argument('--backbone_size', type=str, default='large', help="Backbone size (e.g. 'large', 'base')")
    parser.add_argument('--backbone_type', type=str, default='dinov2', help="Backbone type")
    parser.add_argument('--extractor_type', type=str, default='cn2_base', help="Extractor type")
    parser.add_argument('--distributed', action='store_true', help="Use DistributedDataParallel with torchrun")
    parser.add_argument('--dist_backend', type=str, default='nccl', help="DDP backend")

    args = parser.parse_args()
    print(args)

    # distributed init (torchrun recommended)
    rank = 0
    world_size = 1
    if args.distributed or int(os.environ.get('WORLD_SIZE', '1')) > 1:
        if not dist.is_initialized():
            # Increase timeout to 180 minutes
            import datetime
            timeout = datetime.timedelta(minutes=180)
            dist.init_process_group(backend=args.dist_backend, init_method='env://', timeout=timeout)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        # ensure no stale process group
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    with open(args.splits_path, "r") as fp:
        splits = json.load(fp)
    
    results_local, stats_local = main(args.ckpt_path,
                splits[args.split],
                args.data_path,
                args.out_path,
                setting=args.setting,
                save_inter=args.save_inter,
                image_size=args.image_size,
                use_amp=args.use_amp,
                ttt_enable=args.ttt_enable,
                ttt_lr=args.ttt_lr,
                ttt_iter=args.ttt_iter,
                ttt_layers=args.ttt_layers,
                iou_threshold=args.iou_threshold,
                backbone_size=args.backbone_size,
                backbone_type=args.backbone_type,
                extractor_type=args.extractor_type,
                dice_weight=args.dice_weight,
                save_vis=args.save_vis,
                rank=rank,
                world_size=world_size
                )

    # gather results on rank0
    results = None
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        obj_list = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(results_local, obj_list, dst=0)
        
        # gather stats
        stats_list = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(stats_local, stats_list, dst=0)
        
        if rank == 0:
            results = {}
            for part in obj_list:
                if part is not None:
                    results.update(part)
            
            # Aggregate stats across all ranks
            total_pairs_all = sum(s['total_pairs'] for s in stats_list if s is not None)
            total_time_all = sum(s['total_time'] for s in stats_list if s is not None)
            avg_time_all = total_time_all / total_pairs_all if total_pairs_all > 0 else 0.0
            
            print("\n" + "="*60)
            print("Global inference statistics (aggregated across all ranks):")
            print(f"  Total pairs: {total_pairs_all}")
            print(f"  Total inference time: {total_time_all:.2f} s")
            print(f"  Avg time per pair: {avg_time_all:.4f} s")
            print("="*60 + "\n")
    else:
        results = results_local

    # cleanup
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

"""
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29501 eval_segswap_visttt.py \
    --ckpt_path train/output/1102_dinov3cnlarge_dinov3large_dice5_bs16as16ep200_mxlr1e5_lp20lr1e4/best_test_miou.pth \
    --data_path true_data \
    --splits_path data/split.json \
    --split test \
    --out_path output/1102_dinov3cnlarge_dinov3large_dice5_bs16as16ep200_mxlr1e5_lp20lr1e4 \
    --setting ego-exo \
    --image_size 512 \
    --backbone_type dinov3 \
    --extractor_type dinov3_cn_large \
    --distributed \
    --ttt_enable \
    --ttt_iter 2 \
    --ttt_layers 4 \
    --use_amp

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29501 eval_segswap_visttt.py \
    --ckpt_path train/output/1102_dinov3cnlarge_dinov3large_dice5_bs16as16ep200_mxlr1e5_lp20lr1e4/best_test_miou.pth \
    --data_path true_data \
    --splits_path data/split.json \
    --split test \
    --out_path output/1102_dinov3cnlarge_dinov3large_dice5_bs16as16ep200_mxlr1e5_lp20lr1e4 \
    --setting exo-ego \
    --image_size 512 \
    --backbone_type dinov3 \
    --extractor_type dinov3_cn_large \
    --distributed \
    --ttt_enable \
    --ttt_iter 6 \
    --ttt_layers 11 \
    --use_amp
"""
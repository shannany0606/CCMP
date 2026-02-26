import torch
import os
from enum import Enum
from tqdm import tqdm
import numpy as np
import sys
import eval_utils
sys.path.append('model/')
from train.csegmentor import ConditionalSegmentationModel
import cv2
from torch.utils.data import Dataset, DataLoader
from pycocotools.mask import encode, decode, frPyObjects
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
import torch.distributed as dist
import transformers
from pathlib import Path
import pickle
import math
import json
import os 
import re
from natsort import natsorted
from PIL import Image
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
import train.losses as losses

# Helper functions for SegSwap model
MASKThresh = 0.5

# global distributed states
RANK = 0
WORLD_SIZE = 1
IS_DISTRIBUTED = False

def reshape_img_war(img, size):
    """Reshape image with padding to maintain aspect ratio"""
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
    
    # Ensure mask and image sizes match
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # If mask is all zeros, return original image
    if mask.sum() == 0:
        return image.copy()

    masked = mask > 0.5
    overlay = image.copy()
    
    overlay[masked] = overlay[masked] * (1 - alpha) + np.array(color) * alpha
    
    return overlay

def compute_iou(mask1, mask2):
    """Compute IoU between two binary masks"""
    mask1_bin = mask1 > 0.5
    mask2_bin = mask2 > 0.5
    intersection = np.logical_and(mask1_bin, mask2_bin).sum()
    union = np.logical_or(mask1_bin, mask2_bin).sum()
    if union == 0:
        return 0.0
    return intersection / union

def save_ttt_visualization(
    src_img_bgr,
    tgt_img_bgr,
    cond_mask,
    pred_before,
    pred_after,
    gt_mask,
    save_dir,
    prefix
):
    """
    Save TTT visualizations:
      1) Source image + conditioning mask (green)
      2) Target image + prediction before TTT (blue)
      3) Target image + prediction after TTT (blue)
      4) Target image + ground truth (green)
    """
    os.makedirs(save_dir, exist_ok=True)
    h1, w1 = src_img_bgr.shape[:2]
    h2, w2 = tgt_img_bgr.shape[:2]

    # Align shapes
    if cond_mask.shape[:2] != (h1, w1):
        cond_mask_vis = cv2.resize(cond_mask, (w1, h1), interpolation=cv2.INTER_NEAREST)
    else:
        cond_mask_vis = cond_mask

    if pred_before.shape[:2] != (h2, w2):
        pred_before_vis = cv2.resize(pred_before, (w2, h2), interpolation=cv2.INTER_NEAREST)
    else:
        pred_before_vis = pred_before

    if pred_after.shape[:2] != (h2, w2):
        pred_after_vis = cv2.resize(pred_after, (w2, h2), interpolation=cv2.INTER_NEAREST)
    else:
        pred_after_vis = pred_after

    if gt_mask is not None:
        if gt_mask.shape[:2] != (h2, w2):
            gt_vis = cv2.resize(gt_mask, (w2, h2), interpolation=cv2.INTER_NEAREST)
        else:
            gt_vis = gt_mask
    else:
        gt_vis = None

    img1_with_mask = overlay_mask_on_image(src_img_bgr.astype(np.float32) / 255.0, cond_mask_vis, color=(0, 1, 0), alpha=0.5) * 255
    img2_with_pred_before = overlay_mask_on_image(tgt_img_bgr.astype(np.float32) / 255.0, pred_before_vis, color=(0, 0, 1), alpha=0.5) * 255
    img2_with_pred_after = overlay_mask_on_image(tgt_img_bgr.astype(np.float32) / 255.0, pred_after_vis, color=(0, 0, 1), alpha=0.5) * 255
    cv2.imwrite(f"{save_dir}/{prefix}_source_with_mask.jpg", img1_with_mask.astype(np.uint8))
    cv2.imwrite(f"{save_dir}/{prefix}_target_pred_before.jpg", img2_with_pred_before.astype(np.uint8))
    cv2.imwrite(f"{save_dir}/{prefix}_target_pred_after.jpg", img2_with_pred_after.astype(np.uint8))
    if gt_vis is not None:
        img2_with_gt = overlay_mask_on_image(tgt_img_bgr.astype(np.float32) / 255.0, gt_vis, color=(0, 1, 0), alpha=0.5) * 255
        cv2.imwrite(f"{save_dir}/{prefix}_target_with_gt.jpg", img2_with_gt.astype(np.uint8))

def get_tensors(I1np, I2np, M1np):
    """Convert images and mask to tensors"""
    norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transformINet = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])
    tensor1 = transformINet(I1np).unsqueeze(0).cuda()
    tensor2 = transformINet(I2np).unsqueeze(0).cuda()
    tensor3 = torch.from_numpy(M1np).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor).cuda()
    return tensor1, tensor2, tensor3

# fuse mask
def fuse_mask(mask_list,fill_number_list):
    fused_mask = np.zeros_like(mask_list[0])
    for mask, fill_number in zip(mask_list,fill_number_list):
        fill_number = int(fill_number)
        fused_mask[mask != 0] = fill_number 
    return fused_mask

# metric calculation
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def get_center(mask,h,w):
    y_coords, x_coords = np.where(mask == 1)
    if len(y_coords) == 0 or len(x_coords) == 0:
        return 0.5, 0.5
    
    centroid_y = int(np.mean(y_coords))
    centroid_x = int(np.mean(x_coords))
    centroid_y = centroid_y / h
    centroid_x = centroid_x / w
    return centroid_y, centroid_x

def get_distance(x1,y1,x2,y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def iou(mask1,mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def compute_metric(le_meter,intersection_meter,union_meter,acc_iou_meter,results_list,thr=0.5,topk=3,vis=False):
    pred_list = []
    gt_list = []
    results_list = list(results_list)
    tot = 0
    cor = 0
    for results in results_list:
        gt = results['gt']
        preds = results['pred']
        scores = results['scores']
        preds = preds.astype(np.uint8)
        _,idx = torch.topk(torch.tensor(scores),topk)
        idx = idx.cpu().numpy()
        topk_preds = preds[idx,:]
        max_acc_iou = -1
        max_iou = 0
        max_intersection = 0
        max_union = 0
        max_i = 0
        for i,pred_ in enumerate(topk_preds):
            h,w = pred_.shape[:2]
            pred_y, pred_x = get_center(pred_,h,w)
            gt_y, gt_x = get_center(gt,h,w)
            loc_err = get_distance(pred_x,pred_y,gt_x,gt_y)
            le_meter.update(loc_err)
            intersection, union, _ = intersectionAndUnionGPU(
                torch.tensor(pred_).int().cuda().contiguous().clone(), torch.tensor(gt).int().cuda().contiguous(), 2, ignore_index=255
            )
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = intersection / (union + 1e-5)
            acc_iou[union == 0] = 1.0  # no-object target
            fore_acc_iou = acc_iou[1]
            if fore_acc_iou > max_acc_iou:
                max_acc_iou = fore_acc_iou
                max_iou = acc_iou
                max_intersection = intersection
                max_union = union
                max_i = i
        intersection_meter.update(max_intersection)
        union_meter.update(max_union)
        acc_iou_meter.update(max_iou, n=1)
        pred_list.append(topk_preds[max_i])
        gt_list.append(gt)

        fg_iou = acc_iou[1]
        if fg_iou > 0.5:
            cor += 1
            tot += 1
        else:
            tot += 1

    return pred_list,gt_list, cor, tot

def load_segswap_model(model_path, image_size=512, backbone_size='large', backbone_type='dinov3', 
                        extractor_type='dinov3_cn_large', device='cuda'):
    """Load SegSwap ConditionalSegmentationModel"""
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
    param = torch.load(model_path, weights_only=False, map_location='cpu')
    netEncoder.load_state_dict(param['encoder'], strict=True)
    netEncoder.eval()
    netEncoder.to(device)
    
    return netEncoder

# latest checkpoint path
def get_latest_checkpoint_path(model_path):
    checkpoint_pattern = re.compile(r"checkpoint-(\d+)")
    if os.path.basename(model_path).startswith("checkpoint-") and checkpoint_pattern.match(os.path.basename(model_path)):
        return model_path  
    elif os.path.isdir(model_path):
        checkpoints = [d for d in os.listdir(model_path) if checkpoint_pattern.match(d)]
        if not checkpoints:
            raise ValueError("No checkpoints found in the specified directory.")
        max_checkpoint = max(checkpoints, key=lambda x: int(checkpoint_pattern.match(x).group(1)))
        model_path = os.path.join(model_path, max_checkpoint)
    elif not os.path.exists(model_path):
        raise FileNotFoundError(f"The specified path '{model_path}' does not exist.")
    return model_path

# -------------------------
# Test-Time Training (TTT)
# -------------------------
def test_time_training(net_encoder, tensor1, tensor2, tensor3, lr=5e-6, iterations=3, ttt_layers=3, dice_weight=0.0, use_amp=False):
    """
    Perform small-step optimization for the current sample at test time,
    fine-tuning only the last `ttt_layers` blocks of the backbone.
    Returns a snapshot of initial states and the trainable blocks for restoration.
    """
    # Select the last few layers as trainable blocks
    trainable_blocks = net_encoder.backbone.blocks[-ttt_layers:]
    # Back up initial weights to CPU to avoid extra GPU memory usage
    initial_states = {}
    for i, block in enumerate(trainable_blocks):
        state_cpu = {k: v.detach().cpu() for k, v in block.state_dict().items()}
        initial_states[i] = state_cpu

    # Set training mode and optimizer
    trainable_blocks.train()
    optimizer = optim.AdamW(trainable_blocks.parameters(), lr=lr, weight_decay=1e-3)
    scaler = GradScaler(enabled=use_amp)

    for _ in range(iterations):
        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            output_list, _ = net_encoder(tensor1, tensor3, tensor2)
            fm2 = torch.sigmoid(output_list[-1])
            reversed_output_list, _ = net_encoder(tensor2, fm2, tensor1)
            loss = torch.nn.BCEWithLogitsLoss()(reversed_output_list[-1], tensor3)
            if abs(dice_weight) > 1e-6:
                dice_loss = losses.dice_loss_with_logits(reversed_output_list[-1], tensor3).mean()
                loss = loss + dice_weight * dice_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    net_encoder.eval()
    return initial_states, trainable_blocks

# hyperparameters
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--json_path', type=str, required=True, help='Path to the JSON file with dataset')
parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
parser.add_argument('--root_path', type=str, required=True, help='Root path to the dataset images')
parser.add_argument('--image_size', type=int, default=512, help='Image size for model input')
parser.add_argument('--backbone_size', type=str, default='large', help='Backbone size (e.g., large, base)')
parser.add_argument('--backbone_type', type=str, default='dinov3', help='Backbone type (e.g., dinov3, dinov2)')
parser.add_argument('--extractor_type', type=str, default='dinov3_cn_large', help='Feature extractor type')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
parser.add_argument('--topk', type=int, default=3, help='Top-k predictions to consider')
parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
parser.add_argument('--dist', action='store_true', help='Enable distributed evaluation (use with torchrun)')
parser.add_argument('--dist_backend', type=str, default='nccl', help='Distributed backend')
parser.add_argument('--dist_url', type=str, default='env://', help='Init method for distributed run')
parser.add_argument('--local_rank', type=int, default=-1, help='Local rank passed by torch.distributed/torchrun')
parser.add_argument('--vis_path', type=str, default=None, help='Path to save visualization results')
parser.add_argument('--vis_num', type=int, default=100, help='Number of samples to visualize')
parser.add_argument('--ttt_enable', action='store_true', help='Enable test-time training')
parser.add_argument('--ttt_lr', type=float, default=5e-6, help='Learning rate for test-time training')
parser.add_argument('--ttt_iter', type=int, default=6, help='Iterations for test-time training')
parser.add_argument('--ttt_layers', type=int, default=6, help='Number of backbone layers to fine-tune')
parser.add_argument('--dice_weight', type=float, default=0.0, help='Weight for dice loss in TTT')
parser.add_argument('--iou_threshold', type=float, default=0.05, help='IoU difference threshold for TTT visualization')
args = parser.parse_args()

# init distributed
def init_distributed():
    global RANK, WORLD_SIZE, IS_DISTRIBUTED
    IS_DISTRIBUTED = False
    if args.dist or 'RANK' in os.environ or 'LOCAL_RANK' in os.environ:
        if args.local_rank == -1:
            # prefer env from torchrun
            args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url)
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        IS_DISTRIBUTED = True
    else:
        RANK = 0
        WORLD_SIZE = 1
        IS_DISTRIBUTED = False

init_distributed()

# load json_file
with open(args.json_path, 'r') as f:
    datas = json.load(f)

# load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_segswap_model(
    model_path=args.model_path,
    image_size=args.image_size,
    backbone_size=args.backbone_size,
    backbone_type=args.backbone_type,
    extractor_type=args.extractor_type,
    device=device
)
print('Model loaded successfully!')

# initialize metrics
intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
le_meter = AverageMeter("LE", ":6.3f", Summary.SUM)
iou_meter = AverageMeter("IoU", ":6.3f", Summary.AVERAGE)
shape_acc_meter = AverageMeter("ShapeAcc", ":6.3f", Summary.AVERAGE)
exist_acc_meter = AverageMeter("ExistAcc", ":6.3f", Summary.AVERAGE)
loc_score_meter = AverageMeter("LocScore", ":6.3f", Summary.AVERAGE)

def evaluation():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ensure correct device per-rank
    if device == 'cuda' and args.local_rank is not None and args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
    model.to(device=device, dtype=torch.float).eval()
    
    # Create visualization directory
    if args.vis_path and RANK == 0:
        os.makedirs(args.vis_path, exist_ok=True)
        print(f"Visualization will be saved to {args.vis_path}")
    
    if RANK == 0:
        print(f"Distributed: {IS_DISTRIBUTED} | World Size: {WORLD_SIZE}")
        print(f"Processing {len(datas)} samples...")
        print(f"Image size: {args.image_size}")
        print(f"Model: {args.model_path}")
        print(f"Use AMP: {args.use_amp}")
        if args.vis_path:
            print(f"Will visualize first {args.vis_num} samples")
    
    # Visualization counter (global count, not affected by distributed sharding)
    vis_counter = 0
    
    # shard data across ranks (round-robin)
    indices = list(range(RANK, len(datas), WORLD_SIZE))
    iterator = (datas[i] for i in indices)
    pbar = tqdm(indices, disable=(RANK != 0))
    for _i in pbar:
        sample = datas[_i]
        # Load first frame image and annotations
        first_frame_img_path = os.path.join(args.root_path, sample['first_frame_image'])
        first_frame_img = cv2.imread(first_frame_img_path)[..., ::-1]  # BGR to RGB
        orig_size = first_frame_img.shape[:2]
        first_frame_img = Image.fromarray(reshape_img_war(first_frame_img, (args.image_size, args.image_size)))
        
        # Load target frame image
        target_frame_img_path = os.path.join(args.root_path, sample['image'])
        target_frame_img = cv2.imread(target_frame_img_path)[..., ::-1]  # BGR to RGB
        target_orig_size = target_frame_img.shape[:2]
        target_frame_img = Image.fromarray(reshape_img_war(target_frame_img, (args.image_size, args.image_size)))
        
        # Process each instance in first frame
        first_frame_anns = sample['first_frame_anns']
        target_anns = sample['anns']
        
        for ann_idx, (first_ann, target_ann) in enumerate(zip(first_frame_anns, target_anns)):
            # Decode first frame mask
            first_mask_orig = decode(first_ann['segmentation'])  # Keep original mask for visualization
            first_mask = reshape_img_war(first_mask_orig, (args.image_size, args.image_size))
            
            # Decode target ground truth mask
            gt_mask = decode(target_ann['segmentation'])
            
            # Get tensors
            tensor1, tensor2, tensor3 = get_tensors(first_frame_img, target_frame_img, first_mask)
            # 1) Forward pass BEFORE TTT
            with torch.no_grad():
                with autocast(enabled=args.use_amp):
                    output_list_before, _ = model(tensor1, tensor3, tensor2)
                    output_before = torch.sigmoid(output_list_before[-1])
            pred_before = (output_before > MASKThresh).float().cpu().numpy()[0, 0]
            if args.image_size > 480:
                pred_before = cv2.resize(pred_before, (480, 480), interpolation=cv2.INTER_NEAREST)
            pred_before = eval_utils.remove_pad(pred_before, target_orig_size)
            h, w = gt_mask.shape[:2]
            if pred_before.shape != (h, w):
                pred_before = cv2.resize(pred_before, (w, h), interpolation=cv2.INTER_NEAREST)
            pred_before = pred_before.astype(np.uint8)

            # IoU BEFORE
            iou_before, _shape_acc_before = eval_utils.eval_mask(gt_mask, pred_before)

            # 2) Test-time training (optional)
            initial_states = None
            trainable_blocks = None
            if args.ttt_enable:
                initial_states, trainable_blocks = test_time_training(
                    model,
                    tensor1, tensor2, tensor3,
                    lr=args.ttt_lr,
                    iterations=args.ttt_iter,
                    ttt_layers=args.ttt_layers,
                    dice_weight=args.dice_weight,
                    use_amp=args.use_amp
                )
            
            # 3) Forward pass AFTER TTT
            with torch.no_grad():
                with autocast(enabled=args.use_amp):
                    output_list_after, _ = model(tensor1, tensor3, tensor2)
                    output_after = torch.sigmoid(output_list_after[-1])
            # Restore weights after TTT
            if args.ttt_enable and initial_states is not None and trainable_blocks is not None:
                for i, block in enumerate(trainable_blocks):
                    block.load_state_dict(initial_states[i])
                model.eval()
            
            pred_after = (output_after > MASKThresh).float().cpu().numpy()[0, 0]
            if args.image_size > 480:
                pred_after = cv2.resize(pred_after, (480, 480), interpolation=cv2.INTER_NEAREST)
            pred_after = eval_utils.remove_pad(pred_after, target_orig_size)
            if pred_after.shape != (h, w):
                pred_after = cv2.resize(pred_after, (w, h), interpolation=cv2.INTER_NEAREST)
            pred_after = pred_after.astype(np.uint8)

            # IoU AFTER
            iou_after, _shape_acc_after = eval_utils.eval_mask(gt_mask, pred_after)

            # Choose final prediction for metric accumulation
            pred_mask = pred_after if args.ttt_enable else pred_before
            
            # Compute metrics for this instance using eval_utils
            iou, shape_acc = eval_utils.eval_mask(gt_mask, pred_mask)
            ex_acc = eval_utils.existence_accuracy(gt_mask, pred_mask)
            location_score = eval_utils.location_score(gt_mask, pred_mask, size=(h, w))
            
            # Update meters
            pred_y, pred_x = get_center(pred_mask, h, w)
            gt_y, gt_x = get_center(gt_mask, h, w)
            loc_err = get_distance(pred_x, pred_y, gt_x, gt_y)
            le_meter.update(loc_err)
            iou_meter.update(iou)
            shape_acc_meter.update(shape_acc)
            exist_acc_meter.update(ex_acc)
            loc_score_meter.update(location_score)
            
            # Binary IoU calculation for intersection/union meters
            intersection, union, _ = intersectionAndUnionGPU(
                torch.tensor(pred_mask).int().cuda().contiguous().clone(), 
                torch.tensor(gt_mask).int().cuda().contiguous(), 
                2, ignore_index=255
            )
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            intersection_meter.update(intersection)
            union_meter.update(union)
            
            acc_iou = intersection / (union + 1e-5)
            acc_iou[union == 0] = 1.0
            acc_iou_meter.update(acc_iou, n=1)
            
            # Visualization: if TTT is enabled and IoU change exceeds the threshold,
            # save before/after comparison (rank 0 only).
            if args.vis_path and RANK == 0 and args.ttt_enable:
                improvement = iou_after - iou_before
                print(f"!!!Improvement: {improvement:.3f} iou_before: {iou_before:.3f} iou_after: {iou_after:.3f}")
                if (improvement > args.iou_threshold or improvement < -args.iou_threshold) and vis_counter < args.vis_num:
                    # Read original images (BGR)
                    first_frame_img_orig_bgr = cv2.imread(first_frame_img_path)
                    target_frame_img_orig_bgr = cv2.imread(target_frame_img_path)
                    prefix = f"sample_{vis_counter:04d}_iou_{iou_before:.3f}_to_{iou_after:.3f}"
                    save_ttt_visualization(
                        first_frame_img_orig_bgr,
                        target_frame_img_orig_bgr,
                        first_mask_orig,
                        pred_before,
                        pred_after,
                        gt_mask,
                        args.vis_path,
                        prefix
                    )
                    vis_counter += 1
            # If TTT is not enabled, keep the original basic visualization for the first N samples
            elif args.vis_path and RANK == 0 and not args.ttt_enable and vis_counter < args.vis_num:
                first_frame_img_orig = cv2.imread(first_frame_img_path)
                target_frame_img_orig = cv2.imread(target_frame_img_path)
                first_mask_vis = first_mask_orig.copy()
                if first_mask_vis.shape[:2] != (first_frame_img_orig.shape[0], first_frame_img_orig.shape[1]):
                    first_mask_vis = cv2.resize(first_mask_vis, (first_frame_img_orig.shape[1], first_frame_img_orig.shape[0]), interpolation=cv2.INTER_NEAREST)
                img1_with_mask = overlay_mask_on_image(first_frame_img_orig.astype(np.float32) / 255.0, first_mask_vis, color=(0, 1, 0), alpha=0.5) * 255
                pred_mask_vis = pred_mask.copy()
                if pred_mask_vis.shape[:2] != (target_frame_img_orig.shape[0], target_frame_img_orig.shape[1]):
                    pred_mask_vis = cv2.resize(pred_mask_vis, (target_frame_img_orig.shape[1], target_frame_img_orig.shape[0]), interpolation=cv2.INTER_NEAREST)
                img2_with_pred = overlay_mask_on_image(target_frame_img_orig.astype(np.float32) / 255.0, pred_mask_vis, color=(0, 0, 1), alpha=0.5) * 255
                gt_mask_vis = gt_mask.copy()
                if gt_mask_vis.shape[:2] != (target_frame_img_orig.shape[0], target_frame_img_orig.shape[1]):
                    gt_mask_vis = cv2.resize(gt_mask_vis, (target_frame_img_orig.shape[1], target_frame_img_orig.shape[0]), interpolation=cv2.INTER_NEAREST)
                img2_with_gt = overlay_mask_on_image(target_frame_img_orig.astype(np.float32) / 255.0, gt_mask_vis, color=(0, 1, 0), alpha=0.5) * 255
                sample_id = f"sample_{vis_counter:04d}_iou_{iou:.3f}"
                cv2.imwrite(f"{args.vis_path}/{sample_id}_source_with_mask.jpg", img1_with_mask.astype(np.uint8))
                cv2.imwrite(f"{args.vis_path}/{sample_id}_target_with_pred.jpg", img2_with_pred.astype(np.uint8))
                cv2.imwrite(f"{args.vis_path}/{sample_id}_target_with_gt.jpg", img2_with_gt.astype(np.uint8))
                vis_counter += 1
    
    # synchronize and reduce metrics
    if IS_DISTRIBUTED:
        dist.barrier()
        intersection_meter.all_reduce()
        union_meter.all_reduce()
        acc_iou_meter.all_reduce()
        le_meter.all_reduce()
        iou_meter.all_reduce()
        shape_acc_meter.all_reduce()
        exist_acc_meter.all_reduce()
        loc_score_meter.all_reduce()

    # Compute final metrics (only rank 0 prints)
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    binary_iou = iou_class[1]

    if RANK == 0:
        print('=' * 50)
        print('EVALUATION RESULTS:')
        print('=' * 50)
        print(f'MEAN IOU (from eval_utils): {iou_meter.avg*100:.4f}')
        print(f'BINARY IOU (from intersectionAndUnion): {binary_iou*100:.4f}')
        print(f'MEAN LOCATION SCORE: {loc_score_meter.avg:.4f}')
        print(f'MEAN SHAPE ACC (Boundary): {shape_acc_meter.avg:.4f}')
        print(f'MEAN EXISTENCE ACC: {exist_acc_meter.avg*100:.4f}')
        print(f'MEAN LOCATION ERROR: {le_meter.avg:.4f}')
        print(f'Total instances evaluated: {int(acc_iou_meter.count)}')
        print('=' * 50)

if __name__ == "__main__":
    evaluation()

"""

# Example command with visualization enabled
torchrun --nproc_per_node=8 eval_handal.py \
    --json_path handal/handal_test_visual.json \
    --model_path handal/train_checkpoint/final.pth \
    --root_path handal \
    --image_size 512 \
    --backbone_size large \
    --backbone_type dinov3 \
    --extractor_type dinov3_cn_large \
    --use_amp \
    --dist \
    --vis_path handal/vis_results \
    --vis_num 1000

torchrun --nproc_per_node=8 eval_handal.py \
    --json_path handal/handal_test_visual.json \
    --model_path train/output/1102_dinov3cnlarge_dinov3large_dice5_bs16as16ep200_mxlr1e5_lp20lr1e4/best_test_miou.pth \
    --root_path handal \
    --image_size 512 \
    --backbone_size large \
    --backbone_type dinov3 \
    --extractor_type dinov3_cn_large \
    --use_amp \
    --dist \
    --ttt_enable \
    --vis_path handal/vis_results_ttt \
    --vis_num 100

gdown "https://drive.google.com/file/d/1VMyETGvnciDo2SisjbpNPULnVx-PjPXH/view?usp=drive_link" --fuzzy
"""
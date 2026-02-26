import os
import sys
import math
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LambdaLR, CosineAnnealingLR
import torchvision.transforms as transforms
from pycocotools.mask import decode
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# project imports
sys.path.append('model/')
from train.csegmentor import ConditionalSegmentationModel

# ------------------------
# Helpers
# ------------------------
def reshape_img_war(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Pad any HxW or HxWxC image to a square (aspect-ratio preserved), then resize to `size` (same as eval)."""
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

    return temp  # Same as eval: keep single-channel dim (cv2.resize will handle it)


class HandALPairsDataset(Dataset):
    """Flatten each JSON sample into per-instance (first_frame, first_mask, target_frame, gt_mask) pairs."""

    def __init__(self, json_path: str, root_path: str, image_size: int = 512):
        super().__init__()
        self.root_path = root_path
        self.image_size = image_size

        with open(json_path, 'r') as f:
            datas = json.load(f)

        self.items: List[dict] = []
        for sample in datas:
            first_img_path = os.path.join(root_path, sample['first_frame_image'])
            target_img_path = os.path.join(root_path, sample['image'])
            first_anns = sample['first_frame_anns']
            target_anns = sample['anns']
            for first_ann, target_ann in zip(first_anns, target_anns):
                self.items.append({
                    'first_img': first_img_path,
                    'first_mask': first_ann['segmentation'],
                    'target_img': target_img_path,
                    'gt_mask': target_ann['segmentation'],
                })

        norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]

        # Load images (BGR->RGB) - identical to eval
        first_img = cv2.imread(item['first_img'])[..., ::-1]  # BGR to RGB
        target_img = cv2.imread(item['target_img'])[..., ::-1]  # BGR to RGB

        # Pad+resize to square, then convert to PIL.Image - identical to eval
        first_img_sq = Image.fromarray(reshape_img_war(first_img, (self.image_size, self.image_size)))
        target_img_sq = Image.fromarray(reshape_img_war(target_img, (self.image_size, self.image_size)))

        # Masks - identical to eval
        first_mask = decode(item['first_mask'])  # HxW
        gt_mask = decode(item['gt_mask'])        # HxW
        first_mask_sq = reshape_img_war(first_mask, (self.image_size, self.image_size))
        gt_mask_sq = reshape_img_war(gt_mask, (self.image_size, self.image_size))

        # To tensors - consistent with eval's get_tensors
        I1 = self.transform(first_img_sq)  # PIL.Image -> tensor
        I2 = self.transform(target_img_sq)  # PIL.Image -> tensor
        M1 = torch.from_numpy(first_mask_sq).float().unsqueeze(0)  # HxW -> 1xHxW
        GT = torch.from_numpy(gt_mask_sq).float().unsqueeze(0)     # HxW -> 1xHxW

        return I1, M1, I2, GT


def load_segswap_model(model_path: str = None,
                       image_size: int = 512,
                       backbone_size: str = 'large',
                       backbone_type: str = 'dinov3',
                       extractor_type: str = 'dinov3_cn_large',
                       device: str = 'cuda') -> ConditionalSegmentationModel:
    model = ConditionalSegmentationModel(
        feat_extractor=extractor_type,
        extractor_depth=2,
        backbone_size=backbone_size,
        image_size=image_size,
        upsampler='bilinear',
        backbone_type=backbone_type,
        num_register_tokens=4,
    )
    if model_path and os.path.exists(model_path):
        print(f"Loading init weights from {model_path}")
        # Compatible with PyTorch 2.6 safe-loading policy
        try:
            state = torch.load(model_path, map_location='cpu', weights_only=True)
        except Exception:
            try:
                # Allow numpy scalars as safe globals
                torch.serialization.add_safe_globals([np.core.multiarray.scalar])
                state = torch.load(model_path, map_location='cpu', weights_only=True)
            except Exception:
                # If it still fails, treat as trusted and fall back to weights_only=False
                state = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(state, dict) and 'encoder' in state:
            state = state['encoder']
        # Strict check: fail fast on missing/unexpected keys to avoid unstable training from partial loading
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[load_state] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
            raise RuntimeError("State dict mismatch. Please ensure model/backbone/extractor settings match the checkpoint.")
    model.to(device)
    return model


def dice_loss(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # probs, targets: [B,1,H,W]
    probs = probs.contiguous().view(probs.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def loss_calculation(loss_mask_list, dice_loss_list, dice_weight, aux_weight, n_aux_layers):
    loss = loss_mask_list[-1] + dice_weight * dice_loss_list[-1]
    if abs(aux_weight) < 1e-6:
        return loss, torch.tensor(0.0, device=loss.device)
    aux_loss = 0
    for i in range(n_aux_layers):
        aux_loss += loss_mask_list[i] + dice_weight * dice_loss_list[i]
    aux_loss = aux_weight * aux_loss
    return aux_loss + loss, aux_loss


def build_optimizer_and_scheduler(model: nn.Module,
                                  max_lr: float,
                                  min_lr: float,
                                  epochs: int,
                                  steps_per_epoch: int,
                                  warmup_ratio: float):
    optimizer = AdamW(model.parameters(), lr=max_lr, weight_decay=1e-4)

    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(1, int(warmup_ratio * total_steps))
    cosine_steps = max(1, total_steps - warmup_steps)

    # Linear warmup: from min_lr to max_lr
    start_scale = min_lr / max_lr
    def warmup_lambda(current_step: int):
        if current_step >= warmup_steps:
            return 1.0
        alpha = current_step / max(1, warmup_steps)
        return start_scale + (1.0 - start_scale) * alpha

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # Then cosine anneal down to min_lr
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=min_lr)

    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    return optimizer, scheduler, total_steps, warmup_steps


def init_distributed(args):
    """Initialize distributed training (torchrun)."""
    args.distributed = False
    args.rank = 0
    args.world_size = 1
    args.local_rank = int(os.environ.get('LOCAL_RANK', -1))

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        if args.local_rank == -1:
            args.local_rank = args.rank % max(1, torch.cuda.device_count())
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        dist.barrier()
        args.distributed = True

    return args


def is_main_process(args) -> bool:
    return (not getattr(args, 'distributed', False)) or getattr(args, 'rank', 0) == 0


def train(args):
    # DDP init
    init_distributed(args)
    torch.backends.cudnn.benchmark = True

    device = f"cuda:{args.local_rank}" if args.distributed else ('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset & dataloader
    dataset = HandALPairsDataset(args.json_path, args.root_path, args.image_size)
    if args.distributed:
        train_sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True, drop_last=False)
    else:
        train_sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Model
    model = load_segswap_model(
        model_path=args.init_model_path,
        image_size=args.image_size,
        backbone_size=args.backbone_size,
        backbone_type=args.backbone_type,
        extractor_type=args.extractor_type,
        device=device,
    )
    if args.distributed:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
            broadcast_buffers=False  # Align with train/main.py; avoid potential version issues from buffer sync
        )
    model.train()

    # Loss (shared by linear-probing and main training)
    bce_loss = nn.BCEWithLogitsLoss()

    # amp
    scaler = GradScaler(enabled=args.use_amp)

    os.makedirs(args.output_dir, exist_ok=True)

    global_step = 0

    # =========================
    # Linear probing stage (optional)
    # Freeze encoder/backbone and train the head only.
    # =========================
    if getattr(args, 'lp_n_epoch', 0) and args.lp_n_epoch > 0:
        inner_model = model.module if isinstance(model, DDP) else model
        # Freeze modules consistent with train/main.py
        if hasattr(inner_model, 'encoder'):
            for p in inner_model.encoder.parameters():
                p.requires_grad = False
        if hasattr(inner_model, 'backbone'):
            for p in inner_model.backbone.parameters():
                p.requires_grad = False

        # Build optimizer/scheduler for trainable params only
        lp_trainable_params = [p for p in inner_model.parameters() if p.requires_grad]
        lp_optimizer, lp_scheduler, lp_total_steps, _ = build_optimizer_and_scheduler(
            inner_model,
            max_lr=getattr(args, 'lp_max_lr', 2e-3),
            min_lr=getattr(args, 'lp_min_lr', 2e-4),
            epochs=args.lp_n_epoch,
            steps_per_epoch=max(1, len(dataloader)),
            warmup_ratio=args.warmup_ratio,
        )
        # Ensure optimizer only contains trainable params (same idea as main.py)
        for pg in list(lp_optimizer.param_groups):
            pg['params'].clear()
        if lp_trainable_params:
            lp_optimizer.add_param_group({'params': lp_trainable_params})

        lp_progress = tqdm(total=lp_total_steps, desc='LinearProbe', ncols=120) if is_main_process(args) else None
        for epoch in range(1, args.lp_n_epoch + 1):
            epoch_loss = 0.0
            if args.distributed and isinstance(dataloader.sampler, DistributedSampler):
                dataloader.sampler.set_epoch(epoch)
            for batch in dataloader:
                I1, M1, I2, GT = batch
                I1 = I1.to(device, non_blocking=True)
                M1 = M1.to(device, non_blocking=True)
                I2 = I2.to(device, non_blocking=True)
                GT = GT.to(device, non_blocking=True)

                lp_optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=args.use_amp):
                    output_list, _ = model(I1, M1, I2)
                    logits = output_list[-1]

                    loss_mask_list = []
                    dice_loss_list = []
                    for out_i in output_list:
                        if GT.shape[-2:] != out_i.shape[-2:]:
                            GT_i = F.interpolate(GT, size=out_i.shape[-2:], mode='nearest')
                        else:
                            GT_i = GT
                        loss_mask_list.append(bce_loss(out_i, GT_i))
                        dice_loss_list.append(dice_loss(torch.sigmoid(out_i), GT_i))

                    n_aux_layers = max(0, len(output_list) - 1)
                    loss, aux_loss = loss_calculation(
                        loss_mask_list,
                        dice_loss_list,
                        args.dice_weight,
                        args.aux_weight,
                        n_aux_layers,
                    )

                    if abs(args.consistency_weight) > 1e-6:
                        output = torch.sigmoid(output_list[-1])
                        FM2 = output.type(torch.FloatTensor).cuda()
                        reversed_output_list, _ = model(I2, FM2, I1)
                        rev_logits = reversed_output_list[-1]
                        if M1.shape[-2:] != rev_logits.shape[-2:]:
                            M1_resized = F.interpolate(M1, size=rev_logits.shape[-2:], mode='nearest')
                        else:
                            M1_resized = M1
                        consistency_loss = bce_loss(rev_logits, M1_resized)
                        loss = loss + args.consistency_weight * consistency_loss
                    else:
                        consistency_loss = torch.tensor(0.0, device=logits.device)

                scaler.scale(loss).backward()
                scaler.step(lp_optimizer)
                scaler.update()
                lp_scheduler.step()

                global_step += 1
                epoch_loss += loss.item()
                if lp_progress is not None:
                    lp_progress.update(1)
                    lp_progress.set_postfix({
                        'epoch': epoch,
                        'loss': f"{loss.item():.4f}",
                        'lr': f"{lp_optimizer.param_groups[0]['lr']:.6e}",
                    })

            avg_loss = epoch_loss / max(1, len(dataloader))
            if is_main_process(args):
                print(f"[LP] Epoch {epoch}/{args.lp_n_epoch} - avg_loss: {avg_loss:.6f}")

        if lp_progress is not None:
            lp_progress.close()

        # Unfreeze all parameters for the main training stage
        for p in inner_model.parameters():
            p.requires_grad = True

    # =========================
    # Main training stage
    # =========================
    optimizer, scheduler, total_steps, warmup_steps = build_optimizer_and_scheduler(
        model,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        epochs=args.epochs,
        steps_per_epoch=max(1, len(dataloader)),
        warmup_ratio=args.warmup_ratio,
    )

    progress = tqdm(total=total_steps, desc='Training', ncols=120) if is_main_process(args) else None

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        if args.distributed and isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)
        for batch in dataloader:
            I1, M1, I2, GT = batch
            I1 = I1.to(device, non_blocking=True)
            M1 = M1.to(device, non_blocking=True)
            I2 = I2.to(device, non_blocking=True)
            GT = GT.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=args.use_amp):
                # Dataset already returns correct shapes:
                # I1/I2: [B,3,H,W], M1/GT: [B,1,H,W]
                # Use directly to match eval.
                output_list, _ = model(I1, M1, I2)
                logits = output_list[-1]  # [B,1,H,W]

                # Build loss lists for multi-scale outputs (resize targets as needed)
                loss_mask_list = []
                dice_loss_list = [] 
                for out_i in output_list:
                    if GT.shape[-2:] != out_i.shape[-2:]:
                        GT_i = F.interpolate(GT, size=out_i.shape[-2:], mode='nearest')
                    else:
                        GT_i = GT
                    loss_mask_list.append(bce_loss(out_i, GT_i))
                    dice_loss_list.append(dice_loss(torch.sigmoid(out_i), GT_i))

                n_aux_layers = max(0, len(output_list) - 1)
                loss, aux_loss = loss_calculation(
                    loss_mask_list,
                    dice_loss_list,
                    args.dice_weight,
                    args.aux_weight,
                    n_aux_layers,
                )

                # Consistency loss (no separate consistency_dice_weight)
                if abs(args.consistency_weight) > 1e-6:
                    output = torch.sigmoid(output_list[-1])
                    # Align with train/train.py: use .type(torch.FloatTensor).cuda() to break gradients w.r.t. the original graph
                    FM2 = output.type(torch.FloatTensor).cuda()
                    reversed_output_list, _ = model(I2, FM2, I1)
                    rev_logits = reversed_output_list[-1]
                    if M1.shape[-2:] != rev_logits.shape[-2:]:
                        M1_resized = F.interpolate(M1, size=rev_logits.shape[-2:], mode='nearest')
                    else:
                        M1_resized = M1
                    consistency_loss = bce_loss(rev_logits, M1_resized)
                    loss = loss + args.consistency_weight * consistency_loss
                else:
                    consistency_loss = torch.tensor(0.0, device=logits.device)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()
            if progress is not None:
                progress.update(1)
                progress.set_postfix({
                    'epoch': epoch,
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.6e}",
                })

        avg_loss = epoch_loss / max(1, len(dataloader))
        if is_main_process(args):
            print(f"Epoch {epoch}/{args.epochs} - avg_loss: {avg_loss:.6f}")

    if progress is not None:
        progress.close()

    # Save final weights
    if is_main_process(args):
        final_path = os.path.join(args.output_dir, "final.pth")
        state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        torch.save({'encoder': state_dict}, final_path)
        print(f"Saved final weights: {final_path}")

    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, required=True, help='Path to training JSON (same structure as evaluation)')
    parser.add_argument('--root_path', type=str, required=True, help='Dataset image root directory')
    parser.add_argument('--init_model_path', type=str, default=None, help='Init checkpoint path (may be an eval-saved encoder checkpoint)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')

    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--backbone_size', type=str, default='large')
    parser.add_argument('--backbone_type', type=str, default='dinov3')
    parser.add_argument('--extractor_type', type=str, default='dinov3_cn_large')
    parser.add_argument('--extractor_depth', type=int, default=2)
    parser.add_argument('--upsampler', type=str, default='bilinear')
    parser.add_argument('--n_aux_layers', type=int, default=1)
    parser.add_argument('--num_register_tokens', type=int, default=4)

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--use_amp', action='store_true')

    # Linear probing config (consistent with train/main.py: freeze encoder/backbone, train head only)
    parser.add_argument('--lp_n_epoch', type=int, default=0, help='Linear probing epochs; 0 to skip')
    parser.add_argument('--lp_max_lr', type=float, default=2e-3, help='Max LR in LP stage (default: 2e-3)')
    parser.add_argument('--lp_min_lr', type=float, default=2e-4, help='Min LR in LP stage (default: 2e-4)')

    parser.add_argument('--max_lr', type=float, default=2e-5, help='Maximum learning rate')
    parser.add_argument('--min_lr', type=float, default=2e-6, help='Minimum learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio (relative to total steps)')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank provided by torchrun')
    parser.add_argument('--dice_weight', type=float, default=5.0, help='Dice loss weight (same for main/aux)')
    parser.add_argument('--aux_weight', type=float, default=1.0, help='Total auxiliary loss weight (0 to disable)')
    parser.add_argument('--consistency_weight', type=float, default=10.0, help='Self-supervised consistency loss weight (0 to disable)')

    args = parser.parse_args()

    train(args)

    """
torchrun --nproc_per_node=8 train_handal.py \
  --json_path handal/handal_train_visual.json \
  --root_path handal \
  --init_model_path train/output/1102_dinov3cnlarge_dinov3large_dice5_bs16as16ep200_mxlr1e5_lp20lr1e4/best_test_miou.pth \
  --output_dir handal/train_checkpoint \
  --image_size 512 \
  --backbone_size large \
  --backbone_type dinov3 \
  --extractor_type dinov3_cn_large \
  --batch_size 4 \
  --num_workers 4 \
  --use_amp
    """



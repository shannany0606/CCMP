import logging
import math
from functools import partial
import torch
import torch.nn.functional as F
from datetime import datetime
import os
import sys
import random
import numpy as np

def Warm_cos_lr(max_lr,
                min_lr,
                total_iter,
                warmup_total_iter,
                iter):
    """Cosine learning rate with warm up."""
    if iter <= warmup_total_iter:
        lr = max_lr * pow(iter / float(warmup_total_iter), 2)
    else:
        lr = min_lr + 0.5 * (max_lr - min_lr) * (
            1.0
            + math.cos(
                math.pi
                * (iter - warmup_total_iter)
                / total_iter
            )
        )
    return lr

class Warmup_cos_lr:
    def __init__(self,
                 max_lr,
                 min_lr,
                 iter_per_epoch,
                 num_epoch,
                 warmup_epoch):
        """
        Args:
            max_lr : maximun learning rate in the cosine learning rate scheduler
            min_lr : minimum learning rate in the cosine learning rate scheduler (used in no aug epochs)
            iter_per_epoch : number of iterations in one epoch.
            num_epoch : number of epochs in training.
            warmup_epoch : number of epochs in warm-up.
        """
        warmup_iter = iter_per_epoch * warmup_epoch
        total_iter = iter_per_epoch * num_epoch
        
        self.lr_func = partial(Warm_cos_lr,
                               max_lr,
                               min_lr,
                               total_iter,
                               warmup_iter)

    def update_lr(self, iter):
        return self.lr_func(iter)

def get_logger(logdir):
    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    ts = str(datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # if have gpu
    torch.cuda.manual_seed_all(seed)  # if multiple gpus
    torch.backends.cudnn.deterministic = True  # ensure reproducing for CNN
    torch.backends.cudnn.benchmark = False  # ban acceleration...

def load_pretrained_models(netEncoder, args, logger):
    """
    Load pretrained extractor and backbone weights into netEncoder.
    Handles resizing positional embeddings if target image size differs from 224.
    """

    # Load the feature extractor (encoder)
    extractor_dict = torch.load(args.extractor_pretrain_pth)
    if 'model' in extractor_dict:
        extractor_dict = extractor_dict['model']
    netEncoder.encoder.load_state_dict(extractor_dict, strict=False)
    logger.info("Successfully loaded extractor_pretrain_pth!")

    # Load ViT backbone model for traditional backbones (mae, deit3, dinov2)
    vit_checkpoint = torch.load(args.vit_pretrained_pth)
    vit_dict = vit_checkpoint['model'] if 'model' in vit_checkpoint else vit_checkpoint

    # Set patch size based on backbone type
    patch_size = 14 if args.backbone_type == 'dinov2' else 16

    # Process positional embeddings if needed (skip for dinov3 which uses RoPE)
    if args.backbone_type != 'dinov3':
        if args.image_size != 224 and args.backbone_type != 'dinov2': 
            pos_embed_default = vit_dict['pos_embed']
            target_shape = args.image_size // patch_size

            if args.backbone_type == 'deit3': # deit3 does not have cls token, its shape is [1, 196, 768]
                cls_pos_embed = torch.zeros_like(pos_embed_default[:, 0:1, :])
                pos_embed_resized = pos_embed_default.reshape(1, 224 // patch_size, 224 // patch_size, -1).permute(0, 3, 1, 2)
            else:
                cls_pos_embed = pos_embed_default[:, 0:1, :]
                pos_embed_resized = pos_embed_default[:, 1:, :].reshape(1, 224 // patch_size, 224 // patch_size, -1).permute(0, 3, 1, 2)

            pos_embed_resized = F.interpolate(pos_embed_resized, size=(target_shape, target_shape), mode='bicubic', align_corners=False)
            pos_embed_resized = pos_embed_resized.permute(0, 2, 3, 1).reshape(1, target_shape * target_shape, -1)
        else : 
            pos_embed_default = vit_dict['pos_embed']

            if args.backbone_type == 'deit3':
                cls_pos_embed = torch.zeros_like(pos_embed_default[:, 0:1, :])
                pos_embed_resized = pos_embed_default
            else:
                cls_pos_embed = pos_embed_default[:, 0:1, :]
                pos_embed_resized = pos_embed_default[:, 1:, :]

        cond_pos_embed = cls_pos_embed.clone()
        if args.backbone_type in ['mae', 'dinov2']:
            vit_dict['pos_embed'] = torch.cat([cls_pos_embed, cond_pos_embed, pos_embed_resized], dim=1)
        elif args.backbone_type == 'deit3':
            vit_dict['pos_embed'] = torch.cat([cond_pos_embed, pos_embed_resized], dim=1)
    else:
        # DINOv3 uses RoPE (Rotary Position Embedding), no pos_embed processing needed
        logger.info("DINOv3 model uses RoPE, skipping pos_embed processing")

    # Load the modified ViT weights
    netEncoder.backbone.load_state_dict(vit_dict, strict=True)
    netEncoder.cuda()
    logger.info(f"Successfully loaded {args.backbone_type}_pretrained_pth!")


def save_model(netEncoder, optimizer, path, epoch, best_val_miou, best_test_miou):
    encoder_state = netEncoder.state_dict()
    if hasattr(netEncoder, 'module'):
        encoder_state = netEncoder.module.state_dict()

    param = {
        'encoder': encoder_state,
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_val_miou': best_val_miou,
        'best_test_miou': best_test_miou
    }
    torch.save(param, path)


# coding=utf-8
import os 
import sys 
sys.path.append('model/')
import numpy as np 
from PIL import Image
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import json
import copy

import torch 
import torchvision.transforms as transforms
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from train.csegmentor import ConditionalSegmentationModel

import tqdm

from pycocotools import mask as mask_utils
import utils

from torch.cuda.amp import autocast, GradScaler

import train.losses as losses

MASKThresh = 0.5

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

def load_frame(path, frame_idx, image_size=518, ret_size=False):
    img = cv2.imread(os.path.join(path, f'{frame_idx}.jpg'))[..., ::-1]
    orig_size = img.shape[:-1]
    img = Image.fromarray(reshape_img_war(img, (image_size, image_size)))
    if ret_size:
        return img, orig_size
    return img

def egoexo(netEncoder, annotations, ego, exo, obj, take, anno_path, pred_json, image_size, device, ttt_enable=False, ttt_lr=0.0001, ttt_iter=5, ttt_layers=2, dice_weight=0.0, use_amp=False):

    pred_json['masks'][obj][f'{ego}_{exo}'] = {}
    for idx in annotations['masks'][obj][ego].keys():

        ego_frame = load_frame(path=f'{anno_path}/{take}/{ego}/', frame_idx=idx, image_size=image_size)
        ego_mask = mask_utils.decode(annotations['masks'][obj][ego][idx])
        ego_mask = reshape_img_war(ego_mask, (image_size, image_size))

        exo_frame, exo_size = load_frame(path=f'{anno_path}/{take}/{exo}/', frame_idx=idx, image_size=image_size, ret_size=True)

        tensor1, tensor2, tensor3 = get_tensors(ego_frame, exo_frame, ego_mask, device)
        
        initial_states = None
        trainable_blocks = None
        
        if ttt_enable:
            netEncoder, initial_states, trainable_blocks = test_time_training(
                netEncoder, tensor1, tensor2, tensor3, 
                lr=ttt_lr, iterations=ttt_iter, use_amp=use_amp, ttt_layers=ttt_layers, dice_weight=dice_weight
            )
            
        my, confidence = forward_pass(netEncoder, tensor1, tensor2, tensor3, image_size, use_amp)

        if ttt_enable and initial_states is not None and trainable_blocks is not None:
            for i, block in enumerate(trainable_blocks):
                block.load_state_dict(initial_states[i])
            netEncoder.eval()

        y_step = (my > MASKThresh)
        y_step = utils.remove_pad(y_step, orig_size=exo_size)

        exo_pred = mask_utils.encode(np.asfortranarray(y_step.astype(np.uint8)))
        exo_pred['counts'] = exo_pred['counts'].decode('ascii')
        pred_json['masks'][obj][f'{ego}_{exo}'][idx] = {'pred_mask': exo_pred, 'confidence': confidence}

def exoego(netEncoder, annotations, ego, exo, obj, take, anno_path, pred_json, image_size, device, ttt_enable=False, ttt_lr=0.0001, ttt_iter=5, ttt_layers=2, dice_weight=0.0, use_amp=False):

    pred_json['masks'][obj][f'{exo}_{ego}'] = {}
    for idx in annotations['masks'][obj][exo].keys():

        exo_frame = load_frame(path=f'{anno_path}/{take}/{exo}/', frame_idx=idx, image_size=image_size)
        exo_mask = mask_utils.decode(annotations['masks'][obj][exo][idx])
        exo_mask = reshape_img_war(exo_mask, (image_size, image_size))

        ego_frame = load_frame(path=f'{anno_path}/{take}/{ego}/', frame_idx=idx, image_size=image_size)

        tensor1, tensor2, tensor3 = get_tensors(exo_frame, ego_frame, exo_mask, device)
        
        initial_states = None
        trainable_blocks = None
        
        if ttt_enable:
            netEncoder, initial_states, trainable_blocks = test_time_training(
                netEncoder, tensor1, tensor2, tensor3, 
                lr=ttt_lr, iterations=ttt_iter, use_amp=use_amp, ttt_layers=ttt_layers, dice_weight=dice_weight
            )
            
        my, confidence = forward_pass(netEncoder, tensor1, tensor2, tensor3, image_size, use_amp)

        if ttt_enable and initial_states is not None and trainable_blocks is not None:
            for i, block in enumerate(trainable_blocks):
                block.load_state_dict(initial_states[i])
            netEncoder.eval()

        y_step = (my > MASKThresh)

        ego_pred = mask_utils.encode(np.asfortranarray(y_step.astype(np.uint8)))
        ego_pred['counts'] = ego_pred['counts'].decode('ascii')
        pred_json['masks'][obj][f'{exo}_{ego}'][idx] = {'pred_mask': ego_pred, 'confidence': confidence}

def shard_list_by_rank(items, rank, world_size):
    if world_size <= 1:
        return items
    return items[rank::world_size]

def main(model_path, takes, anno_path, out_path, setting='ego-exo', save_inter=False, image_size=518, use_amp=False,
        ttt_enable=False, ttt_lr=0.0001, ttt_iter=5, ttt_layers=2, dice_weight=0.0, backbone_size='large', backbone_type='dinov2',
        extractor_type='cn2_base', device=None,
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
    
    assigned_takes = shard_list_by_rank(takes, rank, world_size) if is_distributed else takes
    results = {}
    for take in tqdm.tqdm(assigned_takes, disable=(rank != 0)):

        with open(f'{anno_path}/{take}/annotation.json', 'r') as fp:
            annotations = json.load(fp)

        pred_json = {'masks': {}, 'subsample_idx': annotations['subsample_idx']}

        for obj in annotations['masks']:

            pred_json['masks'][obj] = {}

            cams = annotations['masks'][obj].keys()

            exo_cams = [x for x in cams if 'aria' not in x]
            ego_cams = [x for x in cams if 'aria' in x]

            for ego in ego_cams:
                for exo in exo_cams:
                    # ego -> exo
                    if setting == 'ego-exo':
                        egoexo(netEncoder=netEncoder, annotations=annotations,
                              ego=ego, exo=exo, obj=obj, take=take, anno_path=anno_path, pred_json=pred_json, 
                              image_size=image_size, device=device, ttt_enable=ttt_enable, ttt_lr=ttt_lr, 
                              ttt_iter=ttt_iter, ttt_layers=ttt_layers, dice_weight=dice_weight, use_amp=use_amp)
                    elif setting == 'exo-ego':
                        exoego(netEncoder=netEncoder, annotations=annotations,
                              ego=ego, exo=exo, obj=obj, take=take, anno_path=anno_path, pred_json=pred_json, 
                              image_size=image_size, device=device, ttt_enable=ttt_enable, ttt_lr=ttt_lr, 
                              ttt_iter=ttt_iter, ttt_layers=ttt_layers, dice_weight=dice_weight, use_amp=use_amp)
                    else:
                        raise Exception(f"Setting {setting} not recognized.")

        results[take] = pred_json

        if save_inter:
            os.makedirs(f'{out_path}/{take}', exist_ok=True)
            with open(f'{out_path}/{take}/pred_annotations.json', 'w') as fp:
                json.dump(pred_json, fp)

    return results

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
    parser.add_argument('--ttt_iter', type=int, default=3, help="Number of iterations for test-time training")
    parser.add_argument('--ttt_layers', type=int, default=3, help="Number of backbone layers to fine-tune during test-time training")
    parser.add_argument('--dice_weight', type=float, default=0.0, help="Weight for dice loss in test-time training")

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
            # Increase timeout to 720 minutes
            import datetime
            timeout = datetime.timedelta(minutes=720)
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            dist.init_process_group(backend=args.dist_backend, init_method='env://', timeout=timeout)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        # ensure no stale process group
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    with open(args.splits_path, "r") as fp:
        splits = json.load(fp)
    
    results_local = main(args.ckpt_path,
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
                dice_weight=args.dice_weight,
                backbone_size=args.backbone_size,
                backbone_type=args.backbone_type,
                extractor_type=args.extractor_type,
                rank=rank,
                world_size=world_size
                )

    # Save local results to disk first to avoid gather_object OOM/SegFault
    os.makedirs(args.out_path, exist_ok=True)
    part_file = os.path.join(args.out_path, f'part_rank_{rank}.json')
    print(f'Rank {rank}: Saving results to {part_file}')
    with open(part_file, 'w') as fp:
        json.dump(results_local, fp)
    
    # gather results on rank0 via file system (more robust than dist.gather_object for large data)
    results = None
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        dist.barrier() # Wait for all ranks to save
        if rank == 0:
            results = {}
            print('Rank 0: Merging results from files...')
            for r in range(world_size):
                p_file = os.path.join(args.out_path, f'part_rank_{r}.json')
                if os.path.exists(p_file):
                    with open(p_file, 'r') as fp:
                        part_data = json.load(fp)
                        results.update(part_data)
                    os.remove(p_file) # Optional: keep for debug
                else:
                    print(f'Warning: {p_file} missing!')
    else:
        results = results_local

    if (not dist.is_available()) or (not dist.is_initialized()) or rank == 0:
        os.makedirs(args.out_path, exist_ok=True)
        if args.ttt_enable:
            with open(f"{args.out_path}/{args.setting}_{args.split}_results_ttt.json", "w") as fp:
                json.dump({args.setting: {'results': results}}, fp)
        else:
            with open(f"{args.out_path}/{args.setting}_{args.split}_results.json", "w") as fp:
                json.dump({args.setting: {'results': results}}, fp)

    # cleanup
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
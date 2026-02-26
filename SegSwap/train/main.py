# coding=utf-8
import json
import itertools
import os 

import option
args = option.get_option()

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
device = torch.device('cuda')

from datetime import datetime
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler
# import swanlab

import dataloader # dataloader
import train # train
import utils
import csegmentor

torch.cuda.empty_cache()
utils.set_seed(42)

# run = swanlab.init(project="Correspondence", config=args)
class DummyWriter:
    def add_scalar(self, *args, **kwargs):
        pass
    def add_image(self, *args, **kwargs):
        pass
    def close(self):
        pass

writer = None
logger = None

# Distributed initialization
distributed = False
if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and 'LOCAL_RANK' in os.environ:
    distributed = True
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    device = torch.device('cuda', local_rank)

# Create logger and TensorBoard writer on the main process only
is_main_process = (not distributed) or (dist.get_rank() == 0)
if is_main_process:
    writer = SummaryWriter(os.path.join(args.out_dir, 'tb_logs'))
    logger = utils.get_logger(args.out_dir)
    logger.info(args)
else:
    writer = DummyWriter()
    logger = logging.getLogger(f'Exp_rank{dist.get_rank() if dist.is_initialized() else 1}')
    logger.addHandler(logging.NullHandler())

netEncoder = csegmentor.ConditionalSegmentationModel(args.feat_extractor,  
                                                     args.extractor_depth,  
                                                     args.backbone_size,  
                                                     args.image_size,
                                                     args.upsampler,
                                                     args.backbone_type,
                                                     args.n_aux_layers,
                                                     args.num_register_tokens).to(device)


optim = torch.optim.AdamW
optimizer = optim(itertools.chain(*[netEncoder.parameters()]), args.max_lr, weight_decay=args.weight_decay)

actual_iters_per_epoch = (args.iter_epoch + args.grad_accum - 1)// args.grad_accum
lr_scheduler = utils.Warmup_cos_lr(args.max_lr, args.min_lr, actual_iters_per_epoch, args.n_epoch, 1)
Loss = torch.nn.BCEWithLogitsLoss()
scaler = GradScaler(enabled=args.use_amp)

if args.resume_path:
    map_location = {'cuda:%d' % 0: 'cuda:%d' % (device.index if device.index is not None else 0)} if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(args.resume_path, map_location=map_location)
    try:
        netEncoder.load_state_dict(checkpoint['encoder'])
    except RuntimeError:
        # Compatibility for DataParallel checkpoints (with a "module." prefix)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['encoder'].items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[name] = v
        netEncoder.load_state_dict(new_state_dict, strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])

    resume_start_epoch = checkpoint.get('epoch', args.resume_start_epoch - 1) + 1
    train.trainEpoch.iter_counter = ((1 + resume_start_epoch) * actual_iters_per_epoch)
    best_val_miou, best_test_miou = checkpoint.get('best_val_miou', 0.0), checkpoint.get('best_test_miou', 0.0) 
    logger.info(f"Resuming from epoch {resume_start_epoch}, best_val_miou: {best_val_miou:.4f}, best_test_miou: {best_test_miou:.4f}")
else:
    utils.load_pretrained_models(netEncoder, args, logger)
    resume_start_epoch = 0
    best_val_miou, best_test_miou = 0.0, 0.0

# Wrap with DDP (after loading weights)
if distributed:
    netEncoder = DDP(
        netEncoder,
        device_ids=[device.index],
        output_device=device.index,
        find_unused_parameters=True,
        broadcast_buffers=False
    )

trainLoader, valLoader, testLoader = dataloader.getDataloader(args.image_size,
                                                              args.train_dir,
                                                              args.data_dir,
                                                              args.batch_size,
                                                              args.iter_epoch_val,
                                                              args.use_data,
                                                              args.prob_neg)

history = {'trainMIoU':[], 'trainAcc':[], 'trainLoss':[], 'trainLossMask':[], 'trainLossDice':[], 'trainLossCls':[], 'trainLossAux':[], 'trainLossConsistency':[]}
val_history = {'valMIoU':[], 'valMIoU_480':[]}
test_history = {'testMIoU': [], 'testMIoU_480': []}

if not args.resume_path:
    # Optional linear probing warmup: freeze pretrained parts and train heads only
    if args.lp_n_epoch > 0:
        # Freeze pretrained parts: encoder and backbone
        for p in netEncoder.module.encoder.parameters():
            p.requires_grad = False
        for p in netEncoder.module.backbone.parameters():
            p.requires_grad = False
        # Resolve LP-specific learning rates (fallback to global if None)
        lp_max_lr = args.lp_max_lr if getattr(args, 'lp_max_lr', None) is not None else args.max_lr
        lp_min_lr = args.lp_min_lr if getattr(args, 'lp_min_lr', None) is not None else args.min_lr
        # Rebuild optimizer on trainable params only
        trainable_params = [p for p in netEncoder.module.parameters() if p.requires_grad]
        optimizer = optim(trainable_params, lp_max_lr, weight_decay=args.weight_decay)
        # Stage-1 scheduler length
        lp_iters_per_epoch = (args.lp_iter_epoch if args.lp_iter_epoch > 0 else args.iter_epoch)
        lp_actual_iters = (lp_iters_per_epoch + args.grad_accum - 1) // args.grad_accum
        lr_scheduler = utils.Warmup_cos_lr(lp_max_lr, lp_min_lr, lp_actual_iters, args.lp_n_epoch, 1)

        # Run  epochs
        for s_epoch in range(args.lp_n_epoch):
            if distributed and hasattr(trainLoader, 'sampler'):
                try:
                    trainLoader.sampler.set_epoch(s_epoch)
                except Exception:
                    pass
            netEncoder, optimizer, history = train.trainEpoch(trainLoader,
                                                              netEncoder,
                                                              optimizer,
                                                              scaler,
                                                              lr_scheduler,
                                                              history,
                                                              Loss,
                                                              logger,
                                                              lp_iters_per_epoch,
                                                              s_epoch,
                                                              writer,
                                                              args.dice_weight,
                                                              args.consistency_dice_weight,
                                                              args.cls_weight,
                                                              args.aux_weight,
                                                              args.n_aux_layers,
                                                              args.consistency_weight,
                                                              args.grad_accum,
                                                              warmup=(s_epoch == 0),
                                                              use_amp=args.use_amp,
                                                              check_data=args.check_data)
            if is_main_process:
                logger.info('{} Linear Probe TRAINING Epoch {:d}/{:d} | Loss : {:.3f}  Acc : {:.3f}%  mIoU : {:.3f}'.format(
                    datetime.now().time(), s_epoch + 1, args.lp_n_epoch, history['trainLoss'][-1], history['trainAcc'][-1] * 100, history['trainMIoU'][-1]))

        # Unfreeze all for full training
        for p in netEncoder.module.parameters():
            p.requires_grad = True
        # Rebuild optimizer with all params
        optimizer = optim(itertools.chain(*[netEncoder.module.parameters()]), args.max_lr, weight_decay=args.weight_decay)
        # Rebuild main scheduler for full training
        actual_iters_per_epoch = (args.iter_epoch + args.grad_accum - 1)// args.grad_accum
        lr_scheduler = utils.Warmup_cos_lr(args.max_lr, args.min_lr, actual_iters_per_epoch, args.n_epoch, 1)

    # Standard warmup before full training (kept for backward compatibility)
    netEncoder, optimizer, history = train.trainEpoch(trainLoader,
                                                                    netEncoder,
                                                                    optimizer,
                                                                    scaler,
                                                                    lr_scheduler,
                                                                    history,
                                                                    Loss,
                                                                    logger,
                                                                    args.iter_epoch,
                                                                    0,
                                                                    writer,
                                                                    args.dice_weight,
                                                                    args.consistency_dice_weight,
                                                                    args.cls_weight, 
                                                                    args.aux_weight,
                                                                    args.n_aux_layers,
                                                                    args.consistency_weight,
                                                                    args.grad_accum, 
                                                                    warmup=True,
                                                                    use_amp=args.use_amp,
                                                                    check_data=args.check_data)
    if is_main_process:
        msg = '{} \t Warmup {:d} iters |  \t '.format(datetime.now().time(), args.iter_epoch)    
        logger.info(msg)

for epoch in range(resume_start_epoch, args.n_epoch) : 
    if distributed and hasattr(trainLoader, 'sampler'):
        try:
            trainLoader.sampler.set_epoch(epoch)
        except Exception:
            pass
    netEncoder, optimizer, history = train.trainEpoch(trainLoader,
                                                                      netEncoder,
                                                                      optimizer,
                                                                      scaler,
                                                                      lr_scheduler,
                                                                      history,
                                                                      Loss,
                                                                      logger,
                                                                      args.iter_epoch,
                                                                      epoch,
                                                                      writer,
                                                                      args.dice_weight,
                                                                      args.consistency_dice_weight,
                                                                      args.cls_weight, 
                                                                      args.aux_weight,
                                                                      args.n_aux_layers,
                                                                      args.consistency_weight,
                                                                      args.grad_accum, 
                                                                      warmup=False,
                                                                      use_amp=args.use_amp,
                                                                      check_data=args.check_data)
    if is_main_process:
        msg = '{} TRAINING Epoch {:d} | Loss : {:.3f}  Acc : {:.3f}%  mIoU : {:.3f}'.format(datetime.now().time(), epoch, history['trainLoss'][-1], history['trainAcc'][-1] * 100, history['trainMIoU'][-1])
        logger.info(msg)

    if is_main_process:
        netEncoder, val_history = train.evalEpoch(valLoader,
                                              netEncoder,
                                              val_history,
                                              Loss,
                                              args.iter_epoch_val,
                                              epoch,
                                              writer,
                                              args.n_aux_layers,
                                              use_amp=args.use_amp,
                                              check_data=args.check_data)
        msg = '{} VAL Epoch {:d} | MIoU : {:.3f} MIoU_480 : {:.3f}'.format(
            datetime.now().time(), epoch, val_history['valMIoU'][-1], val_history['valMIoU_480'][-1])
        logger.info(msg)

    if is_main_process:
        netEncoder, test_history = train.testEpoch(testLoader,
                                                   netEncoder,
                                                   test_history,
                                                   args.iter_epoch_val,
                                                   epoch,
                                                   writer,
                                                   use_amp=args.use_amp,
                                                   warmup=False)
        msg = '{} TEST Epoch {:d} | MIoU: {:.3f} | MIoU_480: {:.3f}'.format(
            datetime.now().time(), epoch, test_history['testMIoU'][-1], test_history['testMIoU_480'][-1])
        logger.info(msg)
        
        utils.save_model(netEncoder, optimizer, os.path.join(args.out_dir, 'netLast.pth'), epoch, best_val_miou, best_test_miou)

    if is_main_process:
        current_val_miou, current_test_miou = val_history['valMIoU'][-1], test_history['testMIoU'][-1]
        if current_val_miou > best_val_miou:
            logger.info(f"Val mIoU improved from {best_val_miou:.4f} to {current_val_miou:.4f}!")
            best_val_miou = current_val_miou
            utils.save_model(netEncoder, optimizer, os.path.join(args.out_dir, 'best_val_miou.pth'), epoch, best_val_miou, best_test_miou)
        if current_test_miou > best_test_miou:
            logger.info(f"Test mIoU improved from {best_test_miou:.4f} to {current_test_miou:.4f}!")
            best_test_miou = current_test_miou
            utils.save_model(netEncoder, optimizer, os.path.join(args.out_dir, 'best_test_miou.pth'), epoch, best_val_miou, best_test_miou)
        
    # swanlab.log({"train_mIoU": history['trainMIoU'][-1],
    #            "train_loss": history['trainLoss'][-1], 
    #            "train_acc": history['trainAcc'][-1], 
    #            "val_mIoU": val_history['valMIoU'][-1], 
    #            "val_mIoU_480": val_history['valMIoU_480'][-1],
    #            "test_mIoU": test_history['testMIoU'][-1],
    #            "test_mIoU_480": test_history['testMIoU_480'][-1]})

    if is_main_process:
        with open(os.path.join(args.out_dir, 'history.json'), 'w') as f :
            json.dump(history, f)

if 'LOCAL_RANK' in os.environ:
    dist.barrier()
    dist.destroy_process_group()

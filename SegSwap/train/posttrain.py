# coding=utf-8
import json
import itertools
import os 

import option
args = option.get_option()

import torch
device = torch.device('cuda')

from datetime import datetime
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler
import wandb

import dataloader # dataloader
import train # train
import utils
import csegmentor

torch.cuda.empty_cache()
utils.set_seed(42)

run = wandb.init(project="origin_posttrain", config=args)
writer = SummaryWriter(os.path.join(args.out_dir, 'tb_logs'))
logger = utils.get_logger(args.out_dir)
logger.info(args)

netEncoder = csegmentor.ConditionalSegmentationModel(args.feat_extractor,  
                                                     args.extractor_depth,  
                                                     args.backbone_size,  
                                                     args.image_size,
                                                     args.upsampler,
                                                     args.backbone_type,
                                                     args.n_aux_layers)

netEncoder.load_state_dict(torch.load(args.resume_path)['encoder'])
netEncoder.cuda()
actual_iters_per_epoch = (args.iter_epoch + args.grad_accum - 1)// args.grad_accum

trainLoader, _ , _ = dataloader.getDataloader(args.image_size,
                                              args.train_dir,
                                              args.data_dir,
                                              args.batch_size,
                                              0,
                                              args.use_data,
                                              prob_neg=0.5)

history = {'posttrainVA':[], 'posttrainLoss':[]}

optim = torch.optim.AdamW
optimizer = optim(itertools.chain(*[netEncoder.cls_branch.parameters()]), args.max_lr, weight_decay=args.weight_decay)
lr_scheduler = utils.Warmup_cos_lr(args.max_lr, args.min_lr, actual_iters_per_epoch, args.posttrain_epoch, 1)
Loss = torch.nn.BCEWithLogitsLoss()
scaler = GradScaler(enabled=args.use_amp)

netEncoder, optimizer, history = train.posttrainEpoch(trainLoader,
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
                                                        args.aux_weight,
                                                        args.n_aux_layers,
                                                        args.grad_accum, 
                                                        warmup=True,
                                                        use_amp=args.use_amp)
msg = '{} \t Warmup {:d} iters |  \t '.format(datetime.now().time(), args.iter_epoch)    
logger.info(msg)

for epoch in range(args.posttrain_epoch) : 
    netEncoder, optimizer, history = train.posttrainEpoch(trainLoader,
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
                                                            args.aux_weight,
                                                            args.n_aux_layers,
                                                            args.grad_accum, 
                                                            warmup=False,
                                                            use_amp=args.use_amp)
    msg = '{} POSTTRAINING Epoch {:d} | Balanced accuracy: {:.3f} | Loss : {:.3f}'.format(datetime.now().time(), epoch, history['posttrainVA'][-1], history['posttrainLoss'][-1])
    logger.info(msg)

    utils.save_model(netEncoder, optimizer, os.path.join(args.out_dir, 'netLast_posttrain.pth'))
        
    wandb.log({
               "posttrain_loss": history['posttrainLoss'][-1], 
               "posttrain_va": history['posttrainVA'][-1]})

    with open(os.path.join(args.out_dir, 'history_posttrain.json'), 'w') as f :
        json.dump(history, f)
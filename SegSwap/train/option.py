import argparse
import os

def get_option() : 
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--n-epoch', type=int, default=200, help='nb of epochs') 
    parser.add_argument('--iter-epoch', type=int, default=3200, help='iteration of each train epoch')
    parser.add_argument('--iter-epoch-val', type=int, default=100, help='iteration of each val epoch')
    
    parser.add_argument('--max-lr', type=float, default=1e-5, help='max learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='min learning rate')
    parser.add_argument('--batch-size', type=int, default=2, help='nb of samples in a batch size')
    parser.add_argument('--gpu', type=str, default='0', help='gpu devices')

    parser.add_argument('--out-dir', type=str, help='output directory')
    parser.add_argument('--train-dir', type=str, nargs='+', default=['../true_data'], help='training data directory(must with nargs=+)')
    parser.add_argument('--data-dir', type=str, nargs='+', default=['../true_data'], help='training data directory(must with nargs=+)')
    parser.add_argument('--prob-neg', type=float, default=-0.1, help='probability of negative samples')

    parser.add_argument('--backbone-size', type=str, choices=['base', 'large','giant'], default='large', help='different vit models')
    parser.add_argument('--backbone-type', type=str, choices=['mae', 'deit3', 'dinov2', 'dinov3'], default='dinov2', help='backbone type')
    parser.add_argument('--vit-pretrained-pth', type=str, default='../model/dinov2_vitl14_pretrain.pth', help='pretrained vit model path')
    parser.add_argument('--num-register-tokens', type=int, default=0, help='number of register tokens for dinov2 backbone')

    parser.add_argument('--feat-extractor', type=str, choices=['cn2_nano', 'cn2_tiny', 'cn2_base', 'dinov3_cn_tiny', 'dinov3_cn_small', 'dinov3_cn_base', 'dinov3_cn_large'], default='cn2_base', help='different feature extractors for query')
    parser.add_argument('--extractor-pretrain-pth', type=str, default='../model/convnextv2_base_22k_384_ema.pt', help='pretrained weight different feature extractors for query')

    parser.add_argument('--extractor-depth', type=int, choices=[1, 2, 3], default=2, help='which feature to extract: (W/8, H/8), (W/16, H/16) or (W/32, H/32)')

    parser.add_argument('--image-size', type=int, default=518, help='training and testing resolution for images')

    parser.add_argument('--weight-decay', type=float, default=1e-2, help='weight decay')
    parser.add_argument('--use-data', type=str, nargs='+', default=['egoexo', 'exoego', 'egoego', 'exoexo'], help='the data used for training and validation')
    parser.add_argument('--upsampler', type=str, choices=['bilinear', 'convex'], default='bilinear', help='using which upsampler function')
    parser.add_argument('--grad-accum', type=int, default=16, help='gradient accumulation steps')

    parser.add_argument('--dice-weight', type=float, default=5, help='the weight of dice loss in loss calculation')
    parser.add_argument('--consistency-dice-weight', type=float, default=0, help='the weight of consistency dice loss in loss calculation')
    parser.add_argument('--cls-weight', type=float, default=1, help='the weight of classification loss in loss calculation')
    parser.add_argument('--aux-weight', type=float, default=1, help='the weight of auxiliary loss in loss calculation')
    parser.add_argument('--n-aux-layers', type=int, default=1, help='the number of layers contributing to auxiliary loss in loss calculation')
    parser.add_argument('--consistency-weight', type=float, default=10, help='the weight of consistency loss in loss calculation')
    
    parser.add_argument('--check-data', action='store_true', help='use data check or not')
    parser.add_argument('--use-amp', action='store_true', help='use Mixed Precision Training or not')

    # linear probing training controls
    parser.add_argument('--lp-n-epoch', type=int, default=0, help='number of epochs for linear probing; 0 disables even if enabled')
    parser.add_argument('--lp-iter-epoch', type=int, default=0, help='iterations per epoch in linear probing; 0 means use iter-epoch')
    parser.add_argument('--lp-max-lr', type=float, default=None, help='max learning rate for linear probing stage; default follows max-lr')
    parser.add_argument('--lp-min-lr', type=float, default=None, help='min learning rate for linear probing stage; default follows min-lr')

    parser.add_argument('--resume-path', type=str, help='resume path')
    parser.add_argument('--resume-start-epoch', type=int, default=0, help='resume start epoch')
    parser.add_argument('--posttrain-epoch', type=int, default=30, help='posttrain epoch')
    
    args = parser.parse_args()

    # In distributed (torchrun) runs, avoid overriding CUDA_VISIBLE_DEVICES
    if 'LOCAL_RANK' not in os.environ and args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    print (args)
    return args
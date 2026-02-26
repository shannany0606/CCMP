import json
import argparse
import random

import torch

def make_pairs(data_dir, split, setting):
    
    with open(f'{data_dir}/split.json', 'r') as fp:
        splits = json.load(fp)

    split_takes = splits[split]

    pairs=[]
    
    for take in split_takes:
        try:
            with open(f'{data_dir}/{take}/annotation.json', 'r') as fp:
                annotation = json.load(fp) 
        except:
            print(f"{take} annotation.json does not exist!!!")
            continue

        for obj_name in annotation['masks']:
            
            cams = {}
            for cam in annotation['masks'][obj_name]:
                if 'aria' in cam:
                    cams['ego'] = cam 
                else:
                    cams['exo'] = cam
                    
            if 'ego' not in cams or 'exo' not in cams:
                continue
              
            source_cam = cams[setting[:3]]
            target_cam = cams[setting[3:]]
            source_indices = list(annotation['masks'][obj_name].get(source_cam, {}).keys())
            target_indices = list(annotation['masks'][obj_name].get(target_cam, {}).keys())
            
            # create pairs of different-time frames
            if split != 'test' and split != 'val':
                for src_idx in source_indices:
                    diff_target_indices = [idx for idx in target_indices if idx != src_idx]
                    if not diff_target_indices:
                        continue
                        
                    tgt_idx = random.choice(diff_target_indices)
                    
                    source_rgb_path = f'{data_dir}//{take}//{source_cam}//{obj_name}//rgb//{src_idx}'
                    target_rgb_path = f'{data_dir}//{take}//{target_cam}//{obj_name}//rgb//{tgt_idx}'
                    
                    pairs.append((source_rgb_path, target_rgb_path, False))
                    #insert negative samples
                    if torch.rand(1).item() < args.prob_neg:
                        pairs.append((source_rgb_path, target_rgb_path, True))
            
            # create pairs of same-time frames
            if setting in ['egoexo','exoego']:
                for idx in source_indices:
                    if idx not in target_indices:
                        continue
                    source_rgb_path = f'{data_dir}//{take}//{source_cam}//{obj_name}//rgb//{idx}'
                    target_rgb_path = f'{data_dir}//{take}//{target_cam}//{obj_name}//rgb//{idx}'
                    
                    pairs.append((source_rgb_path, target_rgb_path, False))

                    if torch.rand(1).item() < args.prob_neg:
                        pairs.append((source_rgb_path, target_rgb_path, True))

    print(f'{split} - {setting} - pairs: ', len(pairs))
    with open(f'{data_dir}/{split}_{setting}_pairs.json', 'w') as fp:
        json.dump(pairs, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--prob_neg', type=float, default=-0.1)
    args = parser.parse_args()

    for setting in ['egoexo','exoego','egoego','exoexo']:
        for split in ['train','val','test']:
            if split in ['val','test'] and setting in ['egoego','exoexo']:
                continue
            make_pairs(args.data_dir, split, setting)
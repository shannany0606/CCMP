import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import cv2
import PIL.Image as Image
import os 
import random
import numpy as np 
import torch.nn.functional as F
import json

from pycocotools import mask as mask_utils
import albumentations as A
from albumentations.pytorch import ToTensorV2

import warnings
warnings.filterwarnings("ignore")

import copy

IMAGE_INTERPOLATE=cv2.INTER_LINEAR
MASK_INTERPOLATE=cv2.INTER_LINEAR 

def rescale_pad(img, size=(480, 480), interpolate=cv2.INTER_LINEAR):
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

    temp = cv2.resize(temp, size, interpolation=interpolate)

    return temp

def central_padding(img, size=480):
    C = 1
    if len(img.shape) == 2:
        img = img[..., None]
    else:
        H, W, C = img.shape
    
    temp = np.zeros((size, size, C), dtype=img.dtype)

    L = (size - 480) // 2
    temp[L:-L, L:-L] = img

    if C == 1:
        return temp.squeeze(-1)
    return temp

class ImageFolder(Dataset):

    def __init__(self,
                image_size,
                data_dir,
                transform,
                use_data=[],
                batch_size=4,
                iter_epoch_val=100,
                split='train',
                prob_neg=-0.1):
        
        self.data_dir = data_dir
        self.split = split
        self.use_data = use_data
        self.prob_neg = prob_neg
        if split == 'train':
            self.pairs = self._load_all_pairs(data_dir)
            self.nb_pair = len(self.pairs) 
        else:
            self.nb_pair = batch_size * iter_epoch_val
            self.pairs = self._load_all_pairs(data_dir, self.nb_pair) 
        self.mask_annotations = self._load_mask_annotations(data_dir)


        self.image_size = image_size
        self.batch_size = batch_size

        self.transform = transform

    def _load_all_pairs(self, data_dir, nb_pair=-1):
        pairs = []
        if self.split == 'train':
            settings = copy.deepcopy(self.use_data)
            for setting in settings:
                for d in data_dir:
                    pair_file = f'{self.split}_{setting}_pairs.json'
                    if os.path.exists(os.path.join(d, pair_file)):
                        with open(os.path.join(d, pair_file), 'r') as fp:
                            pairs.extend(json.load(fp))
                        print('LOADING: ', pair_file)
        else:
            for setting in ['egoexo', 'exoego']:
                for d in data_dir:
                    pair_file = f'{self.split}_{setting}_pairs.json'
                    if os.path.exists(os.path.join(d, pair_file)):
                        with open(os.path.join(d, pair_file), 'r') as fp:
                            pairs.extend(random.sample(json.load(fp), nb_pair//2))
                        print('LOADING: ', pair_file)

        random.shuffle(pairs)

        if nb_pair > 0: return pairs[:nb_pair]
        return pairs
    
    def _load_mask_annotations(self, data_dir):

        d = data_dir[0]
        with open(f'{d}/split.json', 'r') as fp:
            splits = json.load(fp)
        valid_takes = splits['train'] + splits['val'] + splits['test']

        annotations = {}
        for take in valid_takes:
            with open(f'{d}/{take}/annotation.json', 'r') as fp:
                annotations[take] = json.load(fp)

        return annotations

    def _get_negatives(self, take_id, obj_name):

        idx = random.randint(0, self.nb_pair-1)
        while take_id in self.pairs[idx][1] or obj_name in self.pairs[idx][1]:
            idx = random.randint(0, self.nb_pair-1)

        return self.pairs[idx][1]

    def _split_img_path(self, img_p):
        root, take_id, cam, obj, _type, idx = img_p.split('//')
        return take_id, cam, obj, idx

    def _get_mask(self, rle_obj):
        return mask_utils.decode(rle_obj)
    
    def load_pair(self, idx):
        root_dir = self.data_dir[0]

        img_pth1, img_pth2, negative = self.pairs[idx]
        if torch.rand(1).item() < self.prob_neg:
            negative = True
        take_id1, cam1, obj1, idx1 = self._split_img_path(img_pth1)
        take_id2, cam2, obj2, idx2 = self._split_img_path(img_pth2)
        
        mask_annotation1 = self.mask_annotations[take_id1] 
        mask_annotation2 = self.mask_annotations[take_id2] 

        mask1 = self._get_mask(mask_annotation1['masks'][obj1][cam1][idx1])

        if negative:
            img_pth2 = self._get_negatives(take_id2, obj2)
            take_id2, cam2, obj2, idx2 = self._split_img_path(img_pth2)
            mask2 = np.zeros((self.image_size, self.image_size))
            negative = True 
        else:
            mask2 = self._get_mask(mask_annotation2['masks'][obj2][cam2][idx2])
        
        vid_idx1 = int(idx1)
        vid_idx2 = int(idx2)

        if 'png_data' in root_dir:
            img1 = cv2.imread(f"{root_dir}/{take_id1}/{cam1}/{vid_idx1}.png")  
        else:
            img1 = cv2.imread(f"{root_dir}/{take_id1}/{cam1}/{vid_idx1}.jpg")  
        if 'png_data' in root_dir:
            img2 = cv2.imread(f"{root_dir}/{take_id2}/{cam2}/{vid_idx2}.png")  
        else:
            img2 = cv2.imread(f"{root_dir}/{take_id2}/{cam2}/{vid_idx2}.jpg")

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        pth1 = f"{take_id1}_{cam1}_{vid_idx1}"
        pth2 = f"{take_id2}_{cam2}_{vid_idx2}"

        return img1, mask1, img2, mask2, pth1, pth2, negative

    def __getitem__(self, idx):

        I1, FM1, I2, FM2, pth1, pth2, negative = self.load_pair(idx)

        FM1 = FM1.astype(np.float32)
        FM2 = FM2.astype(np.float32)

        if self.split == 'train':
            I1_scale = max((self.image_size + 1) / I1.shape[0], (self.image_size + 1) / I1.shape[1])
            I1 = cv2.resize(I1, (int(I1.shape[1] * I1_scale), int(I1.shape[0] * I1_scale)), interpolation=IMAGE_INTERPOLATE)
            FM1_scale = max((self.image_size + 1) / FM1.shape[0], (self.image_size + 1) / FM1.shape[1])
            FM1 = cv2.resize(FM1, (int(FM1.shape[1] * FM1_scale), int(FM1.shape[0] * FM1_scale)), interpolation=MASK_INTERPOLATE)

            I2_scale = max((self.image_size + 1) / I2.shape[0], (self.image_size + 1) / I2.shape[1])
            I2 = cv2.resize(I2, (int(I2.shape[1] * I2_scale), int(I2.shape[0] * I2_scale)), interpolation=IMAGE_INTERPOLATE)
            FM2_scale = max((self.image_size + 1) / FM2.shape[0], (self.image_size + 1) / FM2.shape[1])
            FM2 = cv2.resize(FM2, (int(FM2.shape[1] * FM2_scale), int(FM2.shape[0] * FM2_scale)), interpolation=MASK_INTERPOLATE) 
        else:
            I1 = rescale_pad(I1, (self.image_size, self.image_size), interpolate=IMAGE_INTERPOLATE)
            FM1 = rescale_pad(FM1, (self.image_size, self.image_size), interpolate=MASK_INTERPOLATE)
            I2 = rescale_pad(I2, (self.image_size, self.image_size), interpolate=IMAGE_INTERPOLATE)
            FM2 = rescale_pad(FM2, (self.image_size, self.image_size), interpolate=MASK_INTERPOLATE)
            
        augmented1 = self.transform(image=I1, mask=FM1)
        T1, FM1 = augmented1['image'], augmented1['mask']
        augmented2 = self.transform(image=I2, mask=FM2)
        T2, target2 = augmented2['image'], augmented2['mask']

        if (FM1.sum().item() < 1e-6):
            target2 = torch.zeros_like(target2, dtype=torch.float32)
            negative = True
        if (target2.sum().item() < 1e-6):
            negative = True

        target2_480 = F.interpolate(target2.unsqueeze(0).unsqueeze(0), size=(480, 480), mode='bilinear', align_corners=False)
        target2_480 = target2_480.squeeze(0).squeeze(0)

        return {'T1': T1,
                'T2': T2,
                'FM1': FM1.unsqueeze(0).type(torch.FloatTensor),
                'target2': target2.unsqueeze(0).type(torch.FloatTensor),
                'target2_480': target2_480.unsqueeze(0).type(torch.FloatTensor),
                'pth1': pth1,
                'pth2': pth2,
                'exist': float(not negative),
                }
                
    def __len__(self):
        return self.nb_pair

def TrainDataLoader(image_size,
                    img_dir_list,
                    batch_size,
                    iter_epoch_val,
                    use_data,
                    prob_neg,
                    sampler=None):
    transform = A.Compose([
        A.CropNonEmptyMaskIfExists(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], is_check_shapes=False)

    trainSet = ImageFolder(image_size,
                           img_dir_list,
                           transform,
                           use_data,
                           batch_size,
                           iter_epoch_val,
                           split='train',
                           prob_neg=prob_neg)
    valSet = ImageFolder(image_size,
                           img_dir_list,
                           transform,
                           use_data,
                           batch_size,
                           iter_epoch_val,
                           split='val',
                           prob_neg=prob_neg)
    concat_set = ConcatDataset([trainSet, valSet])
    if sampler is None:
        trainLoader = DataLoader(dataset=concat_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    else:
        trainLoader = DataLoader(dataset=concat_set, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=4, drop_last=True)

    return trainLoader

def ValTestDataLoader(image_size,
                    img_dir_list,
                    batch_size,
                    iter_epoch_val,
                    use_data,
                    split):
    transform = A.Compose([ 
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    ValTestSet = ImageFolder(image_size,
                             img_dir_list,
                             transform,
                             use_data,
                             batch_size,
                             iter_epoch_val,
                             split)
    valLoader = DataLoader(dataset=ValTestSet, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

    return valLoader

def getDataloader(image_size, train_dir_list, data_dir_list, batch_size, iter_epoch_val, use_data, prob_neg) : 
    
    sampler = None
    if dist.is_available() and dist.is_initialized():
        # Use a DistributedSampler for training in distributed mode.
        # Build a temporary dataset here to get length; the actual DataLoader construction
        # is delegated to TrainDataLoader.
        transform = A.Compose([
            A.CropNonEmptyMaskIfExists(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], is_check_shapes=False)

        trainSet = ImageFolder(image_size, train_dir_list, transform, use_data, batch_size, iter_epoch_val, split='train', prob_neg=prob_neg)
        valSet = ImageFolder(image_size, train_dir_list, transform, use_data, batch_size, iter_epoch_val, split='val', prob_neg=prob_neg)
        concat_set = ConcatDataset([trainSet, valSet])
        sampler = DistributedSampler(concat_set, shuffle=True, drop_last=True)

        # Rebuild DataLoader via TrainDataLoader (avoid duplicating transforms, etc.)
        trainLoader = TrainDataLoader(image_size, train_dir_list, batch_size, iter_epoch_val, use_data=use_data, prob_neg=prob_neg, sampler=sampler)
    else:
        trainLoader = TrainDataLoader(image_size, train_dir_list, batch_size, iter_epoch_val, use_data=use_data, prob_neg=prob_neg)

    valLoader = ValTestDataLoader(image_size, data_dir_list, batch_size, iter_epoch_val, use_data=use_data, split='val')
    testLoader = ValTestDataLoader(image_size, data_dir_list, batch_size, iter_epoch_val, use_data=use_data, split='test')

    return trainLoader, valLoader, testLoader

if __name__ == "__main__":
    image_size = 224
    train_dir_list = ['/data/EGO-EXO4D-RELATION/full_data']
    data_dir_list = ['../true_data']
    batch_size = 1
    use_data = ['egoexo', 'exoego', 'egoego', 'exoexo']
    trainLoader, valLoader, testLoader = getDataloader(image_size, train_dir_list, data_dir_list, batch_size, use_data)
    for batch in trainLoader: 
        print (batch['T1'].shape)
        print (batch['T2'].shape)
        print (batch['FM1'].shape)
        print (batch['target2'].shape)
        break
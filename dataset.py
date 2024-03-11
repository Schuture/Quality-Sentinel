import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class Clip_Rescale:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, ndarray):
        ndarray = ndarray.astype(np.float32)
        ndarray = np.clip(ndarray, a_min=self.min_val, a_max=self.max_val)
        ndarray = (ndarray - self.min_val) / (self.max_val - self.min_val)
        return ndarray

def crop_slices(slices, target_mask, min_margin=10, random_margin=False):

    rows = np.any(target_mask, axis=1)
    cols = np.any(target_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    if random_margin:
        max_r_margin = max(min_margin, (256-rmax+rmin)//2)
        max_c_margin = max(min_margin, (256-cmax+cmin)//2)
        r_margin = random.randint(min_margin, max_r_margin)
        c_margin = random.randint(min_margin, max_c_margin)
    else:
        r_margin = min_margin + 10
        c_margin = min_margin + 10
    
    rmin = max(rmin - r_margin, 0)
    rmax = min(rmax + r_margin, target_mask.shape[0])
    cmin = max(cmin - c_margin, 0)
    cmax = min(cmax + c_margin, target_mask.shape[1])

    # ensure we crop a square
    rlen, clen = rmax - rmin, cmax - cmin
    if rlen > clen:
        diff = rlen - clen
        cmin = max(cmin - diff // 2, 0)
        cmax = min(cmax + diff // 2, target_mask.shape[1])
    elif clen > rlen:
        diff = clen - rlen
        rmin = max(rmin - diff // 2, 0)
        rmax = min(rmax + diff // 2, target_mask.shape[0])

    cropped_slices = [s[rmin:rmax, cmin:cmax] for s in slices]

    return cropped_slices


class CTDataset(Dataset):
    def __init__(self, root_dir, transform_ct, transform_mask, mode='train', num_samples=10):
        self.root_dir = root_dir
        self.transform_ct = transform_ct
        self.transform_mask = transform_mask
        self.mode = mode
        
        self.sample_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        if mode == 'train':
            self.sample_dirs = self.sample_dirs[:num_samples]
        # self.sample_dirs = self.sample_dirs[:5]
        self.get_data_info()
        print('Data Amount: ', self.__len__())
        
    def __len__(self):
        return len(self.data_info)
    
    def get_data_info(self):
        # pre-load CT and ground truth mask
        self.ct_data = []
        self.gt_masks = []
        self.data_info = []  # element = [ct_and_gt_idx, pred_mask_path, class_idx]

        for i, sample_dir in enumerate(self.sample_dirs):
            ct_path = os.path.join(sample_dir, 'ct.nii.gz')
            gt_mask_path = os.path.join(sample_dir, 'label.nii.gz')
            print('CT data: ', ct_path)
            
            ct_img = sitk.ReadImage(ct_path)
            gt_mask_img = sitk.ReadImage(gt_mask_path)

            # SimpleITK to NumPy array
            ct_img = sitk.GetArrayFromImage(ct_img)
            gt_mask_img = sitk.GetArrayFromImage(gt_mask_img)
            
            self.ct_data.append(ct_img)
            self.gt_masks.append(gt_mask_img)
            
            for class_name in [c for c in sorted(os.listdir(sample_dir)) if not c.endswith('.gz')]:
                for mask_name in sorted(os.listdir(os.path.join(sample_dir, class_name))):
                    pred_mask_path = os.path.join(sample_dir, class_name, mask_name)
                    class_idx = int(class_name.split('_')[1])
                    self.data_info.append([i, pred_mask_path, class_idx])
    
    def __getitem__(self, idx):
        start = time.time()
        ct = self.ct_data[self.data_info[idx][0]]
        mask_class = self.data_info[idx][2]
        gt_mask = self.gt_masks[self.data_info[idx][0]]
        gt_mask = gt_mask == mask_class  # gt => [0, 1] mask

        pred_mask_path = self.data_info[idx][1]

        pred_mask = sitk.ReadImage(pred_mask_path)
        pred_mask = sitk.GetArrayFromImage(pred_mask)
        
        intersect_mask = np.logical_and(pred_mask, gt_mask)
        
        areas_intersect = np.sum(intersect_mask, axis=(1,2))
        
        valid_slices = np.where(areas_intersect >= 20)[0]

        if len(valid_slices) <= 1:  # only small overlap, select sample again
            return self.__getitem__(random.randint(0, self.__len__() - 1))
        slice_idx = np.random.choice(valid_slices)
        
        ct_slice = ct[slice_idx, :, :]
        gt_mask_slice = gt_mask[slice_idx, :, :]
        pred_mask_slice = pred_mask[slice_idx, :, :]
        
        random_margin = True if self.mode in ['train', 'test'] else False
        if self.mode != 'test':
            ct_slice, gt_mask_slice, pred_mask_slice = crop_slices(
                [ct_slice, gt_mask_slice, pred_mask_slice],
                pred_mask_slice,
                min_margin=10,
                random_margin=random_margin
            )
            
            numerator = np.sum(2.0 * gt_mask_slice * pred_mask_slice)
            denominator = (np.sum(gt_mask_slice) + np.sum(pred_mask_slice) + 1e-6)
            true_dice = numerator / denominator
            
            ct_slice = self.transform_ct(ct_slice)
            gt_mask_slice = self.transform_mask(gt_mask_slice.astype(np.float32))
            pred_mask_slice = self.transform_mask(pred_mask_slice.astype(np.float32))
            
            return ct_slice, pred_mask_slice, gt_mask_slice, torch.tensor(true_dice, dtype=torch.float32), mask_class
        else:
            ct_slices, gt_mask_slices, pred_mask_slices = torch.tensor([]), torch.tensor([]), torch.tensor([])
            for i in range(5):
                ct_slice_, gt_mask_slice_, pred_mask_slice_ = crop_slices(
                    [ct_slice, gt_mask_slice, pred_mask_slice],
                    pred_mask_slice,
                    min_margin=10,
                    random_margin=random_margin
                )
                
                numerator = np.sum(2.0 * gt_mask_slice * pred_mask_slice)
                denominator = (np.sum(gt_mask_slice) + np.sum(pred_mask_slice) + 1e-6)
                true_dice = numerator / denominator
                
                ct_slice_ = self.transform_ct(ct_slice_)
                gt_mask_slice_ = self.transform_mask(gt_mask_slice_.astype(np.float32))
                pred_mask_slice_ = self.transform_mask(pred_mask_slice_.astype(np.float32))
                
                ct_slices = torch.cat((ct_slices, ct_slice_.unsqueeze(0)), dim=0)
                gt_mask_slices = torch.cat((gt_mask_slices, gt_mask_slice_.unsqueeze(0)), dim=0)
                pred_mask_slices = torch.cat((pred_mask_slices, pred_mask_slice_.unsqueeze(0)), dim=0)
            
            return ct_slices, pred_mask_slices, gt_mask_slices, torch.tensor(true_dice, dtype=torch.float32), mask_class
        
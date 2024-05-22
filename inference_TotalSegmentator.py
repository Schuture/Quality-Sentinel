import os
import time
import pickle
import random
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import pandas as pd
from model import QualitySentinel

from dataset import Clip_Rescale, crop_slices

class_map = {  # from totalsegmentator => DAP Atlas embedding
 'adrenal_gland_left': 27,
 'adrenal_gland_right': 28,
 'aorta': 114,
 'autochthon_left': 40,
 'autochthon_right': 41,
 'brain': 25,
 'clavicula_left': 93,
 'clavicula_right': 94,
 'colon': 10,
 'duodenum': 9,
 'esophagus': 6,
 'face': 128,
 'femur_left': 103,
 'femur_right': 104,
 'gallbladder': 12,
 'gluteus_maximus_left': 32,
 'gluteus_maximus_right': 33,
 'gluteus_medius_left': 34,
 'gluteus_medius_right': 35,
 'gluteus_minimus_left': 36,
 'gluteus_minimus_right': 37,
 'heart_atrium_left': 106,
 'heart_atrium_right': 108,
 'heart_myocardium': 109,
 'heart_ventricle_left': 110,
 'heart_ventricle_right': 111,
 'hip_left': 100,
 'hip_right': 101,
 'humerus_left': 97,
 'humerus_right': 98,
 'iliac_artery_left': 112,
 'iliac_artery_right': 113,
 'iliac_vena_left': 115,
 'iliac_vena_right': 116,
 'iliopsoas_left': 38,
 'iliopsoas_right': 39,
 'inferior_vena_cava': 117,
 'kidney_left': 15,
 'kidney_right': 16,
 'liver': 13,
 'lung_lower_lobe_left': 120,
 'lung_lower_lobe_right': 122,
 'lung_middle_lobe_right': 123,
 'lung_upper_lobe_left': 121,
 'lung_upper_lobe_right': 124,
 'pancreas': 14,
 'portal_vein_and_splenic_vein': 118,
 'pulmonary_artery': 127,
 'rib_left_1': 67,
 'rib_left_2': 69,
 'rib_left_3': 71,
 'rib_left_4': 73,
 'rib_left_5': 75,
 'rib_left_6': 77,
 'rib_left_7': 79,
 'rib_left_8': 81,
 'rib_left_9': 83,
 'rib_left_10': 85,
 'rib_left_11': 87,
 'rib_left_12': 89,
 'rib_right_1': 68,
 'rib_right_2': 70,
 'rib_right_3': 72,
 'rib_right_4': 74,
 'rib_right_5': 76,
 'rib_right_6': 78,
 'rib_right_7': 80,
 'rib_right_8': 82,
 'rib_right_9': 84,
 'rib_right_10': 86,
 'rib_right_11': 88,
 'rib_right_12': 90,
 'sacrum': 102,
 'scapula_left': 95,
 'scapula_right': 96,
 'small_bowel': 8,
 'spleen': 26,
 'stomach': 7,
 'trachea': 126,
 'urinary_bladder': 19,
 'vertebrae_C1': 43,
 'vertebrae_C2': 44,
 'vertebrae_C3': 45,
 'vertebrae_C4': 46,
 'vertebrae_C5': 47,
 'vertebrae_C6': 48,
 'vertebrae_C7': 49,
 'vertebrae_L1': 62,
 'vertebrae_L2': 63,
 'vertebrae_L3': 64,
 'vertebrae_L4': 65,
 'vertebrae_L5': 66,
 'vertebrae_T1': 50,
 'vertebrae_T2': 51,
 'vertebrae_T3': 52,
 'vertebrae_T4': 53,
 'vertebrae_T5': 54,
 'vertebrae_T6': 55,
 'vertebrae_T7': 56,
 'vertebrae_T8': 57,
 'vertebrae_T9': 58,
 'vertebrae_T10': 59,
 'vertebrae_T11': 60,
 'vertebrae_T12': 61}

with open('label_embedding.pkl', 'rb') as file:
    embedding_dict = pickle.load(file)

def load_nii(path):
    try:
        ct_img = sitk.ReadImage(path)
        ct_img = sitk.GetArrayFromImage(ct_img)
        return ct_img
    except:
        return None

def inference_for_one_sample(ct_path, mask_paths, _classes, model,
                             device, transform_ct, transform_mask):
    start = time.time()
    GPU_time = 0
    crop_time = 0
    aug_time = 0
    ct_data = load_nii(ct_path)
    if ct_data is None:
        return ['n/a' for i in range(len(_classes))]
    dices = []

    for mask_path, _class in zip(mask_paths, _classes):
        mask_this_class = load_nii(mask_path).astype(np.float32)
        
        slice_dices = []
        
        valid_slices = np.where(np.any(mask_this_class > 0, axis=(1, 2)))[0]
        if len(valid_slices) > 10:  # sample 10 slices only
            valid_slices = [valid_slices[int(len(valid_slices)/10*idx)] for idx in range(10)]
        
        if len(valid_slices) == 0:
            print(f'Class {_class} has 0 slice.')
            dices.append('n/a')
            continue
        
        for slice_idx in valid_slices:
            crop_start = time.time()
            ct_slice, pred_mask_slice = crop_slices(
                [ct_data[slice_idx, :, :], mask_this_class[slice_idx, :, :]],
                mask_this_class[slice_idx, :, :]
            )
            crop_time += time.time() - crop_start
            
            aug_start = time.time()
            ct_slice = transform_ct(ct_slice).unsqueeze(0)
            pred_mask_slice = transform_mask(pred_mask_slice).unsqueeze(0)
            aug_time += time.time() - aug_start
            text_embedding = embedding_dict[class_map[_class]]
            
            image_tensor = torch.cat((ct_slice, pred_mask_slice), dim=1).to(device)
            embedding_tensor = text_embedding.to(device)
            
            GPU_start = time.time()
            predicted_dice = model(image_tensor, embedding_tensor)
            GPU_time += time.time() - GPU_start
            
            slice_dices.append(predicted_dice.detach().cpu().item())
            
            # visualize ct, mask, prediction to debug
            if visualize_seg_pred and slice_idx == valid_slices[len(valid_slices)//2]:
                plt.figure(figsize=(12,5), dpi=150)
                plt.subplot(121)
                plt.imshow(ct_slice.cpu().squeeze().permute(1, 0)*0.25+0.5, cmap='gray')
                plt.title(f'CT image (class {_class})')
                plt.subplot(122)
                plt.imshow(pred_mask_slice.cpu().squeeze().permute(1, 0)*0.5+0.5, cmap='gray')
                plt.title(f'Mask (pred dice {round(slice_dices[-1], 3)})')
                plt.savefig(f'inference_class_{_class}.png')
        
        dices.append(np.mean(slice_dices))
        print(f'Mean dice for class {_class} ({len(slice_dices)} slices) is {np.mean(slice_dices)}')
    
    mean_dice = np.mean([dice for dice in dices if dice != 'n/a'])
    print(f'\nMean dice for this sample is {mean_dice},'+\
          f'inference time: {round(time.time()-start, 3)}s, '+\
          f'crop time: {round(crop_time, 3)}s, '+\
          f'aug time: {round(aug_time, 3)}s, '+\
          f'GPU time: {round(GPU_time, 3)}s.')
    
    return dices

if __name__ == "__main__":
    visualize_seg_pred = False
    model_name = 'resnet50'
    model_path = "model.pth"
    data_dir = "./Totalsegmentator_dataset"
    output_csv = "inference_results_totalseg.csv"
    
    transform_ct = transforms.Compose([
        Clip_Rescale(min_val=-200, max_val=200),
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.25])
    ])

    transform_mask = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QualitySentinel(hidden_dim=50, backbone=model_name, embedding='text_embedding')
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    ct_samples = os.listdir(data_dir)  # sample names
    _classes = list(class_map.keys())  # class names

    # initialize csv file
    if not os.path.exists(output_csv):
        results = {"CT_Sample": [sample for sample in ct_samples]}
        for i in range(104):  # 104 classes
            results[_classes[i]] = [np.nan for _ in ct_samples]
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
    
    for i in range(len(ct_samples)):
        print(f'\nStart to inference [{i+1}/{len(ct_samples)}]: {ct_samples[i]}')
        
        ct_path = os.path.join(data_dir, ct_samples[i], 'ct.nii.gz')
        mask_paths = [os.path.join(data_dir, ct_samples[i], 'segmentations', f'{_class}.nii.gz') for _class in _classes]
        dices = inference_for_one_sample(ct_path, mask_paths, _classes, model,
                                         device, transform_ct, transform_mask)
        
        # real csv and update
        df = pd.read_csv(output_csv)
        for j, dice in enumerate(dices):
            df.loc[df["CT_Sample"] == ct_samples[i], _classes[j]] = dice
        df.to_csv(output_csv, index=False)








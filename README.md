# Quality-Sentinel
This is the repository of Quality Sentinel ([paper](https://arxiv.org/pdf/2406.00327)), a label (mask) quality evaluation tool for medical image segmentation, implemented with PyTorch. 
- Below shows labels with lower (left) and higher (right) predicted DSC in AbdomenAtlas dataset. 
- This model helps to diagnose the data quality in a large-scale CT image segmentation dataset.

![Illustration of Mask Quality Analysis on AbdomenAtlas](./figs/1721716857638.jpg)

### 1. Brief Introduction of this method

**Model input**: an image-label pair. The image is a 2D CT slice, the label is a binary mask.

**Model output**: estimates DSC of the mask to ground truths.

Two technical novelties:

1. The text-driven condition module embeds the organ names, serving as the conditional input of the model to recognize 142 different organs and improving the model performance.
2. The training of the model involves a compositional loss, combining optimal pair ranking and MSE, to align predicted with actual DSC.

### 2. Framework for Quality Sentinel

(1) Training Framework

![Framework](./figs/framework.png)

(2) Dataset Construction

Both the training and testing data are drawn from the DAP Atlas dataset featuring 142 organs. We fine-tuned the pretrained [STUNet](https://github.com/uni-medical/STU-Net) on the [DAP Atlas](https://github.com/alexanderjaus/AtlasDataset). Model checkpoints were saved at specified epochs: 10; 20; 30; 40; 50; 100; 200; 300; 400; 500. From each checkpoint, pseudo labels were generated, creating a dataset of CT scans paired with pseudo labels of varying quality and their corresponding ground truth DSC.

### 3. Quick Start

The Quality Sentinel dataset and the trained model are shared in ([Google Drive](https://drive.google.com/drive/folders/1KIenyE_OQ8rixHXPhscVPvK9sRrvpXwz?usp=sharing)). After downloading the zipped dataset, unzip it to this directory directly.

#### 3.1 Train the model

Run

```
python train.py
```

and it would read the data from ./Quality_Sentinel_data_50samples and save the model as best_resnet50_model_40_samples.pth.

#### 3.2 Inference on the TotalSegmentator dataset

Download TotalSegmentatorV1.0 dataset from [Zenodo](https://zenodo.org/records/6802614). Unzip the Totalsegmentator_dataset.zip to this directory directly, and run

```
python inference_TotalSegmentator.py
```

#### 3.3 Code for inference on a single 2D image-label pair

Follow the code below to do inference. The correspondence between \[_class\] and text embedding is in the [DAP_Atlas_label_name.csv](./DAP_Atlas_label_name.csv).

```
import torchvision.transforms as transforms
from model import QualitySentinel
from dataset import Clip_Rescale, crop_slices

with open('label_embedding.pkl', 'rb') as file:
    embedding_dict = pickle.load(file)

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

model = QualitySentinel(hidden_dim=50, backbone=model_name, embedding='text_embedding')
model.load_state_dict(torch.load("best_resnet50_model_40_samples.pth"))
model.to(device)
model.eval()

# ct_data shoule be a [D,H,W] 3D volume, mask_this_class should be the same size with binary values
ct_slice, pred_mask_slice = crop_slices(
    [ct_data[slice_idx, :, :], mask_this_class[slice_idx, :, :]],
    mask_this_class[slice_idx, :, :]
)

# ct_slice is a CT slice of original HU values
# pred_mask_slice is the 0/1 mask of the target
ct_slice = transform_ct(ct_slice).unsqueeze(0)
pred_mask_slice = transform_mask(pred_mask_slice).unsqueeze(0)

# find the text embedding of your target, _class is an integer key
text_embedding = embedding_dict[_class]

image_tensor = torch.cat((ct_slice, pred_mask_slice), dim=1).to(device)
embedding_tensor = text_embedding.to(device)

predicted_dice = model(image_tensor, embedding_tensor)
```


### 4. Results for Quality Sentinel

(1) The scatter plot of ground truth and predicted DSC on testing data, the high linear correlation coefficient demonstrates the performance of the model. 

<img src="./figs/scatter_plot.png" width = "500" height = "360" alt="The predicted DSC vs GT DSC" align=center />

(2) Human-in-the-Loop (active learning) results of label quality ranking methods on the TotalSegmentator. Quality Sentinel helps to reduce annotation costs, or improve the data efficiency.

<img src="./figs/active_learning.png" width = "800" height = "250" alt="Active Learning" align=center />

(3) Semi-supervised learning results of label quality ranking methods on the TotalSegmentator. Quality Sentinel outperforms all alternatives. It also significantly reduces quality estimation costs (6 times less time, 60 times less RAM, and 20,000 times less disk space compared to MC dropout) by employing a 2D model that evaluates only the output mask slices instead of extensive 3D computation.

<img src="./figs/semi-supervised_learning.png" width = "800" height = "150" alt="Semi-Supervised Learning" align=center />

### 5. Environment

The code is developed with Intel Xeon Gold 5218R CPU@2.10GHz and 8 Nvidia Quadro RTX 8000 GPUs.

The install script requirements.txt has been tested on an Ubuntu 20.04 system.

## Citation

If you find the benchmark method and results useful in your research, please consider citing:

```
@article{chen2024quality,
title={Quality Sentinel: Estimating Label Quality and Errors in Medical Segmentation Datasets},
author={Chen, Yixiong and Zhou, Zongwei and Yuille, Alan},
journal={arXiv preprint arXiv:2406.00327},
year={2024}
}
```














# Quality-Sentinel
This is the repository of Quality Sentinel, a label (mask) quality evaluation tool for medical image segmentation, implemented with PyTorch.

### 1. Brief Introduction of this method
The model takes an image-label pair of CT multi-organ segmentation as input and estimates DSC relative to ground truths.

Two technical novelties:

1. The text-driven condition module embeds the organ names, serving as the conditional input of the model to recognize 142 different organs and improving the model performance.
2. The training of the model involves a compositional loss, combining optimal pair ranking and MSE, to align predicted with actual DSC.

### 2. Framework for Quality Sentinel

(1) Training Framework

![Framework](./figs/framework.png)

(2) Dataset Construction

Both the training and testing data are drawn from the DAP Atlas dataset featuring 142 organs. We fine-tuned the pretrained [STUNet](https://github.com/uni-medical/STU-Net) on the [DAP Atlas](https://github.com/alexanderjaus/AtlasDataset). Model checkpoints were saved at specified epochs: 10; 20; 30; 40; 50; 100; 200; 300; 400; 500. From each checkpoint, pseudo labels were generated, creating a dataset of CT scans paired with pseudo labels of varying quality and their corresponding ground truth DSC.

### 3. Quick Start

#### 3.1 Train the model
```
python train.py
```

#### 3.2 Inference on the TotalSegmentator dataset
```
python inference_TotalSegmentator.py
```

#### 3.3 Code for inference on a single 2D image-label pair

First download [label embedding](https://drive.google.com/file/d/1re11nHwnBM8Cc_sE4IWD9FgIieggOFoT/view?usp=sharing) and [model checkpoint](https://drive.google.com/file/d/1-iDgjQB8nWtC-nEVLVmz3JbgkxfBF8eH/view?usp=sharing). And then follow the code below to do inference. The correspondence between \[_class\] and text embedding is in the DAP_Atlas_label_name.csv.

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

<img src="./figs/semi-supervised_learning.png" width = "800" height = "180" alt="Semi-Supervised Learning" align=center />



















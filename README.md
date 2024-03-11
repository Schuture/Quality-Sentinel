# Quality-Sentinel
This is the repository of Quality Sentinel, a label quality evaluation model for medical image segmentation, implemented with PyTorch.

### 1. Brief Introduction of this method
The model accepts an image-label pair of CT multi-organ segmentation as input and estimates DSC relative to ground truths.

The text-driven condition module embeds the organ names, serving as the conditional input of the model to recognize 142 different organs and improving the model performance. The training of the model involves a compositional loss, combining optimal pair ranking and MSE, to align predicted with actual DSC.

### 2. Framework for Quality Sentinel

![Framework](./figs/framework.png)

### 3. Regression Results for Quality Sentinel

The scatter plot of ground truth and predicted DSC on testing data, the high linear correlation coefficient demonstrates the performance of the model.
<img src="./figs/scatter_plot.png" width = "500" height = "360" alt="The predicted DSC vs GT DSC" align=center />







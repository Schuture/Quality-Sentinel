# Quality-Sentinel
This is the repository of Quality Sentinel, a label quality evaluation model for medical image segmentation, implemented with PyTorch.

### 1. Brief Introduction of this method
The model accepts an image-label pair of CT multi-organ segmentation as input and predicts the quality of this label. It employs a vision encoder to estimate DSC relative to ground truths.

The text-driven condition module embeds the organ names, serving as the conditional input of the model to recognize 142 different organs and improving the model performance. The training of the model involves a compositional loss, combining optimal pair ranking and MSE, to align predicted with actual DSC.

### 2. Framework for Quality Sentinel

![Framework](./figs/framework.png)

### 3. Regression Results for Quality Sentinel

![ScatterPlot](./figs/scatter_plot.png)






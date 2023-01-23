# 504_project

## Files-
### LoadDataset.py- 
Load Dataset from COCO and generate heatmaps and joint weights 
### Loss.py-
Implemented Adpative Heatmap Loss and Calculates PCK accuracy
### VIT_modules.py-
Patch Embedding, VIT backbone and Clasical Decoder model class implementation
### train.py-
Main file and train VIT decoder model

## Credits-
### LoadDataset.py- 
Majority of the functions are from ViTPose github- https://github.com/ViTAE-Transformer/ViTPose
### Loss.py-
We have implemented the loss function and PCK accuracy function is taken from VitPose github- https://github.com/ViTAE-Transformer/ViTPose
### VIT_modules.py-
We have implemented the model ourselves using "An image is worth 16x16 words" for VIT and "ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation" for Decoder as reference
### train.py-
We have implemented it ourselves.

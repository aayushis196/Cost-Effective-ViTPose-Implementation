# 504_project
This project presents a low computationally intensive deployment of ViTPose, which uses state-of-the-art Vision Transformers for a popular vision problem of pose estimation. The model, consisting of a Vision Transformer Network followed by a decoder, is implemented, and its parameters are adjusted with respect to a less image resolution of 128 x 128 and a smaller dataset. Particular design choices during the process are discussed, along with the decisions made, and an easily reproducible implementation is described. A final test accuracy of 32% is obtained on a dataset of around 110k images with a split in the trainvalidation-test set as 80-10-10.

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


import os
import cv2
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from torch import Tensor
from torch import nn, optim
from torchsummary import summary
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from Loss import AdaptiveWingLoss
from LoadDataset import LoadDataset
from VIT_Modules import ViTPose, ClassicDecoder  


if __name__ == "__main__":
    path = "/home/biped-lab/504_project/pretrain/mae_pretrain_vit_base.pth"
    checkpoint =   torch.load(path)
    state_dict = checkpoint['model']
    print(checkpoint.keys())
    print(state_dict.keys())
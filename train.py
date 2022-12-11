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

def train_ViTPose(
    pose,
    data_loader,
    learning_rate: float = 5e-3,
    weight_decay: float = 1e-4,
    max_iters: int = 5000,
    log_period: int = 20,
    num_epochs = 10,
    device: str = "cuda",
):
    """
    Train ViTPose. We use adamW optimizer and step decay.
    """
    HeatmapLoss =  AdaptiveWingLoss(use_target_weight=True)
    pose.to(device=device)
    

      # Optimizer: use adamW
      # Use SGD with momentum:
    optimizer = optim.adamW(params, lr=0.001, betas=(0.9, 0.999))

    for epoch in num_epochs:  

      # Keep track of training loss for plotting.
      train_loss_history = []
      val_loss_history= []
            # Each epoch has a training and validation phase
      for phase in ['train', 'val']:
          if phase == 'train':
              pose.train()  # Set model to training mode
          else:
              pose.eval()   # Set model to evaluate mode
      # detector.train()
        
          total_loss=torch.tensor(0.0)
          for _iter in range(max_iters):

              images, target_heatmap, t_h_weight  = next(iter(data_loader[phase]))           #Seperate for train and eval
              images = images.to(device)
              target_heatmap=target_heatmap.to(device)
              t_h_weight = rearrange(t_h_weight, "B C H W ->  B H C W")
              t_h_weight=t_h_weight.to(device)

              if phase=='Train':

                  pose.zero_grad()
                  
                  generated_heatmaps= pose(images)

                  # Dictionary of loss scalars.
                  losses = HeatmapLoss(images, target_heatmap, t_h_weight)

                  total_loss+=losses

                  losses.backward()
                  optimizer.step()

              elif phase=='val':
                  with torch.no_grad():
                     generated_heatmaps = pose(images)
                     losses = HeatmapLoss(images, target_heatmap, t_h_weight)

              total_loss+=losses

          # Print losses periodically.
          loss_str = f"[Epoch {epoch}][Phase {phase}][loss: {total_loss*images.size(0)/(max_iters):.3f}]"
          # for key, value in losses.items():
          #     loss_str += f"[{key}: {value:.3f}]"
              
          print(loss_str)

          if phase == 'train':
              train_loss_history.append(total_loss.item())  
          else:
              val_loss_history.append(total_loss.item())


    # Plot training loss.
    torch.save(model.state_dict(), model_save_path)
    plt.title("Training loss history")
    plt.xlabel(f"Iteration (x {log_period})")
    plt.ylabel("Loss")
    plt.plot(loss_history)
    plt.show()

def tensor_to_image(tensor):
    """
    Convert a torch tensor into a numpy ndarray for visualization.

    Inputs:
    - tensor: A torch tensor of shape (3, H, W) with
      elements in the range [0, 1]

    Returns:
    - ndarr: A uint8 numpy array of shape (H, W, 3)
    """
    tensor = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    ndarr = tensor.to("cpu", torch.uint8).numpy()
    return ndarr

if __name__ == "__main__":
    in_channels = 3
    patch_size = 16
    emb_size = 768
    img_size = (192, 256)
    heatmap_size = (48, 64)
    depth = 6                     #Depth of transformer layer
    kernel_size = (4,4)
    deconv_filter = 256
    out_channels = 17

    batch_size = 10

    learning_rate = 5e-3
    weight_decay = 1e-4
    max_iters = 500
    log_period = 20
    num_epochs = 10
    device = "cuda"

    img_directory = "/home/biped-lab/504_project/coco/images/train2017/"
    annotation_path = "/home/biped-lab/504_project/coco/annotations/person_keypoints_train2017.json"
    model_save_path = "/home/biped-lab/504_project/coco/model/"
    print(torch.__version__)
    print(torch.cuda.is_available())
    train = LoadDataset(img_directory, annotation_path,img_size, heatmap_size, test_mode = False)
    model = ViTPose(in_channels,patch_size,emb_size,img_size,depth,kernel_size,deconv_filter,out_channels)
    train_loader = torch.utils.data.DataLoader(dataset=train, 
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=4)  
    
    # tensor, target, weight =  next(iter(train_loader))
    # weight = rearrange(weight, "B C H W ->  B H C W")
    # print("Image size: ", tensor.shape)
    # print("heatmap size: ", target.shape)
    # print("Target weights: ",weight.shape)
    # image = tensor_to_image(tensor[4])
    # data = Image.fromarray(image)
    # data.show()
    
    train_ViTPose( model, train_loader, learning_rate, weight_decay, max_iters, log_period, num_epochs, device, model_save_path)
     

    
    
    
    






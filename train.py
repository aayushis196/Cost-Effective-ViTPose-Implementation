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
import torch.utils.data as data_utils
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from LoadDataset import LoadDataset
from VIT_Modules import ViTPose, ClassicDecoder
from Loss import AdaptiveWingLoss, pose_pck_accuracy
from Utility import tensor_to_image

def train_ViTPose(
    model,
    train_loader,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    max_iters: int = 500,
    log_period: int = 20,
    num_epochs = 10,
    device: str = "cuda",
    model_save_path: str  =  " " ):

    """
    Train ViTmodel We use adamW optimizer and step decay.
    """
    
    model.to(device=device)
    HeatmapLoss =  AdaptiveWingLoss(use_target_weight=True)
    

    #   Optimizer: use adamW
    #   Use SGD with momentum:
    optimizer = optim.AdamW(model.parameters(), learning_rate, betas=(0.9, 0.999))
    for epoch in range(num_epochs):  

      
        # Keep track of training loss for plotting.
        loss_history = []

        # detector.train()
        
        total_loss=torch.tensor(0.0).to(device)
        for _iter in range(max_iters):
            # Ignore first arg (image path) during training.
            images, target_heatmap, t_h_weight  = next(iter(train_loader))
            
            t_h_weight = rearrange(t_h_weight, "B C H W ->  B H C W")
            
            images = images.to(device)
            target_heatmap = target_heatmap.to(device)
            t_h_weight =  t_h_weight.to(device)
            
            model.train()
            model.zero_grad()

            generated_heatmaps = model(images)

            # Dictionary of loss scalars.
            losses = HeatmapLoss(generated_heatmaps, target_heatmap, t_h_weight)

            total_loss+=losses

            losses.backward()
            optimizer.step()
            if(_iter%10 == 0):
                _, avg_acc, _ = pose_pck_accuracy(generated_heatmaps.detach().cpu().numpy(),
                                                    target_heatmap.detach().cpu().numpy(),
                                                    t_h_weight.detach().cpu().squeeze(-1).numpy() > 0)

                loss_str = f"[Epoch {epoch}][ITER: {_iter}][loss: {losses:.8f}][Accuracy: {avg_acc:.8f}]" 
                # print("TH WERIGHT: ",t_h_weight[0])
                # im = tensor_to_image(images[0].detach().cpu())
                # data = Image.fromarray(im)
                # data.show()
                # break
                print(loss_str)          

        # Print losses periodically.
        loss_str = f"[Epoch {epoch}][loss: {total_loss*images.size(0)/(max_iters):.8f}]"
        # for key, value in losses.items():
        #     loss_str += f"[{key}: {value:.3f}]"

            
        print(loss_str)
        loss_history.append(total_loss.item())

    # Plot training loss.
    torch.save(model.state_dict(), model_save_path)
    plt.title("Training loss history")
    plt.xlabel(f"Iteration (x {log_period})")
    plt.ylabel("Loss")
    plt.plot(loss_history)
    plt.show()




if __name__ == "__main__":
    in_channels = 3
    patch_size = 16
    emb_size = 768
    img_size = (128,128)
    heatmap_size = ((img_size[0]//patch_size)*4, (img_size[1]//patch_size)*4)
    depth = 12                    #Depth of transformer layer
    kernel_size = (4,4)
    deconv_filter = 256
    out_channels = 17
    train_dataset_size = 30000
    batch_size = 50

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
    indices = torch.arange(train_dataset_size)
    train_30k= data_utils.Subset(train, indices)
    print("TAKING SUBSET OF THE CURRENT DATASET")
    print("New Dataset Size: ",train_30k.__len__())
    model = ViTPose(in_channels,patch_size,emb_size,img_size,depth,kernel_size,deconv_filter,out_channels)

    train_loader = torch.utils.data.DataLoader(dataset=train_30k, 
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=4)
      
    #print(summary(model,input_size=(in_channels, img_size[0], img_size[1])))
    tensor, target, weight =  next(iter(train_loader))
    # # weight = rearrange(weight, "B C H W ->  B H C W")
    # print("Image size: ", tensor[0].shape)
    # # print("heatmap size: ", target.shape)
    # # print("Target weights: ",weight.shape)
    image = tensor_to_image(tensor[0])
    data = Image.fromarray(image)
    data.show()
    print(weight[0])
    
    # train_ViTPose( model, train_loader, learning_rate, weight_decay, max_iters, log_period, num_epochs, device, model_save_path)
     

    
    
    
    






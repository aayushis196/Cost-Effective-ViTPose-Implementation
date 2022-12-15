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
    model_save_path: str  =  " ",
    use_checkpoint = False,
    val = False ):

    """
    Train ViTmodel We use adamW optimizer and step decay.
    """
    
    model.to(device=device)
    HeatmapLoss =  AdaptiveWingLoss(use_target_weight=True)
    

    #   Optimizer: use adamW
    #   Use SGD with momentum:
    optimizer = optim.AdamW(model.parameters(), learning_rate, betas=(0.9, 0.999))
    
    avg_acc = 0
   
    if val:
        num_epochs = 1
        trained_model = torch.load(os.path.join(model_save_path,"model_params1.pth"))
        model.load_state_dict(trained_model)
    
    if use_checkpoint:
        print("LOADING PRETRAINED WEIGHTS")
        trained_model = torch.load(os.path.join(model_save_path,"model_params1.pth"))
        model.load_state_dict(trained_model)
        print("Pretrained weights matched")
    try:
        for epoch in range(num_epochs): 

            iterator = iter(train_loader)
            # Keep track of training loss for plotting.
            loss_history = []

            # detector.train()
            
            total_loss=torch.tensor(0.0).to(device)
            total_acc = 0

            for _iter in range(max_iters):
                # Ignore first arg (image path) during training.
                
                if not val:
                    images, target_heatmap, t_h_weight  = next(iterator)
                                    
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

                    _, avg_acc, _ = pose_pck_accuracy(generated_heatmaps.detach().cpu().numpy(),
                                                            target_heatmap.detach().cpu().numpy(),
                                                            t_h_weight.detach().cpu().squeeze(-1).numpy() > 0)
                    total_acc += avg_acc

                    if(_iter%log_period == 0 and _iter != 0):
                        _, avg_acc, _ = pose_pck_accuracy(generated_heatmaps.detach().cpu().numpy(),
                                                            target_heatmap.detach().cpu().numpy(),
                                                            t_h_weight.detach().cpu().squeeze(-1).numpy() > 0)
                        
                        loss_str = f"[Epoch {epoch}][ITER: {_iter}][loss: {losses:.8f}][Accuracy: {total_acc/(_iter+1):.8f}]" 
                        print(loss_str)


                else:
                    # VALIDATION
                    with torch.no_grad():
                        images, target_heatmap, t_h_weight  = next(iterator)
                        
                        
                        t_h_weight = rearrange(t_h_weight, "B C H W ->  B H C W")
                    
                        images = images.to(device)
                        target_heatmap = target_heatmap.to(device)
                        t_h_weight =  t_h_weight.to(device)
                        
                        model.eval()
                        generated_heatmaps = model(images)
                            
                        _, avg_acc, _ = pose_pck_accuracy(generated_heatmaps.detach().cpu().numpy(),
                                                            target_heatmap.detach().cpu().numpy(),
                                                            t_h_weight.detach().cpu().squeeze(-1).numpy() > 0)
                    
                        total_acc += avg_acc
                        print("Accuracy : ", avg_acc)
                        if(_iter%log_period == 0 and _iter != 0): 

                            loss_str = f"[Epoch {epoch}][ITER: {_iter}][Accuracy: {total_acc/_iter:.8f}]" 
                            print(loss_str)

                
            

            if not val:    
                # Print losses periodically.
                loss_str = f"[Epoch {epoch}][loss: {total_loss*images.size(0)/(max_iters):.8f}]"
                # for key, value in losses.items():
                #     loss_str += f"[{key}: {value:.3f}]"           
                print(loss_str)
                loss_history.append(total_loss.item())
                
            else:
                print("Total_accuracy = ",total_acc/max_iter)
            
            if not val and (total_acc/max_iters) > 0.8:
                break

        print("------------MODEL TRAINNED SUCESSFULLY------------")
        torch.save(model.state_dict(),os.path.join(model_save_path,"model_params1.pth"))
        with open(os.path.join(model_save_path,"loss.txt"), "w") as output:
            output.write(str(loss_history))

    except KeyboardInterrupt:
        print("------------TRAINING SUSPENDED------------")
        torch.save(model.state_dict(),os.path.join(model_save_path,"model_params1.pth"))
        with open(os.path.join(model_save_path,"loss.txt"), "w") as output:
            output.write(str(loss_history))




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
    train_dataset_size = 90000
    val_dataset_size = 10000
    batch_size = 25

    learning_rate = 2e-3
    weight_decay = 1e-4
    train_max_iters = train_dataset_size//batch_size
    val_max_iters = val_dataset_size//batch_size

    log_period = 20
    num_epochs = 200
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
    train_loader = torch.utils.data.DataLoader(dataset=train_30k, 
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=4)

    model = ViTPose(in_channels,patch_size,emb_size,img_size,depth,kernel_size,deconv_filter,out_channels)

    train_ViTPose( model, train_loader, learning_rate, weight_decay, train_max_iters, log_period, num_epochs, device, model_save_path,use_checkpoint = True, val = False)
    
    
    # val_indices = torch.arange(50000, 60000)
    # val_10k= data_utils.Subset(train, val_indices)

    # val_loader = torch.utils.data.DataLoader(dataset=val_10k, 
    #                                             batch_size=batch_size,
    #                                             shuffle=False,
    #                                             num_workers=4)
    # print("Val Dataset Size: ",val_10k.__len__())

    # train_ViTPose( model, val_loader, learning_rate, weight_decay, val_max_iters, log_period, num_epochs, device, model_save_path,val = True)
    #print(summary(model,input_size=(in_channels, img_size[0], img_size[1])))
    # tensor, target, weight =  next(iter(val_loader))
    # # # weight = rearrange(weight, "B C H W ->  B H C W")
    # # print("Image size: ", tensor[0].shape)
    # # # print("heatmap size: ", target.shape)
    # # # print("Target weights: ",weight.shape)
    # image = tensor_to_image(tensor[5])
    # data = Image.fromarray(image)
    # data.show()
    # print(weight[5])
    
    
     

    
    
    
    






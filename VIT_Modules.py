import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import cv2
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import Tensor
from PIL import Image
from pycocotools.coco import COCO
import torchvision.transforms as transforms
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

pre_trained_weights_path = "/home/biped-lab/504_project/pretrain/mae_pretrain_vit_base.pth"
pre_trained_weights = torch.load(pre_trained_weights_path)
weights = pre_trained_weights["model"]
device = "cuda"

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: tuple = (224,224)):

        self.patch_size = patch_size
        self.img_size = img_size
        self.emb_size = emb_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
         
        # self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))
        # self.positions = nn.Parameter(torch.randn(int((img_size[0]//patch_size)*(img_size[1]//patch_size)) + 1, emb_size))
        
        # self.cls_token = nn.Parameter(weights["cls_token"])
        self.positions = self.reshape_position_embedding()
        
        
        self.projection.weight = nn.Parameter(weights["patch_embed.proj.weight"])
        self.projection.bias = nn.Parameter(weights["patch_embed.proj.bias"])

        
                
    def forward(self, x: Tensor) -> Tensor:
        b,_,_,_ = x.shape
        # print(x.shape)
        x = self.projection(x)
        # cls_tokens = repeat(self.cls_token,'() n e -> b n e',b=b)
        # x = torch.cat([cls_tokens,x], dim = 1)
        x  =  x + self.positions[:,1:] + self.positions[:,:1]
        return x

    def reshape_position_embedding(self):

        pos_embed_temp = nn.Parameter(weights["pos_embed"])
        num_patches = (self.img_size[0]//self.patch_size) * (self.img_size[1]//self.patch_size)
        original_size = int((pos_embed_temp.shape[-2] - 1)**0.5)
        patches_along_height = (self.img_size[0]//self.patch_size)
        patches_along_width =  (self.img_size[1]//self.patch_size)
        cls_token = pos_embed_temp[:,:1]
        pos_tokens = pos_embed_temp[:,1:]
        
        pos_tokens = pos_tokens.reshape(-1,original_size,original_size,self.emb_size).permute(0,3,1,2)

        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(patches_along_height, patches_along_width), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((cls_token, pos_tokens), dim=1)

        return new_pos_embed.to(device)


class MultiHeadAttention(nn.Module):
    def __init__(self,block, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

        #qkv_layer
        weight_str = "blocks." + str(block) + ".attn.qkv.weight"
        bias_str = "blocks." + str(block) + ".attn.qkv.bias" 
        self.qkv.weight = nn.Parameter(weights[weight_str])
        self.qkv.bias = nn.Parameter(weights[bias_str])
               
        #projection_layer
        weight_str = "blocks." + str(block) + ".attn.proj.weight"
        bias_str = "blocks." + str(block) + ".attn.proj.bias" 
        self.projection.weight = nn.Parameter(weights[weight_str])
        self.projection.bias = nn.Parameter(weights[bias_str])
      
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn, block, norm):
        super().__init__()
        self.fn = fn

        #norm layer
        weight_str = "blocks." + str(block) + ".norm"+ str(norm)+ ".weight"
        bias_str = "blocks." + str(block) + ".norm"+ str(norm)+ ".bias"
        self.fn[0].weights = nn.Parameter(weights[weight_str])
        self.fn[0].bias = nn.Parameter(weights[bias_str])
       
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Module):
    def __init__(self,block, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__()
        self.feed_forward_block = nn.Sequential(
                                    nn.Linear(emb_size, expansion * emb_size),
                                    nn.GELU(),
                                    nn.Dropout(drop_p),
                                    nn.Linear(expansion * emb_size, emb_size),
                                )
        #fc layer 1
        weight_str = "blocks." + str(block) + ".mlp.fc1.weight"
        bias_str = "blocks." + str(block) + ".mlp.fc1.bias"
        self.feed_forward_block[0].weight = nn.Parameter(weights[weight_str])
        self.feed_forward_block[0].bias = nn.Parameter(weights[bias_str])

        #fc layer 2
        weight_str = "blocks." + str(block) + ".mlp.fc2.weight"
        bias_str = "blocks." + str(block) + ".mlp.fc2.bias"
        self.feed_forward_block[3].weight = nn.Parameter(weights[weight_str])
        self.feed_forward_block[3].bias = nn.Parameter(weights[bias_str])
     
    def forward(self,input):
        output = self.feed_forward_block(input)
        return output

# class RearrangeOutput(nn.Module):
#     def __init__(self, img_size, patch_size):
#         super().__init__()
#         self.linear=nn.Linear((img_size[0]//patch_size)*(img_size[1]//patch_size)+1,(img_size[0]//patch_size)*(img_size[1]//patch_size))
#         self.img_size = img_size
#         self.patch_size=patch_size
        
        
#     def forward(self, x, **kwargs):
#         x = self.linear(x)
#         x = rearrange(x, "b (h n) d -> b h n d", h=self.img_size[0]//self.patch_size)
#         return x

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 block,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(

            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(block = block,
                                        emb_size = emb_size, 
                                        **kwargs),
                    nn.Dropout(drop_p)
                ),
                block=block, norm = 1
            ),
        
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(block = block,
                                    emb_size = emb_size, 
                                    expansion=forward_expansion, 
                                    drop_p=forward_drop_p),
                
                    nn.Dropout(drop_p)
                ),
                block= block, norm = 2 
            )
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(block = i,**kwargs) for i in range(depth)])

class ClassicDecoder(nn.Module):
    def __init__(self,     
                img_size,
                patch_size,
                in_channels: int = 768,
                kernel_size = (4,4),
                deconv_filter = 256,
                out_channels: int = 17,
                ):
    
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.deconv_layers = nn.Sequential(
                                nn.ConvTranspose2d(in_channels = in_channels,
                                                out_channels= deconv_filter,
                                                kernel_size=kernel_size,
                                                stride = 2,
                                                padding = 1,
                                                output_padding = 0),
                                nn.BatchNorm2d(deconv_filter),
                                nn.ReLU(inplace=True),

                                nn.ConvTranspose2d(in_channels = deconv_filter,
                                                out_channels= deconv_filter,
                                                kernel_size=kernel_size,
                                                stride=2,
                                                padding=1,
                                                output_padding=0),
                                nn.BatchNorm2d(deconv_filter),
                                nn.ReLU(inplace = True)
        )

        self.last_convolution = nn.Conv2d(in_channels = deconv_filter,
                                        out_channels=out_channels,
                                        kernel_size = 1,
                                        stride=1,
                                        padding=0)

    def forward(self,input):
        B,_,_= input.shape

        input = input.permute(0,2,1).reshape(B,-1,self.img_size[0]//self.patch_size,self.img_size[1]//self.patch_size)
        
        deconv_output = self.deconv_layers(input)
        
        output = self.last_convolution(deconv_output)
        
        return output

class ViT(nn.Sequential):
    def __init__(self,     
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: tuple = (192, 256),
                depth: int = 6,                     #Depth of transformer layer
                kernel_size = (4,4),
                deconv_filter: int = 256,
                out_channels: int = 17
                ):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size),
            
            # RearrangeOutput(img_size, patch_size),                               #Rearrange the output as per paper
            
        )

class ViTPose(nn.Module):
    def __init__(self,     
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: tuple = (192, 256),
                depth: int = 12,                     #Depth of transformer layer
                kernel_size = (4,4),
                deconv_filter: int = 256,
                out_channels: int = 17
                ):
        super().__init__()


        self.model = ViT(in_channels,patch_size,emb_size,img_size,depth,kernel_size,deconv_filter,out_channels)
        self.decoder = ClassicDecoder(img_size,patch_size,emb_size, kernel_size, deconv_filter, out_channels)
    def forward(self,image):
        out = self.model(image)
        output = self.decoder(out)
        return output
       
        

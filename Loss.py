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


class AdaptiveWingLoss(nn.Module):
    """Adaptive wing loss. paper ref: 'Adaptive Wing Loss for Robust Face
    Alignment via Heatmap Regression' Wang et al. ICCV'2019.
    Args:
        alpha (float), omega (float), epsilon (float), theta (float)
            are hyper-parameters.
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self,
                 alpha=2.1,
                 omega=14,
                 epsilon=1,
                 theta=0.5,
                 use_target_weight=False,
                 loss_weight=1.):
        super().__init__()
        self.alpha = float(alpha)
        self.omega = float(omega)
        self.epsilon = float(epsilon)
        self.theta = float(theta)
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def criterion(self, pred, target):
        """Criterion of Wingless Adaptive Loss. Paper Ref: Wang, Bo- "Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression"
        Args:
            pred (torch.Tensor[NxKxHxW]): Predicted heatmaps.
            target (torch.Tensor[NxKxHxW]): Ground Truth heatmaps. K is number of keypoints and N is batch size
        
        """
        N, K, H, W = pred.shape
        self.delta = torch.abs(target - pred)
        
        A = self.omega*(1/(1+torch.pow(self.theta/self.epsilon, self.alpha-target)))*torch.pow(self.theta/self.epsilon, self.alpha-target-1)*(self.alpha-target)/self.epsilon
        C = self.theta*A-self.omega*torch.log(1+torch.pow(self.theta/self.epsilon, self.alpha-target))
        losses = torch.where( self.delta < self.theta, self.omega*torch.log(1+ torch.pow(self.delta/self.epsilon, self.alpha-target)), A*self.delta - C)
        return torch.mean(losses)

    def forward(self, output, target, target_weight):
        """Forward function.
        Args:
            output (torch.Tensor[NxKxHxW]): Output heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            loss = self.criterion(output* target_weight ,               
                                  target* target_weight)                  
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight


def _calc_distances(preds, targets, mask, normalize):
    """Calculate the distances between model preds and target( ground truth). 
        We then normalize the distance by the heatmap size.
    Note:
        batch_size: N
        num_keypoints: K
        dimension of keypoints/joints: D=2 
    Args:
        preds (np.ndarray[N, K, D]): Predicted keypoint location.
        targets (np.ndarray[N, K, D]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        normalize (np.ndarray[N, D]): Typical value is heatmap_size
    Returns:
        np.ndarray[K, N]: The normalized distances. 
            If target keypoints are missing, the distance is -1.
    """
    N, K, D = preds.shape
    # set mask=0 when normalize==0
    normalized_mask = mask.copy()
    normalized_mask[np.where((normalize == 0).sum(1))[0], :] = 0
    distances = np.full((N, K), -1, dtype=np.float32)
    # handle invalid values
    normalize[np.where(normalize <= 0)] = 1e6

    distances[normalized_mask.squeeze()] = np.linalg.norm(
        ((preds - targets) / normalize[:, None, :])[normalized_mask.squeeze()], axis=-1)
    return distances.T


def _get_max_preds(heatmaps):
    """Get keypoint predictions from heat maps.
    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W
    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
    Returns:
        tuple: A tuple containing aggregated results.
        - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        - confidence (np.ndarray[N, K, 1]): Confidence of the keypoints.
    """
    N, K, H, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    confidence = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(confidence, (1, 1, 2)) > 0.0, preds, -1)
    return preds, confidence

def pose_pck_accuracy(output, target, mask, thr=0.05, normalize=None):
    """ Credit : VitPose- https://github.com/ViTAE-Transformer/ViTPose
    Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints from heatmaps.
    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.
        - batch_size: N
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W
    Args:
        output (np.ndarray[N, K, H, W]): prediction heatmaps.
        target (np.ndarray[N, K, H, W]): Groundtruth heatmaps.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation. Default 0.05.
        normalize (np.ndarray[N, 2]): Normalization factor for H&W.
    Returns:
        tuple: A tuple containing keypoint accuracy.
        - np.ndarray[K]: Accuracy of each keypoint.
        - float: Averaged accuracy across all keypoints.
        - int: Number of valid keypoints.
    """
    N, K, H, W = output.shape
    if K == 0:
        return None, 0, 0
    if normalize is None:
        normalize = np.tile(np.array([[H, W]]), (N, 1))

    pred, _ = _get_max_preds(output)
    gt, _ = _get_max_preds(target)
    distances = _calc_distances(pred, gt, mask, normalize)
    
    accuracy = np.array()
    for d in distances:
        distance_valid = distances != -1
        num_distance_valid = torch.sum(distance_valid)
        if num_distance_valid > 0:
            accuracy.append(torch.sum(distances[distance_valid] < thr) / num_distance_valid) 
        else : 
            accuracy.append(-1)
    valid_accuracy = accuracy[accuracy >= 0]
    if len(valid_accuracy) > 0 :
        avg_accuracy = valid_accuracy.mean() 
    else:
        avg_accuracy = 0
    return accuracy, avg_accuracy, len(valid_accuracy)

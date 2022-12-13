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
        """Criterion of wingloss.
        Note:
            batch_size: N
            num_keypoints: K
        Args:
            pred (torch.Tensor[NxKxHxW]): Predicted heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
        """
        H, W = pred.shape[2:4]
        delta = (target - pred).abs()

        A = self.omega * (                                                        #A, and C make loss fn continuous and smooth at condition described below
            1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - target))
        ) * (self.alpha - target) * (torch.pow(
            self.theta / self.epsilon,
            self.alpha - target - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(
            1 + torch.pow(self.theta / self.epsilon, self.alpha - target))

        losses = torch.where(
            delta < self.theta,
            self.omega *
            torch.log(1 +
                      torch.pow(delta / self.epsilon, self.alpha - target)),
            A * delta - C)                                         # Adaptive wing loss from the paper

        return torch.mean(losses)

    def forward(self, output, target, target_weight):
        """Forward function.
        Note:
            batch_size: N
            num_keypoints: K
        Args:
            output (torch.Tensor[NxKxHxW]): Output heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            loss = self.criterion(output.to("cuda") * target_weight ,                                   #.unsqueeze(-1),
                                  target .to("cuda")* target_weight)                  #.unsqueeze(-1))
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight


def _calc_distances(preds, targets, mask, normalize):
    """Calculate the normalized distances between preds and target.
    Note:
        batch_size: N
        num_keypoints: K
        dimension of keypoints: D (normally, D=2 or D=3)
    Args:
        preds (np.ndarray[N, K, D]): Predicted keypoint location.
        targets (np.ndarray[N, K, D]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        normalize (np.ndarray[N, D]): Typical value is heatmap_size
    Returns:
        np.ndarray[K, N]: The normalized distances. \
            If target keypoints are missing, the distance is -1.
    """
    N, K, _ = preds.shape
    # set mask=0 when normalize==0
    _mask = mask.copy()
    _mask[np.where((normalize == 0).sum(1))[0], :] = False
    distances = np.full((N, K), -1, dtype=np.float32)
    # handle invalid values
    normalize[np.where(normalize <= 0)] = 1e6
    # print("Mask shape: ",_mask.shape)
    # print("ANS: ", np.linalg.norm(((preds - targets) / normalize[:, None, :]), axis=-1).shape)
    # print("Preds: ",preds[0])
    # print("target: ",targets[0])
    # print("visibility: ", _mask.squeeze())
    distances[_mask.squeeze()] = np.linalg.norm(
        ((preds - targets) / normalize[:, None, :])[_mask.squeeze()], axis=-1)
    return distances.T


def _distance_acc(distances, thr=0.5):
    """Return the percentage below the distance threshold, while ignoring
    distances values with -1.
    Note:
        batch_size: N
    Args:
        distances (np.ndarray[N, ]): The normalized distances.
        thr (float): Threshold of the distances.
    Returns:
        float: Percentage of distances below the threshold. \
            If all target keypoints are missing, return -1.
    """
    distance_valid = distances != -1
    num_distance_valid = distance_valid.sum()
    if num_distance_valid > 0:
        return (distances[distance_valid] < thr).sum() / num_distance_valid
    return -1


def _get_max_preds(heatmaps):
    """Get keypoint predictions from score maps.
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
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals

def pose_pck_accuracy(output, target, mask, thr=0.05, normalize=None):
    """Calculate the pose accuracy of PCK for each individual keypoint and the
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
        output (np.ndarray[N, K, H, W]): Model output heatmaps.
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
    return keypoint_pck_accuracy(pred, gt, mask, thr, normalize)


def keypoint_pck_accuracy(pred, gt, mask, thr, normalize):
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints for coordinates.
    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.
        - batch_size: N
        - num_keypoints: K
    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation.
        normalize (np.ndarray[N, 2]): Normalization factor for H&W.
    Returns:
        tuple: A tuple containing keypoint accuracy.
        - acc (np.ndarray[K]): Accuracy of each keypoint.
        - avg_acc (float): Averaged accuracy across all keypoints.
        - cnt (int): Number of valid keypoints.
    """
    distances = _calc_distances(pred, gt, mask, normalize)

    acc = np.array([_distance_acc(d, thr) for d in distances])
    valid_acc = acc[acc >= 0]
    cnt = len(valid_acc)
    avg_acc = valid_acc.mean() if cnt > 0 else 0
    return acc, avg_acc, cnt













# def _calc_distances(preds, targets, mask, normalize):
#     """Calculate the normalized distances between preds and target.
#     Note:
#         batch_size: N
#         num_keypoints: K
#         dimension of keypoints: D (normally, D=2 or D=3)
#     Args:
#         preds (np.ndarray[N, K, D]): Predicted keypoint location.
#         targets (np.ndarray[N, K, D]): Groundtruth keypoint location.
#         mask (np.ndarray[N, K]): Visibility of the target. False for invisible
#             joints, and True for visible. Invisible joints will be ignored for
#             accuracy calculation.
#         normalize (np.ndarray[N, D]): Typical value is heatmap_size
#     Returns:
#         np.ndarray[K, N]: The normalized distances. \
#             If target keypoints are missing, the distance is -1.
#     """
#     N, K, _ = preds.shape
#     # set mask=0 when normalize==0
#     _mask = mask.clone()
#     _mask[torch.where((normalize == 0).sum(1))[0], :] = False
#     # _mask = torch.tile(_mask,(N,K,2))
#     distances = torch.full((N, K), -1, dtype=torch.float32)
#     # handle invalid values
#     normalize[torch.where(normalize <= 0)] = 1e6
#     distances[_mask.squeeze()] = torch.linalg.norm(((preds - targets)/ normalize[:, None, :])[_mask.squeeze()], axis=-1)
#     return torch.transpose(distances, 0, 1)

#     # N, K, _ = preds.shape
#     # # set mask=0 when normalize==0
#     # _mask = mask.copy()
#     # _mask[np.where((normalize == 0).sum(1))[0], :] = False
#     # distances = np.full((N, K), -1, dtype=np.float32)
#     # # handle invalid values
#     # normalize[np.where(normalize <= 0)] = 1e6
#     # distances[_mask] = np.linalg.norm(
#     #     ((preds - targets) / normalize[:, None, :])[_mask], axis=-1)
#     # return distances.T


# def _get_max_preds(heatmaps):
#     """Get keypoint predictions from score maps.
#     Note:
#         batch_size: N
#         num_keypoints: K
#         heatmap height: H
#         heatmap width: W
#     Args:
#         heatmaps (torch.tensor[N, K, H, W]): model predicted heatmaps.
#     Returns:
#         tuple: A tuple containing aggregated results.
#         - preds (torch.tensor[N, K, 2]): Predicted keypoint location.
#         - maxvals (torch.tensor[N, K, 1]): Scores (confidence) of the keypoints.
#     """

#     N, K, _, W = heatmaps.shape
#     heatmaps_reshaped = heatmaps.reshape((N, K, -1))
#     idx = torch.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
#     maxvals = torch.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

#     preds = torch.tile(idx, (1, 1, 2)).to(torch.float32)
#     preds[:, :, 0] = preds[:, :, 0] % W
#     preds[:, :, 1] = torch.div(preds[:, :, 1], W, rounding_mode='floor')

#     preds = torch.where(torch.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
#     return preds, maxvals

#     # N, K, _, W = heatmaps.shape
#     # heatmaps_reshaped = heatmaps.reshape((N, K, -1))
#     # idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
#     # maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

#     # preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
#     # preds[:, :, 0] = preds[:, :, 0] % W
#     # preds[:, :, 1] = preds[:, :, 1] // W

#     # preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
#     # return preds, maxvals

# def _distance_acc(distances, thr=0.5):
#     """Return the percentage below the distance threshold, while ignoring
#     distances values with -1.
#     Note:
#         batch_size: N
#     Args:
#         distances (np.ndarray[N, ]): The normalized distances.
#         thr (float): Threshold of the distances.
#     Returns:
#         float: Percentage of distances below the threshold. \
#             If all target keypoints are missing, return -1.
#     """
#     distance_valid = distances != -1
#     num_distance_valid = distance_valid.sum()
#     if num_distance_valid > 0:
#         return (distances[distance_valid] < thr).sum() / num_distance_valid
#     return -1


# def pose_pck_accuracy(output, target, mask, thr=0.05,normalize = None):
#     """Calculate the pose accuracy of PCK for each individual keypoint and the
#     averaged accuracy across all keypoints from heatmaps.
#     Note:
#         PCK metric measures accuracy of the localization of the body joints.
#         The distances between predicted positions and the ground-truth ones
#         are typically normalized by the bounding box size.
#         The threshold (thr) of the normalized distance is commonly set
#         as 0.05, 0.1 or 0.2 etc.
#         - batch_size: N
#         - num_keypoints: K
#         - heatmap height: H
#         - heatmap width: W
#     Args:
#         output (torch.tensor[N, K, H, W]): Model output heatmaps.
#         target (torch.tensor[N, K, H, W]): Groundtruth heatmaps.
#         mask (torch.tensor[N, K]): Visibility of the target. False for invisible
#             joints, and True for visible. Invisible joints will be ignored for
#             accuracy calculation.
#         thr (float): Threshold of PCK calculation. Default 0.05.
#         normalize (torch.tensor[N, 2]): Normalization factor for H&W.
#     Returns:
#         tuple: A tuple containing keypoint accuracy.
#         - torch.tensor[K]: Accuracy of each keypoint.
#         - float: Averaged accuracy across all keypoints.
#         - int: Number of valid keypoints.
#     """
#     N, K, H, W = output.shape
#     if K == 0:
#         return None, 0, 0
#     if normalize is None:
#         normalize = torch.tile(torch.Tensor([[H, W]]), (N, 1))

#     pred, _ = _get_max_preds(output)
#     gt, _ = _get_max_preds(target)
#     return keypoint_pck_accuracy(pred, gt, mask, thr, normalize)
    
#     # N, K, H, W = output.shape
#     # if K == 0:
#     #     return None, 0, 0
#     # if normalize is None:
#     #     normalize = np.tile(np.array([[H, W]]), (N, 1))

#     # pred, _ = _get_max_preds(output)
#     # gt, _ = _get_max_preds(target)
#     # return keypoint_pck_accuracy(pred, gt, mask, thr, normalize)

# def keypoint_pck_accuracy(pred, gt, mask, thr, normalize):
#     """Calculate the pose accuracy of PCK for each individual keypoint and the
#     averaged accuracy across all keypoints for coordinates.
#     Note:
#         PCK metric measures accuracy of the localization of the body joints.
#         The distances between predicted positions and the ground-truth ones
#         are typically normalized by the bounding box size.
#         The threshold (thr) of the normalized distance is commonly set
#         as 0.05, 0.1 or 0.2 etc.
#         - batch_size: N
#         - num_keypoints: K
#     Args:
#         pred (torch.tensor[N, K, 2]): Predicted keypoint location.
#         gt (torch.tensor[N, K, 2]): Groundtruth keypoint location.
#         mask (torch.tensor[N, K]): Visibility of the target. False for invisible
#             joints, and True for visible. Invisible joints will be ignored for
#             accuracy calculation.
#         thr (float): Threshold of PCK calculation.
#         normalize (torch.tensor[N, 2]): Normalization factor for H&W.
#     Returns:
#         tuple: A tuple containing keypoint accuracy.
#         - acc (torch.tensor[K]): Accuracy of each keypoint.
#         - avg_acc (float): Averaged accuracy across all keypoints.
#         - cnt (int): Number of valid keypoints.
#     """
#     distances = _calc_distances(pred, gt, mask, normalize)
#     acc = torch.Tensor([_distance_acc(d, thr) for d in distances])
#     valid_acc = acc[acc >= 0]
#     cnt = len(valid_acc)
#     avg_acc = valid_acc.mean() if cnt > 0 else 0
#     return acc, avg_acc, cnt


#     # distances = _calc_distances(pred, gt, mask, normalize)

#     # acc = np.array([_distance_acc(d, thr) for d in distances])
#     # valid_acc = acc[acc >= 0]
#     # cnt = len(valid_acc)
#     # avg_acc = valid_acc.mean() if cnt > 0 else 0
#     # return acc, avg_acc, cnt

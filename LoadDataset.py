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



class LoadDataset(Dataset):
    """Class for loading the COCO dataset with keypoints for pose detection"""
    def __init__(self, img_directory, annotation_path, image_size, heatmap_size, test_mode = False):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            img_directory: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.img_directory = img_directory
        self.coco = COCO(annotation_path)
        self.img_ids = self.coco.getImgIds()
        self.num_joints = 17
        self.use_gt_bbox = False  #why are we using GT boxes and bounding boxes? 
        self.bbox_file = "/content/drive/MyDrive/EECS 504 Project/Dataset/annotations/person_keypoints_train2017.json"      # ?? Apparently for testing we use bounding boxes and for training we use keypoints: BB are used in the loss function, distances are normalized based on the size of bb. So they should be used in test as well
        self.det_bbox_thr = 00 #0.0
        self.use_nms =  True           #True
        self.soft_nms = False
        self.nms_thr = 1.0
        self.oks_thr = 0.9    #What is oks_threshold?
        self.vis_thr = 0.2    #what is vis_threshold?
        self.joint_weights = [1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5]
        self.test_mode = test_mode
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.transform_to_tensor = transforms.Compose([transforms.ToTensor()])
        self.transform = transforms.Compose([
                            # to-tensor
                            transforms.ToTensor(),
                            # resize
                            transforms.Resize((self.image_size[0],self.image_size[1])),
                            # normalize
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                        ])
                        
        self.id2name,self.name2id = self._get_mapping_id_name()

        self.db = self._get_db()
        self.num_images = len(self.img_ids)
        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')
        
    def _get_db(self):
      """Load dataset."""
      if (not self.test_mode) or self.use_gt_bbox:
          # use ground truth bbox
          #Train mode
          gt_db = self._load_coco_keypoint_annotations()
      else:
          # use bbox from detection
          # Test mode
          gt_db = self._load_coco_person_detection_results()
      return gt_db

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
     
        
            
        flag = False
        image_data = self.db[index]
        img_path = image_data['image_file']
        if(index == 0):
            flag = True
            print("Joints visible: ", image_data["joints_3d_visible"])

        image_data["image"]  = Image.open(os.path.normpath(img_path)).convert('RGB')

        image_data["image"], image_data["joints_3d"] = self.get_affine_transform(image_data)
        image = self.transform(image_data["image"])

        target, target_weight = self._udp_generate_target(image_data["joints_3d"], image_data["joints_3d_visible"], 0.5,flag)
        
        target = self.transform_to_tensor(target)
        target_weight = self.transform_to_tensor(target_weight)

        return image, target, target_weight

    def __len__(self):
        return len(self.img_ids)

    def _load_coco_keypoint_annotations(self):
        """Ground truth bbox and keypoints."""
        gt_db = []
        for img_id in self.img_ids:
            gt_db.extend(self._load_coco_keypoint_annotation_kernel(img_id))
        return gt_db

    def _load_coco_keypoint_annotation_kernel(self, img_id):
      """load annotation from COCOAPI.
      Note:
          bbox:[x1, y1, w, h]
      Args:
          img_id: coco image id
      Returns:
          dict: db entry
      """
      img_ann = self.coco.loadImgs(img_id)[0]
      width = img_ann['width']
      height = img_ann['height']
      num_joints = self.num_joints
    
      ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
      objs = self.coco.loadAnns(ann_ids)

      # sanitize bboxes
      valid_objs = []
      for obj in objs:
          if 'bbox' not in obj:
              continue
          x, y, w, h = obj['bbox']
          x1 = max(0, x)
          y1 = max(0, y)
          x2 = min(width - 1, x1 + max(0, w - 1))
          y2 = min(height - 1, y1 + max(0, h - 1))
          if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
              obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
              valid_objs.append(obj)
      objs = valid_objs

      bbox_id = 0
      rec = []
      for obj in objs:
          if 'keypoints' not in obj:
              continue
          if max(obj['keypoints']) == 0:
              continue
          if 'num_keypoints' in obj and obj['num_keypoints'] == 0:
              continue
          joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
          joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

          keypoints = np.array(obj['keypoints']).reshape(-1, 3)
          joints_3d[:, :2] = keypoints[:, :2]
          joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

          center, scale = self._xywh2cs(*obj['clean_bbox'][:4])  ##why are we converting bounding box to two params?

          image_file = os.path.join(self.img_directory, self.id2name[img_id]) 
          rec.append({
              'image_file': image_file,
              'center': center,
              'scale': scale,
              'bbox': obj['clean_bbox'][:4],
              'rotation': 0,
              'joints_3d': joints_3d,
              'joints_3d_visible': joints_3d_visible,
              'bbox_score': 1,
              'bbox_id': bbox_id
          })
          bbox_id = bbox_id + 1

      return rec
      
    def _get_mapping_id_name(self):
        """
        Args:
            imgs (dict): dict of image info.
        Returns:
            tuple: Image name & id mapping dicts.
            - id2name (dict): Mapping image id to name.
            - name2id (dict): Mapping image name to id.
        """
        imgs = self.coco.imgs
        id2name = {}
        name2id = {}
        for image_id, image in imgs.items():
            file_name = image['file_name']
            id2name[image_id] = file_name
            name2id[file_name] = image_id

        return id2name, name2id

    def _xywh2cs(self, x, y, w, h, padding=1.25):

        """This encodes bbox(x,y,w,h) into (center, scale)
        Args:
            x, y, w, h (float): left, top, width and height
            padding (float): bounding box padding factor
        Returns:
            center (np.ndarray[float32](2,)): center of the bbox (x, y).
            scale (np.ndarray[float32](2,)): scale of the bbox w & h.
        """
        aspect_ratio = self.image_size[0] / self.image_size[1]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

        if (not self.test_mode) and np.random.rand() < 0.3:
            center += 0.4 * (np.random.rand(2) - 0.5) * [w, h]

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        # pixel std is 200.0
        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
        # padding to include proper amount of context
        scale = scale * padding

        return center, scale
    def get_affine_transform(self, image_data):
        
        rotation = image_data["rotation"]
        center = image_data["center"]
        scale = image_data["scale"]
        img = image_data["image"]
        joints_3d = image_data["joints_3d"]
        
        trans = self.get_warp_matrix(rotation,
                                    center * 2.0,
                                    [self.image_size[0] - 1.0, self.image_size[1] - 1.0],
                                    scale * 200.0)
        
        if not isinstance(img, list):
            img = cv2.warpAffine(
                np.array(img),
                trans, (int(self.image_size[0]), int(self.image_size[1])),
                flags=cv2.INTER_LINEAR)
        else:
            img = [
                cv2.warpAffine(
                    i,
                    trans, (int(self.image_size[0]), int(self.image_size[1])),
                    flags=cv2.INTER_LINEAR) for i in np.array(img)
            ]

            joints_3d[:, 0:2] = \
                self.warp_affine_joints(joints_3d[:, 0:2].copy(), trans)
        
        return img, joints_3d     

    def get_warp_matrix(self, theta, size_input, size_dst, size_target):
        """Calculate the transformation matrix under the constraint of unbiased.
        Paper ref: Huang et al. The Devil is in the Details: Delving into Unbiased
        Data Processing for Human Pose Estimation (CVPR 2020).

        Args:
            theta (float): Rotation angle in degrees.
            size_input (np.ndarray): Size of input image [w, h].
            size_dst (np.ndarray): Size of output image [w, h].
            size_target (np.ndarray): Size of ROI in input plane [w, h].

        Returns:
            np.ndarray: A matrix for transformation.
        """
        theta = np.deg2rad(theta)
        matrix = np.zeros((2, 3), dtype=np.float32)
        scale_x = size_dst[0] / size_target[0]
        scale_y = size_dst[1] / size_target[1]
        matrix[0, 0] = math.cos(theta) * scale_x
        matrix[0, 1] = -math.sin(theta) * scale_x
        matrix[0, 2] = scale_x * (-0.5 * size_input[0] * math.cos(theta) +
                                0.5 * size_input[1] * math.sin(theta) +
                                0.5 * size_target[0])
        matrix[1, 0] = math.sin(theta) * scale_y
        matrix[1, 1] = math.cos(theta) * scale_y
        matrix[1, 2] = scale_y * (-0.5 * size_input[0] * math.sin(theta) -
                                0.5 * size_input[1] * math.cos(theta) +
                                0.5 * size_target[1])
        return matrix


    def warp_affine_joints(self,joints, mat):
        """Apply affine transformation defined by the transform matrix on the
        joints.

        Args:
            joints (np.ndarray[..., 2]): Origin coordinate of joints.
            mat (np.ndarray[3, 2]): The affine matrix.

        Returns:
            np.ndarray[..., 2]: Result coordinate of joints.
        """
        joints = np.array(joints)
        shape = joints.shape
        joints = joints.reshape(-1, 2)
        return np.dot(
            np.concatenate((joints, joints[:, 0:1] * 0 + 1), axis=1),
            mat.T).reshape(shape)


    def _udp_generate_target(self, joints_3d, joints_3d_visible, sigma,flag):
        """Generate the target heatmap via 'UDP' approach. Paper ref: Huang et
        al. The Devil is in the Details: Delving into Unbiased Data Processing
        for Human Pose Estimation (CVPR 2020).
        Note:
            - num keypoints: K
            - heatmap height: H
            - heatmap width: W
            - num target channels: C
            - C = K if target_type=='GaussianHeatmap'
            - C = 3*K if target_type=='CombinedTarget'
        Args:
            cfg (dict): data config
            joints_3d (np.ndarray[K, 3]): Annotated keypoints.
            joints_3d_visible (np.ndarray[K, 3]): Visibility of keypoints.
            factor (float): kernel factor for GaussianHeatmap target or
                valid radius factor for CombinedTarget.
            target_type (str): 'GaussianHeatmap' or 'CombinedTarget'.
                GaussianHeatmap: Heatmap target with gaussian distribution.
                CombinedTarget: The combination of classification target
                (response map) and regression target (offset map).
        Returns:
            tuple: A tuple containing targets.
            - target (np.ndarray[C, H, W]): Target heatmaps.
            - target_weight (np.ndarray[K, 1]): (1: visible, 0: invisible)
        """
        num_joints = self.num_joints
        image_size = self.image_size
        heatmap_size = self.heatmap_size
        joint_weights = self.joint_weights
        
        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_3d_visible[:, 0]
       
        target = np.zeros((num_joints, heatmap_size[1], heatmap_size[0]),
                            dtype=np.float32)

        tmp_size = sigma * 3

        # prepare for gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, None]
       
        for joint_id in range(num_joints):

            feat_stride_x = (image_size[0] - 1.0) / (heatmap_size[0] - 1.0)
            feat_stride_y = (image_size[1] - 1.0) / (heatmap_size[1] - 1.0)
            mu_x = int(joints_3d[joint_id][0] / feat_stride_x + 0.5)
            mu_y = int(joints_3d[joint_id][1] / feat_stride_y + 0.5)

            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

            if flag: print(f"Joint id: {joint_id}, x,y:{joints_3d[joint_id]}, feat stride:{feat_stride_x}, mu_x:{mu_x}, mu_y:{mu_y}, br:{br}, heatmap_size:{heatmap_size}")

            if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                if flag: print("HERE, jointid: ",joint_id)
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            mu_x_ac = joints_3d[joint_id][0] / feat_stride_x
            mu_y_ac = joints_3d[joint_id][1] / feat_stride_y
            x0 = y0 = size // 2
            x0 += mu_x_ac - mu_x
            y0 += mu_y_ac - mu_y
            g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return np.moveaxis(target,0,-1), target_weight

    
    # def collate_fn(self, data):
    #     """Creates mini-batch tensors from the list of tuples (image, caption).
        
    #     We should build custom collate_fn rather than using default collate_fn, 
    #     because merging caption (including padding) is not supported in default.
    #     Args:
    #         data: list of tuple (image, caption). 
    #             - image: torch tensor of shape (3, 256, 256).
    #             - caption: torch tensor of shape (?); variable length.
    #     Returns:
    #         images: torch tensor of shape (batch_size, 3, 256, 256).
    #         targets: torch tensor of shape (batch_size, padded_length).
    #         lengths: list; valid length for each padded caption.
    #     """
    #     # images = data[0]
    #     # targets = data[1]
    #     # target_weights = data[2]
    #     # get image as a tensor
    #     print("HERE!!")
    #     # print(data[0])
    #     print(len(data))
    #     img_paths = data['image_file']
    #     batch = len(img_paths)
    #     images = torch.Tensor(batch, 3,self.image_size[0], self.image_size[1])
    #     targets = torch.Tensor(batch, self.num_joints, self.heatmap_size[0],self.heatmap_size[1])
    #     target_weights = torch.Tensor(batch, self.num_joints, 1)
    #     img_list = []
    #     tgt_list = []
    #     tw_list = []

    #     for i in range(len(img_paths)):
    #        image = Image.open(os.path.normpath(img_paths[i])).convert('RGB')
    #        image = self.transform(image)
    #        target,target_weight = self._udp_generate_target(data["joints_3d"][i], data["joints_3d_visible"][i], 2)
    #        target = self.transform_to_tensor(target)
    #        target_weight = self.transform_to_tensor(target_weight)
    #        img_list.append(image)
    #        tgt_list.append(target)
    #        tw_list.append(target_weight)

    #     # print(img_list[0].shape)
    #     torch.cat(img_list, out=images)
    #     torch.cat(tgt_list, out=targets)
    #     torch.cat(tw_list, out=target_weights)
        
    #     return images, targets, target_weights
    # def get_loader(self,batch_size, shuffle, num_workers):
    #     """Returns torch.utils.data.DataLoader for custom coco dataset."""

    #     data_loader = torch.utils.data.DataLoader(dataset=self.db, 
    #                                             batch_size=batch_size,
    #                                             shuffle=shuffle,
    #                                             num_workers=num_workers,
    #                                             collate_fn=self.collate_fn)
    #     return data_loader
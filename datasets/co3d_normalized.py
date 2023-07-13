import os
import json
import gzip

import matplotlib.pyplot as plt
import cv2
import numpy as np
import imageio
import yaml
from PIL import Image
from torch.utils.data import Dataset

from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras, get_world_to_view_transform
import torch
from pytorch3d.transforms import matrix_to_quaternion
from pytorch3d.renderer.implicit.raysampling import NDCMultinomialRaysampler
from datasets.bird import default_loader
import random
from PIL import ImageFilter, ImageOps
import torchvision
from geo_utils import get_cam_transform, get_relative_cam_transform, mat2quat, transform_relative_pose
import pytorch3d
from pytorch3d.structures.pointclouds import Pointclouds
from pytorch3d.io import load_ply
from utils import pytorch3d_to_kaolin
import pickle


class Co3DSeqNormalizedDataset(Dataset):
    def __init__(self, root, mode, num_views=5, categories=["toyplane"], sample_mode="random"):
        self.root = root
        self.num_views = num_views
        self.mode = mode
        self.sample_mode = sample_mode
        self.scene_paths = []
        for cat in categories:
            scene_ids = os.listdir(os.path.join(root, cat))
            cur_scene_paths = [os.path.join(cat, scene_id) for scene_id in scene_ids]

            split_ratio = 0.8
            split_ind = int(len(cur_scene_paths) * split_ratio)
            if self.mode == "train":
                cur_scene_paths = cur_scene_paths[:split_ind]
                # self.scene_paths = self.scene_paths[:50] * (len(self.scene_paths) // 50)
            elif self.mode == "val":
                # self.scene_paths = self.scene_paths[:split_ind]
                cur_scene_paths = cur_scene_paths[split_ind:]

            self.scene_paths.extend(cur_scene_paths)

        self.sampled_frame_ids = None

        # self.scene_paths = ["toyplane/403_52953_103399"] * len(self.scene_paths)
        # self.scene_paths = self.scene_paths[:50] * (len(self.scene_paths) // 50)

    def __len__(self):
        return len(self.scene_paths)

    def preprocess_co3d(self, img, m):
        transform = torchvision.transforms.RandomAffine(5, translate=(0.02, 0.02))

        m = (m[..., None] > 0.4)
        ones = (np.ones(img.shape) * 255).astype(np.uint8)
        t_rgb = img * m + ones * (1 - m)
        t_rgb = torch.from_numpy(t_rgb)
#         t_rgb = self.transform(t_rgb.permute(2, 0, 1)).permute(1, 2, 0)
        t_rgb = t_rgb / 255.0
        
        t_s = torch.from_numpy(m).float()
#         t_s = self.transform(t_s.permute(2, 0, 1)).permute(1, 2, 0)
        
        rgb = torch.cat([t_rgb, t_s], dim=-1).permute(2, 0, 1)
        original_rgb = torch.cat([torch.from_numpy(img) / 255.0, t_s], dim=-1).permute(2, 0, 1)

        # if self.mode[:5] == "train":
        #     transformed_joint = transform(torch.cat([rgb, original_rgb], dim=0))
        #     rgb, original_rgb = transformed_joint[:4], transformed_joint[4:]

        return rgb, original_rgb
    
    def __getitem__(self, index):
        scene_path = self.scene_paths[index]
        verts, _ = pytorch3d.io.load_ply(os.path.join(self.root, scene_path, "pointcloud.ply"))
        with open(os.path.join(self.root, scene_path, "meta_info.pkl"), 'rb') as f:
            Ks, RTs = pickle.load(f)

        # convert RT from opencv to pytorch3d
        opencv_to_pytorch3d = torch.eye(3)
        opencv_to_pytorch3d[0, 0] = -1
        opencv_to_pytorch3d[1, 1] = -1

        RT_pytorch3d = opencv_to_pytorch3d @ RTs
        Rs = RT_pytorch3d[:, :, :3].transpose(1, 2)
        Ts = RT_pytorch3d[:, :, 3]

        image_dir = os.path.join(self.root, scene_path, "images")
        mask_dir = os.path.join(self.root, scene_path, "masks")

        frame_ids = [f.split(".jpg")[0] for f in os.listdir(image_dir)]

        # train val split based on frame
        split_ratio = 0.8
        split_ind = int(len(frame_ids) * split_ratio)
        if self.mode == "train_frame_split":
            frame_list_indicis = list(range(len(frame_ids)))
            val_frame_list_indicis = set(frame_list_indicis[split_ind:])
            # val_frame_list_indicis = set(frame_list_indicis[::5])
            # val_frame_list_indicis = set(frame_list_indicis[int(len(frame_ids) * 0.4):int(len(frame_ids) * 0.6)])

            train_frame_list_indicis = [f_id for f_id in frame_list_indicis if f_id not in val_frame_list_indicis]
            frame_ids = [frame_ids[f_id] for f_id in train_frame_list_indicis]
            Rs = Rs[train_frame_list_indicis]
            Ts = Ts[train_frame_list_indicis]

            if self.sample_mode == "near20":
                start_idx = np.random.randint(0, int(0.8 * len(frame_ids)))
                end_idx = start_idx + int(0.2 * len(frame_ids))
                frame_ids = frame_ids[start_idx:end_idx]
                Rs = Rs[start_idx:end_idx]
                Ts = Ts[start_idx:end_idx]

        elif self.mode == "val_frame_split":
            frame_list_indicis = list(range(len(frame_ids)))
            # val_frame_list_indicis = frame_list_indicis[::5]
            val_frame_list_indicis = frame_list_indicis[split_ind:]
            # val_frame_list_indicis = frame_list_indicis[int(len(frame_ids) * 0.4):int(len(frame_ids) * 0.6)]

            frame_ids = [frame_ids[f_id] for f_id in val_frame_list_indicis]
            Rs = Rs[val_frame_list_indicis]
            Ts = Ts[val_frame_list_indicis]


        # no predefined sampled ids
        max_retry = 5
        sampled_frame_ids = self.sampled_frame_ids
        if sampled_frame_ids is None:
            sampled_frame_ids = []

            for i in range(self.num_views):
                valid = False
                n_try = 0
                while not valid and n_try < max_retry:
                    sampled_id = np.random.randint(0, len(frame_ids))
                    mask = plt.imread(os.path.join(mask_dir, f"{frame_ids[sampled_id]}.png"))
                    mask = (mask[..., 0]) > 0.5
                    if np.sum(mask) != 0:
                        valid = True
                    n_try += 1

                sampled_frame_ids.append(sampled_id)

        sampled_R = []
        sampled_T = []
        sampled_R_pytorch3d = []
        sampled_T_pytorch3d = []
        sampled_focal_length = []
        preprocessed_data = []
        sampled_img_paths = []
        for i in sampled_frame_ids:
            i_p = os.path.join(image_dir, f"{frame_ids[i]}.jpg")
            s_p = os.path.join(mask_dir, f"{frame_ids[i]}.png")
            p_d = self.preprocess_co3d(plt.imread(i_p), plt.imread(s_p)[:, :, 0])
            preprocessed_data.append(p_d)
            sampled_img_paths.append(i_p)

        for i in sampled_frame_ids:
            R = Rs[i]
            T = Ts[i]
            K = Ks[i]
            sampled_R_pytorch3d.append(R)
            sampled_T_pytorch3d.append(T)

        # make the first frame canonical
        sampled_R_pytorch3d = torch.stack(sampled_R_pytorch3d)
        sampled_T_pytorch3d = torch.stack(sampled_T_pytorch3d)
        cam_transform_pytorch3d = get_cam_transform(sampled_R_pytorch3d, sampled_T_pytorch3d)
        rel_cam_transform_pytorch3d = get_relative_cam_transform(cam_transform_pytorch3d[:1], cam_transform_pytorch3d[1:])
        identity = torch.eye(4)[None]
        new_cam_transform_pytorch3d = torch.cat([
            identity, 
            transform_relative_pose(identity, rel_cam_transform_pytorch3d)
        ], dim=0)
        rel_origin_transform_pytorch3d = get_relative_cam_transform(identity, cam_transform_pytorch3d[:1])[0]

        # new_verts = (rel_origin_transform_opencv[:3, :3] @ verts.T).T + rel_origin_transform_opencv[3, :3]
        verts = pytorch3d.transforms.transform3d.Transform3d(
            matrix=rel_origin_transform_pytorch3d
        ).transform_points(verts)
        sampled_R_pytorch3d = new_cam_transform_pytorch3d[:, :3, :3]
        sampled_T_pytorch3d = new_cam_transform_pytorch3d[:, 3, :3]
        center = torch.mean(verts, dim=0).clone()




        for R, T in zip(sampled_R_pytorch3d, sampled_T_pytorch3d):
            R, T = pytorch3d_to_kaolin(R, T)
            sampled_R.append(R)
            sampled_T.append(T)
            focal_length = torch.FloatTensor(
                [
                    K[0][0] / K[0][2], 
                    K[1][1] / K[1][2]
                ]
            )
            sampled_focal_length.append(focal_length)

        rgbs = torch.stack([d[0] for d in preprocessed_data], dim=0)
        original_rgbs = torch.stack([d[1] for d in preprocessed_data], dim=0)
        sampled_R = torch.stack(sampled_R)
        sampled_T = torch.stack(sampled_T)
        sampled_focal_length = torch.stack(sampled_focal_length)
        absolute_poses = get_cam_transform(sampled_R, sampled_T)
        relative_posees = get_relative_cam_transform(absolute_poses[:1], absolute_poses[1:])
        
        cam_quat = mat2quat(relative_posees)

        data = {
            "images": rgbs,
            "scene_path": scene_path,
            "original_images": original_rgbs,
            # "edge": edge,
            "path": sampled_img_paths,
            "label": "unavailable",
            "initial_pose": absolute_poses[0],
            "absolute_poses": absolute_poses,
            "relative_poses": relative_posees,
            "focal_length": sampled_focal_length, 
            "relative_poses_quat": cam_quat,
            # "verts": verts,
            "R_pytorch3d": sampled_R_pytorch3d,
            "T_pytorch3d": sampled_T_pytorch3d,
            "center": center
        }
        return {"data": data}
    

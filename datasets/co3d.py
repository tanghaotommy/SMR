import gzip
import json
import os
from torch.utils.data import Dataset
from PIL import Image
import glob

import torch.utils.data as data
import torch
import torchvision
import numpy as np
import random
from PIL import ImageFilter, ImageOps
from .bird import default_loader


class Co3DDataset(Dataset):
    def __init__(
        self,
        path,
        image_size,
        train=True,
        categories=None,
        transform=None,
        loader=default_loader,
        amodal=0,
        official_split=True,
        num_val_scenes=10,
    ):
        self.path = path
        if train:
            self.mode = "train"
        else:
            self.mode = "val"
        self.official_split = official_split
        self.image_size = image_size
        self.transform = transform
        self.loader = loader
        self.num_val_scenes = num_val_scenes
        self.amodal = amodal
        self.frame_paths = []
        self.mask_paths = []
        self.ignore_indicis = set()
        if categories is None:
            raise NotImplementedError
        
        for cat in categories:
            self._prepare_one_cat_official(cat)

        new_frame_paths = []
        new_mask_paths = []
        for i in range(len(self.frame_paths)):
            if i not in self.ignore_indicis:
                new_frame_paths.append(self.frame_paths[i])
                new_mask_paths.append(self.mask_paths[i])
        
        self.frame_paths = new_frame_paths
        self.mask_paths = new_mask_paths

        # n_data = 100
        # self.frame_paths = self.frame_paths[:n_data]
        # self.mask_paths = self.mask_paths[:n_data]

    def _prepare_one_cat_official(self, cat):
        cat_path = os.path.join(self.path, cat)
        if self.mode == "train": 
            set_lists = json.load(open(os.path.join(cat_path, "set_lists.json")))
            train_seq = set_lists["train_known"]
            cat_scene_paths = set([])    
            for train_pair in train_seq:
                (scene_id, _, frame_path) = train_pair
                fn = frame_path.split("/")[-1].split(".jpg")[0]
                self.frame_paths.append([os.path.join(self.path, frame_path), cat])
                self.mask_paths.append(os.path.join(self.path, cat, scene_id, "masks", f"{fn}.png"))
        else:
            assert self.mode == 'val'
            selected_seq = json.load(open(os.path.join(cat_path, "eval_batches_multisequence.json")))
            if self.num_val_scenes:
                selected_seq = selected_seq[:self.num_val_scenes]
            for seq in selected_seq:
                for val_pair in seq:
                    (scene_id, _, frame_path) = val_pair
                    fn = frame_path.split("/")[-1].split(".jpg")[0]
                    self.frame_paths.append([os.path.join(self.path, frame_path), cat])
                    self.mask_paths.append(os.path.join(self.path, cat, scene_id, "masks", f"{fn}.png"))

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        img_path, label = self.frame_paths[idx]
        seg_path = self.mask_paths[idx]
        target_height, target_width = self.image_size, self.image_size

        # image and its flipped image
        img = self.loader(img_path)
        seg = Image.open(seg_path)

        if np.sum(seg) == 0:
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        ys, xs = np.where(seg)
        obj_w = xs.max() - xs.min()
        obj_h = ys.max() - ys.min()
        obj_size = max(obj_w, obj_h)

        if obj_size > self.image_size:
            left = (xs.min() + xs.max()) // 2 - obj_size // 2
            right = left + obj_size

            upper = (ys.min() + ys.max()) // 2 - obj_size // 2
            lower = upper + obj_size
        else:
            left = (xs.min() + xs.max()) // 2 - target_width // 2
            right = left + target_width

            upper = (ys.min() + ys.max()) // 2 - target_height // 2
            lower = upper + target_height


        if self.amodal == 2:
            W, H = img.size
            enlarge_ratio = 1.5
            x_c = (left + right) // 2
            y_c = (upper + lower) // 2
            obj_w = right - left
            obj_h = upper - lower

            left = int(x_c - obj_w * enlarge_ratio / 2)
            right = int(x_c + obj_w * enlarge_ratio / 2)

            upper = int(y_c + obj_h * enlarge_ratio / 2)
            lower = int(y_c - obj_h * enlarge_ratio / 2)

        seg = seg.crop((left, upper, right, lower))
        img = img.crop((left, upper, right, lower))
        W, H = img.size

        if self.mode == "train":
            if random.uniform(0, 1) < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                seg = seg.transpose(Image.FLIP_LEFT_RIGHT)

            h = random.randint(int(0.90 * H), int(0.99 * H))
            w = random.randint(int(0.90 * W), int(0.99 * W))
            left = random.randint(0, W-w)
            upper = random.randint(0, H-h)
            right = random.randint(w - left, W)
            lower = random.randint(h - upper, H)
            img = img.crop((left, upper, right, lower))
            seg = seg.crop((left, upper, right, lower))

            # if random.uniform(0, 1) < 0.5:
            #     angle = random.randint(0, 90)
            #     if random.uniform(0, 1) < 0.5:
            #         angle = 360 - angle
            #     img = img.rotate(angle)
            #     seg = seg.rotate(angle)


        W, H = img.size
        desired_size = max(W, H)
        delta_w = desired_size - W
        delta_h = desired_size - H
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        img = ImageOps.expand(img, padding)
        seg = ImageOps.expand(seg, padding)

        img = img.resize((target_height, target_width))
        seg = seg.resize((target_height, target_width))
        seg = seg.point(lambda p: p > 160 and 255)


        edge = seg.filter(ImageFilter.FIND_EDGES)
        edge = edge.filter(ImageFilter.SMOOTH_MORE)
        edge = edge.point(lambda p: p > 20 and 255)
        edge = torchvision.transforms.functional.to_tensor(edge).max(0, True)[0]

        original_img = torchvision.transforms.functional.to_tensor(img)
        seg = torchvision.transforms.functional.to_tensor(seg).max(0, True)[0]
        

        img = original_img * seg + torch.ones_like(original_img) * (1 - seg)
        rgbs = torch.cat([img, seg], dim=0)
        original_rgbs = torch.cat([original_img, seg], dim=0)

        data= {'images': rgbs, 'path': img_path, 'label': label,
               'edge': edge, "original_images": original_rgbs}

        return {'data': data}
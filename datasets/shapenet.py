import torch
from torch.utils.data import Dataset
import random
from PIL import ImageFilter, ImageOps
import torchvision
import numpy as np
from PIL import Image
import os


class ShapeNetMultiView(Dataset):
    def __init__(self, data_path, category_ids=["02691156"], num_views=5, image_size=128):
        self.data_path = data_path
        self.num_views = num_views
        self.image_size = image_size
        self.object_dict = {}
        self.object_path = []
        for cat_id in category_ids:
            self.load_one_category(cat_id)

        # n_data = 10
        # self.object_path = self.object_path[:n_data]

    def load_one_category(self, category_id):
        objects = os.listdir(os.path.join(self.data_path, category_id))
        
        for obj_name in objects:
            obj_path = os.path.join(self.data_path, category_id, obj_name)
            frames_path = os.listdir(os.path.join(obj_path, "img"))

            self.object_path.append(obj_path)
            self.object_dict[obj_path] = frames_path

    def __len__(self):
        return len(self.object_path)
    
    def __getitem__(self, idx):
        obj_path = self.object_path[idx]
        model_id = obj_path.split("/")[-1]            
        mask_dir_path = os.path.join(obj_path, "seg")
        image_paths = self.object_dict[obj_path]
        input_views = np.random.choice(
            np.arange(len(image_paths)), size=self.num_views, replace=True
        )
        image_dir_path = os.path.join(obj_path, "img")
        # hard code resize value
        img_paths = [os.path.join(image_dir_path, image_paths[i]) for i in input_views]
        seg_paths = [os.path.join(
            mask_dir_path, 
            image_paths[i]
        ) for i in input_views]

        preprocessed_data = []
        obj_size = self.image_size
        for i in range(len(img_paths)):
            i_p, s_p = img_paths[i], seg_paths[i]
            preprocessed_data.append(self.preprocess(i_p, s_p, obj_size=obj_size))

        rgbs = torch.stack([d[0] for d in preprocessed_data], dim=0).squeeze(0)
        original_rgbs = torch.stack([d[1] for d in preprocessed_data], dim=0).squeeze(0)
        edge = torch.stack([d[2] for d in preprocessed_data], dim=0).squeeze(0)
        path = [f"{a.split('/')[-1]}" for a in img_paths]
        if len(path) == 1:
            path = path[0]
        
        data = {
            "images": rgbs,
            "original_images": original_rgbs,
            "edge": edge,
            "path": path,
            "label": "unavailable"
        }
        return {"data": data}

    def preprocess(self, img_path, seg_path, obj_size):
        img = Image.open(img_path)
        seg = Image.open(seg_path)
        target_height, target_width = self.image_size, self.image_size
        W, H = img.size
    
        ys, xs = np.where(seg)
        if len(ys):
            cx = (xs.min() + xs.max()) // 2
            cy = (ys.min() + ys.max()) // 2
        else:
            cx = W // 2
            cy = H // 2

        left = cx - obj_size // 2
        right = left + obj_size

        upper = cy - obj_size // 2
        lower = upper + obj_size
        
        seg = seg.crop((left, upper, right, lower))
        img = img.crop((left, upper, right, lower))
        W, H = img.size
        
        # if self.mode == "train":
        #     if random.uniform(0, 1) < 0.5:
        #         img = img.transpose(Image.FLIP_LEFT_RIGHT)
        #         seg = seg.transpose(Image.FLIP_LEFT_RIGHT)

        #     h = random.randint(int(0.90 * H), int(0.99 * H))
        #     w = random.randint(int(0.90 * W), int(0.99 * W))
        #     left = random.randint(0, W-w)
        #     upper = random.randint(0, H-h)
        #     right = random.randint(w - left, W)
        #     lower = random.randint(h - upper, H)
        #     seg = seg.crop((left, upper, right, lower))
        #     img = img.crop((left, upper, right, lower))
            

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
        
        return rgbs, original_rgbs, edge
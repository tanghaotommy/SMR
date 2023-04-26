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


CO3D_camera_dim = [1920, 1080]
CO3D_scaled_dim = [320, 180]


class Co3DSeqDataset(Dataset):
    def __init__(
        self,
        path,
        mode,
        number_of_images=5,
        image_size=128,
        categories=["bottle"],
        official_split=True,
        amodal=0,
        num_val_scenes=10,
        loader=default_loader,
    ):
        self.path = path
        self.number_of_images = number_of_images
        self.mode = mode
        self.official_split = official_split
        self.amodal = amodal
        self.loader = loader
        self.image_size = image_size
        # self.official_test_n_images = num_val_scenes
        self.val_scenes = num_val_scenes

        self.scene_paths = []
        self.scene_seq_dict = {}
        self.pose_dict = None
        self.num_sample_retry = 3
        
        for cat in categories:
            if official_split:
                self._prepare_one_cat_official(cat)
            else:
                self._prepare_one_cat_old(cat)
        self.render_kwargs = {
            'min_dist': 0.,
            'max_dist': 20.}
        # if mode == 'val':
        #     self.points_per_item = None
        # num_data = 10
        # self.scene_paths = self.scene_paths[:num_data]

    def _prepare_one_cat_old(self, cat):
        scene_ids = os.listdir(os.path.join(self.path, cat))
        valid_scene_ids = []
        if self.pose_dict is not None:
            frame_annotation_path = os.path.join(self.path, cat, "frame_annotations.jgz")
            with gzip.open(frame_annotation_path, "rt", encoding="utf8") as zipfile:
                frame_annots_list = json.load(zipfile)
            for entry in frame_annots_list:
                frame_path = entry["image"]["path"]
                self.pose_dict[frame_path] = entry
        for scene_id in scene_ids:
            if os.path.isdir(os.path.join(self.path, cat, scene_id)):
                valid_scene_ids.append(scene_id)
        hold_out_scene = self.hold_out_scene
        if hold_out_scene:
            if hold_out_scene < 1:
                cutoff = int(len(valid_scene_ids) * hold_out_scene)
            else:
                cutoff = hold_out_scene
            if self.mode == 'train':
                if cutoff < len(valid_scene_ids):
                    valid_scene_ids = valid_scene_ids[:-cutoff]
                else:
                    valid_scene_ids = []
            else:
                assert self.mode == 'val'
                if self.val_scenes < len(valid_scene_ids):
                    valid_scene_ids = valid_scene_ids[-self.val_scenes:]
        cat_scene_paths = [os.path.join(cat, scene_id) for scene_id in valid_scene_ids]
        self.scene_paths.extend(cat_scene_paths)
        for cat_scene_path in cat_scene_paths:
            scene_path = os.path.join(self.path, cat_scene_path)
            image_dir_path = os.path.join(scene_path, "images")
            self.scene_seq_dict[cat_scene_path] = os.listdir(image_dir_path)

    def _prepare_one_cat_official(self, cat):
        cat_path = os.path.join(self.path, cat)
        if self.pose_dict is not None:
            frame_annotation_path = os.path.join(self.path, cat, "frame_annotations.jgz")
            with gzip.open(frame_annotation_path, "rt", encoding="utf8") as zipfile:
                frame_annots_list = json.load(zipfile)
            for entry in frame_annots_list:
                frame_path = entry["image"]["path"]
                self.pose_dict[frame_path] = entry
        if self.mode == "train": 
            set_lists = json.load(open(os.path.join(cat_path, "set_lists.json")))
            train_seq = set_lists["train_known"]
            cat_scene_paths = set([])    
            for train_pair in train_seq:
                (scene_id, _, frame_path) = train_pair
                cat_scene_path = os.path.join(cat, scene_id)
                cat_scene_paths.add(cat_scene_path)
                if cat_scene_path not in self.scene_seq_dict:
                    self.scene_seq_dict[cat_scene_path] = []
                self.scene_seq_dict[cat_scene_path].append(
                    frame_path.split("/")[-1]
                )
            self.scene_paths.extend(list(cat_scene_paths))
        else:
            assert self.mode == 'val'
            multi_seq_list = json.load(open(os.path.join(cat_path, "eval_batches_multisequence.json")))
            selected_seq = multi_seq_list
            cat_scene_paths = set([])  
            # for seq in multi_seq_list:
            #     if len(seq) == self.official_test_n_images:
            #         selected_seq.append(seq)
            if self.val_scenes:
                selected_seq = selected_seq[:self.val_scenes]
            for seq in selected_seq:
                for val_pair in seq:
                    (scene_id, _, frame_path) = val_pair
                    cat_scene_path = os.path.join(cat, scene_id)
                    cat_scene_paths.add(cat_scene_path)
                    if cat_scene_path not in self.scene_seq_dict:
                        self.scene_seq_dict[cat_scene_path] = []
                    self.scene_seq_dict[cat_scene_path].append(
                        frame_path.split("/")[-1]
                    )
            self.scene_paths.extend(list(cat_scene_paths))
            # for seq in selected_seq:
            #     frame_list = [val_pair[2] for val_pair in seq]
            #     self.scene_paths.append(frame_list)

    def __len__(self):
        return len(self.scene_paths)

    def _get_camera_pose(self, pose_key):
        entry = self.pose_dict[pose_key]
        vp = entry["viewpoint"]
        camera_pos = PerspectiveCameras(
            focal_length=torch.tensor(vp["focal_length"])[None],
            principal_point=torch.tensor(vp["principal_point"], dtype=torch.float)[None],
            R=torch.tensor(vp["R"], dtype=torch.float)[None],
            T=torch.tensor(vp["T"], dtype=torch.float)[None],
        )
        RT_matrix = camera_pos.get_world_to_view_transform()
        return camera_pos, RT_matrix

    @staticmethod
    def _convert_camera_pose(RT_matrix):
        RT_matrix = RT_matrix.get_matrix()
        R = RT_matrix[..., :-1, :-1]
        quaternion_R = matrix_to_quaternion(R)
        # normalized_quaternion_R = quaternion_R / torch.maximum(torch.linalg.norm(quaternion_R, dim=-1, ord=2), torch.tensor(1e-6))
        T = RT_matrix[..., -1, :-1]
        camera_pose = torch.cat([quaternion_R, T], axis=-1).numpy()
        return camera_pose

    def __getitem__(self, idx):
        if self.official_split and self.mode == 'val':
            # frame_list = self.scene_paths[idx]
            # query_frame = frame_list[0].split("/")[-1]
            # scene_hash = frame_list[0].removesuffix(query_frame).removesuffix("images/")
            # scene_path = os.path.join(self.path, scene_hash)
            # mask_dir_path = os.path.join(scene_path, "masks")
            # input_views = np.arange(len(frame_list))
            # image_paths = [frame.split("/")[-1] for frame in frame_list]

            scene_hash = self.scene_paths[idx]
            scene_path = os.path.join(self.path, scene_hash)
            mask_dir_path = os.path.join(scene_path, "masks")
            image_paths = self.scene_seq_dict[scene_hash]
            input_views = np.random.choice(
                np.arange(len(image_paths)), size=self.number_of_images, replace=True
            )
        else:
            scene_hash = self.scene_paths[idx]
            scene_path = os.path.join(self.path, scene_hash)
            mask_dir_path = os.path.join(scene_path, "masks")
            image_paths = self.scene_seq_dict[scene_hash]
            input_views = np.random.choice(
                np.arange(len(image_paths)), size=self.number_of_images, replace=False
            )
        image_dir_path = os.path.join(scene_path, "images")
        # hard code resize value
        img_paths = [os.path.join(image_dir_path, image_paths[i]) for i in input_views]
        seg_paths = [os.path.join(
            mask_dir_path, 
            image_paths[i].replace(".jpg", ".png")
        ) for i in input_views]
        
        mask_images = [Image.open(
            os.path.join(
                mask_dir_path, 
                image_paths[i].replace(".jpg", ".png")
            ))
            for i in input_views
        ]
#         print(mask_images[0].shape)
#         plt.imshow(np.array(mask_images[0]))
        
        enlarge_ratio = 1
        if self.amodal == 2:
            enlarge_ratio = 1.5
        
        obj_sizes = [self.get_object_size(m) for m in mask_images]
        if not obj_sizes[0][1]:
            # no mask for the initial frame, need to resample
            return self.__getitem__(random.randint(0, self.__len__() - 1))
        
        obj_size = max([o[0] for o in obj_sizes])
            
        preprocessed_data = []
        
        for i in range(len(img_paths)):
            i_p, s_p = img_paths[i], seg_paths[i]
            num_retry = self.num_sample_retry

            while 1:
                p_d = self.preprocess(i_p, s_p, obj_size * enlarge_ratio) 
                # no mask for the object
                if p_d[0][-1].sum() == 0:
                    new_input_view = np.random.choice(
                        np.arange(len(image_paths))
                    )
                    # print("Resample: ", new_input_view)
                    i_p = os.path.join(image_dir_path, image_paths[new_input_view])
                    s_p = os.path.join(
                        mask_dir_path, 
                        image_paths[new_input_view].replace(".jpg", ".png")
                    )
                    img_paths[i] = i_p
                    seg_paths[i] = s_p
                else:
                    break
                
            
            preprocessed_data.append(p_d)
            
#         preprocessed_data = [
#             self.preprocess(img_paths[i], seg_paths[i], obj_size * enlarge_ratio) 
#             for i in range(len(input_views))
#         ]
        
        rgbs = torch.stack([d[0] for d in preprocessed_data], dim=0)
        original_rgbs = torch.stack([d[1] for d in preprocessed_data], dim=0)
        edge = torch.stack([d[2] for d in preprocessed_data], dim=0)
        
        data = {
            "images": rgbs,
            "original_images": original_rgbs,
            "edge": edge,
            "path": img_paths,
            "label": "unavailable"
        }
        return {"data": data}
        
    def get_object_size(self, seg):
        ys, xs = np.where(seg)
        
        if len(ys):
            obj_w = xs.max() - xs.min()
            obj_h = ys.max() - ys.min()
            has_mask = True
        else:
            # no mask for this image
            obj_w = obj_h = self.image_size
            has_mask = False
        
        return max(obj_w, obj_h), has_mask
        
    def preprocess(self, img_path, seg_path, obj_size):
        img = self.loader(img_path)
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
        
#         plt.imshow(img)
#         plt.show()
        
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
            seg = seg.crop((left, upper, right, lower))
            img = img.crop((left, upper, right, lower))
            

#         plt.imshow(img)
#         plt.show()

#         plt.imshow(img)
#         plt.show()
        
#         plt.imshow(seg)
#         plt.show()

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
        
        return rgbs, original_rgbs, edge

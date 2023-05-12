from pytorch3d.datasets import (
    R2N2,
    ShapeNetCore,
    collate_batched_meshes,
    render_cubified_voxels,
)
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex
)
from pytorch3d.structures import Meshes
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
import random
from PIL import ImageFilter, ImageOps
import torchvision
import numpy as np
from PIL import Image
import os
import multiprocessing


# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
#     torch.cuda.set_device(device)
# else:
device = torch.device("cpu")

class ShapeNetMultiView(Dataset):
    def __init__(self, data_path, category_ids=["02691156"], num_views=5, distance=1, elevation=30, random=False):
        shapenet_dataset = ShapeNetCore(data_path, version=2)
        self.id_lists = [i for i in range(len(shapenet_dataset)) if shapenet_dataset.synset_ids[i] in category_ids]
        self.shapenet_dataset = shapenet_dataset
        self.num_views = num_views
        self.distance = distance
        self.elevation = elevation
        self.image_size = 128

    def __len__(self):
        return len(self.id_lists)
    
    def __getitem__(self, idx):
        shapenet_model = self.shapenet_dataset[self.id_lists[idx]]
        model_id = shapenet_model["model_id"]                
        #   Rendering setting
        num_views = self.num_views

        # Get a batch of viewing angles. 
        # elev = torch.linspace(0, 360, num_views)

        if num_views > 1:
            if random:
                azim = torch.randint(-180, 181, (num_views,))
            else:
                azim = torch.linspace(-180, 180, num_views)
        else:
            azim = [random.randint(-180, 180)]

        # Place a point light in front of the object. As mentioned above, the front of 
        # the cow is facing the -z direction. 
        lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])

        # Initialize an OpenGL perspective camera that represents a batch of different 
        # viewing angles. All the cameras helper methods support mixed type inputs and 
        # broadcasting. So we can view the camera from the a distance of dist=2.7, and 
        # then specify elevation and azimuth angles for each viewpoint as tensors. 
        R, T = look_at_view_transform(dist=1, elev=30, azim=azim)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=self.image_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )
        images_by_model_ids = self.shapenet_dataset.render(
            model_ids=[
                model_id
            ],
            device=device,
            cameras=cameras,
            raster_settings=raster_settings,
            lights=lights,
            shader_type=SoftPhongShader
        )

        # preprocessed_data = []
        # obj_size = self.image_size
        # for image in images_by_model_ids:
        #     img = image[:, :, :3]
        #     seg = image[:, :, 3] > 0
        #     img = Image.fromarray(np.uint8(img*255))
        #     seg = Image.fromarray(np.uint8(seg*255))

        return images_by_model_ids, [f"{model_id}_{a}" for a in azim], model_id

num_views = 10
SHAPENET_PATH = "/datasets01/ShapeNetCore.v2/080320"
category_id = "02691156"
shapenet_mv = ShapeNetMultiView(SHAPENET_PATH, num_views=10, category_ids=[category_id])
save_dir = f"/checkpoint/haotang/data/shapenet_multiview_random_{num_views}/{category_id}"


def process_single(id):
    images, names, model_id = shapenet_mv[id]
    images = images.cpu()
    print(f"Processing {model_id} ...")

    for image, n in zip(images, names):
        model_id, view = n.split("_")
        img = image[:, :, :3]
        seg = image[:, :, 3] > 0
        img = Image.fromarray(np.uint8(img*255))
        seg = Image.fromarray(np.uint8(seg*255))

        save_path = os.path.join(save_dir, model_id, "img")
        os.makedirs(save_path, exist_ok=True)
        img.save(os.path.join(save_path, f"{view}.jpg"))

        save_path = os.path.join(save_dir, model_id, "seg")
        os.makedirs(save_path, exist_ok=True)
        seg.save(os.path.join(save_path, f"{view}.jpg"))
    
    print(f"Finished {model_id} ...")

def main():
    ids_list = list(range(len(shapenet_mv)))
    num_processes = 8
    pool = multiprocessing.Pool(num_processes)
    pool.map(process_single, ids_list)

    pool.close()
    pool.join()
    # process_single(list(range(len(shapenet_mv))))

if __name__ == "__main__":
    main()
import argparse
from collections import defaultdict
import csv
import os
import random
import math
from socket import gethostname
import tqdm
import shutil
import imageio
import numpy as np
import trimesh

# import torch related
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
from torchvision.transforms.functional import to_pil_image
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

# import kaolin related
import kaolin as kal
from kaolin.render.camera import generate_perspective_projection
from kaolin.render.mesh import dibr_rasterization, texture_mapping, \
                               spherical_harmonic_lighting, prepare_vertices


# import from folder
from fid_score import calculate_fid_given_paths
from datasets.bird import Dataset as BirdDataset
from datasets.co3d import Co3DDataset
from datasets.co3d_seq import Co3DSeqDataset
from datasets.co3d_normalized import Co3DSeqNormalizedDataset
from datasets.shapenet import ShapeNetMultiView
from geo_utils import get_cam_transform, get_relative_cam_transform, mat2quat, get_relative_pose, transform_relative_pose
import geo_utils
from utils import camera_position_from_spherical_angles, generate_transformation_matrix, compute_gradient_penalty, Timer, spherical_angles_from_camera_position
from models.model import VGG19, CameraEncoder, ShapeEncoder, LightEncoder, TextureEncoder, MultiViewTextureEncoder
from models.meshformer import MultiViewMeshFormer
from eval_utils import compute_pose_metric
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP
import time

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--outf', help='folder to output images and model checkpoints')
parser.add_argument('--pretrained_path', default=None, help='folder to pretrained output images and model checkpoints')
parser.add_argument('--dataset', help='dataset name', default='bird')
parser.add_argument('--dataroot', help='path to dataset root dir')
parser.add_argument('--template_path', default='template/sphere.obj', help='template mesh path')
parser.add_argument('--category', type=str, default='bird', help='list of object classes to use')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imageSize', type=list, default=[128, 128], help='the height / width of the input image to network')
parser.add_argument('--nk', type=int, default=5, help='size of kerner')
parser.add_argument('--nf', type=int, default=32, help='dim of unit channel')
parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='leaning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', default=1, type=int, help='enables cuda')
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
parser.add_argument('--multigpus', action='store_true', default=False, help='whether use multiple gpus mode')
parser.add_argument('--resume', action='store_true', default=True, help='whether resume ckpt')
parser.add_argument('--lambda_gan', type=float, default=0.0001, help='parameter')
parser.add_argument('--lambda_reg', type=float, default=1.0, help='parameter')
parser.add_argument('--lambda_data', type=float, default=1.0, help='parameter')
parser.add_argument('--lambda_ic', type=float, default=1.0, help='parameter')
parser.add_argument('--lambda_lc', type=float, default=0.001, help='parameter')
parser.add_argument('--azi_scope', type=float, default=360, help='parameter')
parser.add_argument('--elev_range', type=str, default="0~30", help='~ separated list of classes for the lsun data set')
parser.add_argument('--dist_range', type=str, default="2~6", help='~ separated list of classes for the lsun data set')
parser.add_argument('--amodal', type=int, default=0, help='which amodal type to use. 0 no amodal, 1 amodal without moving to center, 2 amodal with object to center')
parser.add_argument('--model', help='model name', default='SMR')
parser.add_argument('--visualization_epoch', type=int, default=1)
parser.add_argument('--val_epoch', type=int, default=10)
parser.add_argument('--ddp', action='store_true', default=False, help='whether use distributedparallel')
parser.add_argument('--refine_pose', action='store_true', default=False, help='whether refine initial pose')
parser.add_argument('--num_refine', type=int, default=3, help='number of pose refinement iterations')
parser.add_argument('--eval_only', action='store_true', default=False, help='whether run only evaluation')
parser.add_argument('--norm_layer', default=None, help='which normalization to use')
parser.add_argument('--n_gpu_per_node', type=int, default=2, help='number of GPUs per node')

opt = parser.parse_args()
print(opt)

# if opt.manualSeed is None:
#     opt.manualSeed = random.randint(1, 10000)
# # print("Random Seed: ", opt.manualSeed)
# random.seed(opt.manualSeed)
# torch.manual_seed(opt.manualSeed)

TRAINING_CATEGORIES = [
    "apple",
    "backpack",
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "cup",
    "donut",
    "hairdryer",
    "handbag",
    "hydrant",
    "keyboard",
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "stopsign",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "tv",
    "umbrella",
    "vase",
    "wineglass",
]

# TRAINING_CATEGORIES = ["bicycle"]

TEST_CATEGORIES = [
    "ball",
    "book",
    "couch",
    "frisbee",
    "hotdog",
    "kite",
    "remote",
    "sandwich",
    "skateboard",
    "suitcase",
]


def deep_copy(att, index=None, detach=False):
    if index is None:
        index = torch.arange(att['distances'].shape[0]).cuda()

    copy_att = {}
    for key, value in att.items():
        copy_keys = ['azimuths', 'elevations', 'distances', 'vertices', 'delta_vertices', 'textures', 'lights']
        if key in copy_keys:
            if detach:
                copy_att[key] = value[index].clone().detach()
            else:
                copy_att[key] = value[index].clone()
    return copy_att


class DiffRender(object):
    def __init__(self, filename_obj, image_size):
        self.image_size = image_size
        # camera projection matrix
        camera_fovy = np.arctan(1.0 / 2.5) * 2
        self.cam_proj = generate_perspective_projection(camera_fovy, ratio=image_size[0]/image_size[1])

        mesh = kal.io.obj.import_mesh(filename_obj, with_materials=True)
        # the sphere is usually too small (this is fine-tuned for the clock)

        # get vertices_init
        vertices = mesh.vertices
        vertices.requires_grad = False
        vertices_max = vertices.max(0, True)[0]
        vertices_min = vertices.min(0, True)[0]
        vertices = (vertices - vertices_min) / (vertices_max - vertices_min)
        vertices_init = vertices * 2.0 - 1.0 # (1, V, 3)

        # get face_uvs
        faces = mesh.faces
        uvs = mesh.uvs.unsqueeze(0)
        face_uvs_idx = mesh.face_uvs_idx
        face_uvs = kal.ops.mesh.index_vertices_by_faces(uvs, face_uvs_idx).detach()
        face_uvs.requires_grad = False

        self.num_faces = faces.shape[0]
        self.num_vertices = vertices_init.shape[0]
        face_size = 3

        # flip index
        # face_center = (vertices_init[0][faces[:, 0]] + vertices_init[0][faces[:, 1]] + vertices_init[0][faces[:, 2]]) / 3.0
        # face_center_flip = face_center.clone()
        # face_center_flip[:, 2] *= -1
        # self.flip_index = torch.cdist(face_center, face_center_flip).min(1)[1]
        
        # flip index
        vertex_center_flip = vertices_init.clone()
        vertex_center_flip[:, 2] *= -1
        self.flip_index = torch.cdist(vertices_init, vertex_center_flip).min(1)[1]

        ## Set up auxiliary connectivity matrix of edges to faces indexes for the flat loss
        edges = torch.cat([faces[:,i:i+2] for i in range(face_size - 1)] +
                        [faces[:,[-1,0]]], dim=0)

        edges = torch.sort(edges, dim=1)[0]
        face_ids = torch.arange(self.num_faces, dtype=torch.long).repeat(face_size)
        edges, edges_ids = torch.unique(edges, sorted=True, return_inverse=True, dim=0)
        nb_edges = edges.shape[0]
        # edge to faces
        sorted_edges_ids, order_edges_ids = torch.sort(edges_ids)
        sorted_faces_ids = face_ids[order_edges_ids]
        # indices of first occurences of each key
        idx_first = torch.where(
            torch.nn.functional.pad(sorted_edges_ids[1:] != sorted_edges_ids[:-1],
                                    (1,0), value=1))[0]
        num_faces_per_edge = idx_first[1:] - idx_first[:-1]
        # compute sub_idx (2nd axis indices to store the faces)
        offsets = torch.zeros(sorted_edges_ids.shape[0], dtype=torch.long)
        offsets[idx_first[1:]] = num_faces_per_edge
        sub_idx = (torch.arange(sorted_edges_ids.shape[0], dtype=torch.long) -
                torch.cumsum(offsets, dim=0))
        num_faces_per_edge = torch.cat([num_faces_per_edge,
                                    sorted_edges_ids.shape[0] - idx_first[-1:]],
                                    dim=0)
        max_sub_idx = 2
        edge2faces = torch.zeros((nb_edges, max_sub_idx), dtype=torch.long)
        edge2faces[sorted_edges_ids, sub_idx] = sorted_faces_ids
        edge2faces = edge2faces

        ## Set up auxiliary laplacian matrix for the laplacian loss
        vertices_laplacian_matrix = kal.ops.mesh.uniform_laplacian(self.num_vertices, faces)

        self.vertices_init = vertices_init
        self.faces = faces
        self.face_uvs = face_uvs
        self.edge2faces = edge2faces
        self.vertices_laplacian_matrix = vertices_laplacian_matrix

    def render(self, **attributes):
        azimuths = attributes['azimuths']
        elevations = attributes['elevations']
        distances = attributes['distances']
        batch_size = azimuths.shape[0]
        device = azimuths.device
        cam_proj = self.cam_proj.to(device)

        vertices = attributes['vertices']
        textures = attributes['textures']
        lights = attributes['lights']

        faces = self.faces.to(device)
        face_uvs = self.face_uvs.to(device)

        num_faces = faces.shape[0]

        object_pos = torch.tensor([[0., 0., 0.]], dtype=torch.float, device=device).repeat(batch_size, 1)
        camera_up = torch.tensor([[0., 1., 0.]], dtype=torch.float, device=device).repeat(batch_size, 1)
        # camera_pos = torch.tensor([[0., 0., 4.]], dtype=torch.float, device=device).repeat(batch_size, 1)
        camera_pos = camera_position_from_spherical_angles(distances, elevations, azimuths, degrees=True)
        cam_transform = generate_transformation_matrix(camera_pos, object_pos, camera_up)

        face_vertices_camera, face_vertices_image, face_normals = \
           prepare_vertices(vertices=vertices,
                faces=faces, camera_proj=cam_proj, camera_transform=cam_transform
            )

        face_normals_unit = kal.ops.mesh.face_normals(face_vertices_camera, unit=True)
        face_normals_unit = face_normals_unit.unsqueeze(-2).repeat(1, 1, 3, 1)
        face_attributes = [
            torch.ones((batch_size, num_faces, 3, 1), device=device),
            face_uvs.repeat(batch_size, 1, 1, 1),
            face_normals_unit
        ]

        image_features, soft_mask, face_idx = dibr_rasterization(
            self.image_size[0], self.image_size[1], face_vertices_camera[:, :, :, -1],
            face_vertices_image, face_attributes, face_normals[:, :, -1])

        # image_features is a tuple in composed of the interpolated attributes of face_attributes
        # texture_coords, mask = image_features
        texmask, texcoord, imnormal = image_features

        texcolor = texture_mapping(texcoord, textures, mode='bilinear')
        coef = spherical_harmonic_lighting(imnormal, lights)
        image = texcolor * texmask * coef.unsqueeze(-1) + torch.ones_like(texcolor) * (1 - texmask)
        image = torch.clamp(image, 0, 1)
        render_img = image
        
        render_silhouttes = soft_mask[..., None]
        rgbs = torch.cat([render_img, render_silhouttes], axis=-1).permute(0, 3, 1, 2)

        attributes['face_normals'] = face_normals
        attributes['faces_image'] = face_vertices_image.mean(dim=2)
        attributes['visiable_faces'] = face_normals[:, :, -1] > 0.1
        return rgbs, attributes
    
    def render_w_camera(self, **attributes):
        vertices = attributes['vertices']
        textures = attributes['textures']
        lights = attributes['lights']
        focal_length = attributes["focal_length"]
        center = attributes['center']
        num_views = attributes["num_views"]
        center = center.repeat_interleave(num_views, dim=0).unsqueeze(1)
        vertices = vertices + center

        batch_size = vertices.shape[0]
        device = vertices.device
        cam_proj = self.cam_proj.to(device)
        cam_transform = attributes["absolute_poses"]
        _, _, k, _ = cam_transform.shape
        cam_transform = cam_transform.view(batch_size, k, k)
        cam_transform = cam_transform[:, :, :3]

        faces = self.faces.to(device)
        face_uvs = self.face_uvs.to(device)

        num_faces = faces.shape[0]
        z_direction = - torch.ones([focal_length.shape[0], focal_length.shape[1], 1]).to(device)
        cam_proj = torch.cat([focal_length, z_direction], dim=2)

        face_vertices_camera, face_vertices_image, face_normals = \
           prepare_vertices(vertices=vertices,
                faces=faces, camera_proj=cam_proj, camera_transform=cam_transform
            )

        face_normals_unit = kal.ops.mesh.face_normals(face_vertices_camera, unit=True)
        face_normals_unit = face_normals_unit.unsqueeze(-2).repeat(1, 1, 3, 1)
        face_attributes = [
            torch.ones((batch_size, num_faces, 3, 1), device=device),
            face_uvs.repeat(batch_size, 1, 1, 1),
            face_normals_unit
        ]

        image_features, soft_mask, face_idx = dibr_rasterization(
            self.image_size[0], self.image_size[1], face_vertices_camera[:, :, :, -1],
            face_vertices_image, face_attributes, face_normals[:, :, -1])

        # image_features is a tuple in composed of the interpolated attributes of face_attributes
        # texture_coords, mask = image_features
        texmask, texcoord, imnormal = image_features

        texcolor = texture_mapping(texcoord, textures, mode='bilinear')
        coef = spherical_harmonic_lighting(imnormal, lights)
        if torch.isnan(coef).sum():
            print("===========coef============isnan")
        if torch.isnan(texcolor).sum():
            print("===========texcolor============isnan")
        if torch.isnan(texmask).sum():
            print("===========texmask============isnan")
        image = texcolor * texmask * coef.unsqueeze(-1) + torch.ones_like(texcolor) * (1 - texmask)
        if torch.isnan(image).sum():
            print("===========image============isnan")
        image = torch.clamp(image, 0, 1)
        render_img = image
        
        render_silhouttes = soft_mask[..., None]
        rgbs = torch.cat([render_img, render_silhouttes], axis=-1).permute(0, 3, 1, 2)
        if torch.isnan(render_silhouttes).sum():
            print(cam_transform)
            print(torch.isnan(cam_transform).sum())
            print("===========render_silhouttes============isnan")

        if torch.isnan(rgbs).sum():
            print("======set rendered rgbs to 0========")
            rgbs = torch.ones_like(rgbs)

        attributes['face_normals'] = face_normals
        attributes['faces_image'] = face_vertices_image.mean(dim=2)
        attributes['visiable_faces'] = face_normals[:, :, -1] > 0.1
        return rgbs, attributes

    def render_torch3d(self, **attributes):
        pass

    def recon_att(self, pred_att, target_att):
        def angle2xy(angle):
            angle = angle * math.pi / 180.0
            x = torch.cos(angle)
            y = torch.sin(angle)
            return torch.stack([x, y], 1)

        loss_azim = torch.pow(angle2xy(pred_att['azimuths']) -
                     angle2xy(target_att['azimuths']), 2).mean()
        loss_elev = torch.pow(angle2xy(pred_att['elevations']) -
                     angle2xy(target_att['elevations']), 2).mean()
        loss_dist = torch.pow(pred_att['distances'] - target_att['distances'], 2).mean()
        loss_cam = loss_azim + loss_elev + loss_dist

        loss_shape = torch.pow(pred_att['vertices'] - target_att['vertices'], 2).mean()
        loss_texture = torch.pow(pred_att['textures'] - target_att['textures'], 2).mean()
        loss_light = 0.1  * torch.pow(pred_att['lights'] - target_att['lights'], 2).mean()
        return loss_cam, loss_shape, loss_texture, loss_light

    def recon_data(self, pred_data, gt_data):
        image_weight = 1.
        mask_weight = 1.

        pred_img = pred_data[:, :3]
        pred_mask = pred_data[:, 3]
        gt_img = gt_data[:, :3]
        # gt_seg = gt_data[:, 3].unsqueeze(1)
        # gt_img = gt_img * gt_seg + torch.ones_like(gt_img) * (1 - gt_seg)
        gt_mask = gt_data[:, 3]
        loss_image = torch.mean(torch.abs(pred_img - gt_img) * gt_mask.unsqueeze(1))
        loss_mask = kal.metrics.render.mask_iou(pred_mask, gt_mask)

        loss_data = image_weight * loss_image + mask_weight * loss_mask
        return loss_data

    def recon_flip(self, att):
        Na = att['delta_vertices']
        Nf = Na.index_select(1, self.flip_index.to(Na.device))
        Nf[..., 2] *= -1

        loss_norm = (Na - Nf).norm(dim=2).mean()
        return loss_norm

    def calc_reg_loss(self, att):
        laplacian_weight = 0.1
        flat_weight = 0.001

        # laplacian loss
        delta_vertices = att['delta_vertices']
        device = delta_vertices.device

        vertices_laplacian_matrix = self.vertices_laplacian_matrix.to(device)
        edge2faces = self.edge2faces.to(device)
        face_normals = att['face_normals']
        nb_vertices = delta_vertices.shape[1]
        
        delta_vertices_laplacian = torch.matmul(vertices_laplacian_matrix, delta_vertices)
        loss_laplacian = torch.mean(delta_vertices_laplacian ** 2) * nb_vertices * 3
        # flat loss
        mesh_normals_e1 = face_normals[:, edge2faces[:, 0]]
        mesh_normals_e2 = face_normals[:, edge2faces[:, 1]]
        faces_cos = torch.sum(mesh_normals_e1 * mesh_normals_e2, dim=2)
        loss_flat = torch.mean((faces_cos - 1) ** 2) * edge2faces.shape[0]

        loss_reg = laplacian_weight * loss_laplacian + flat_weight * loss_flat
        return loss_reg


# network of landmark consistency
class Landmark_Consistency(nn.Module):
    def __init__(self, num_landmarks, dim_feat, num_samples):
        super(Landmark_Consistency, self).__init__()
        self.num_landmarks = num_landmarks
        self.num_samples = num_samples

        n_features = dim_feat
        self.classifier = nn.Sequential(
            nn.Conv1d(n_features, 1024, 1, 1, 0), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Conv1d(1024, self.num_landmarks, 1, 1, 0)
        )
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, img_feat, landmark_2d, visiable):
        batch_size = landmark_2d.shape[0]
        grid_x = landmark_2d.unsqueeze(1)  # (N, 1, V, 2)

        feat_sampled = F.grid_sample(img_feat, grid_x, mode='bilinear', padding_mode='zeros')  # (N, C, 1, V)
        feat_sampled = feat_sampled.squeeze(dim=2).transpose(1, 2)  # (N, V, C)
        feature_agg = feat_sampled.transpose(1, 2)   # (N, F, V)
        # feature_agg = torch.cat([feat_sampled, landmark_2d], dim=2).transpose(1, 2) # (B, F, V)

        # select feature
        select_index = torch.randperm(self.num_landmarks)[:self.num_samples].cuda()
        feature_agg = feature_agg.index_select(2, select_index) # (B, F, 64)
        logits = self.classifier(feature_agg) # (B, num_landmarks, 64)
        logits = logits.transpose(1, 2).reshape(-1, self.num_landmarks) # (B*64, num_landmarks)

        labels = torch.arange(self.num_landmarks)[None].repeat(batch_size, 1).cuda() # (B, V)
        labels = labels.index_select(1, select_index).view(-1) # (B*64,)

        visiable = visiable.index_select(1, select_index).view(-1).float()
        loss = (self.cross_entropy(logits, labels) * visiable).sum() / visiable.sum()
        return loss

class AttributeEncoder(nn.Module):
    def __init__(self, num_vertices, vertices_init, azi_scope, elev_range, dist_range, nc, nf, nk):
        super(AttributeEncoder, self).__init__()
        self.num_vertices = num_vertices
        self.vertices_init = vertices_init

        self.camera_enc = CameraEncoder(nc=nc, nk=nk, azi_scope=azi_scope, elev_range=elev_range, dist_range=dist_range)
        self.shape_enc = ShapeEncoder(nc=nc, nk=nk, num_vertices=self.num_vertices)
        self.texture_enc = TextureEncoder(nc=nc, nk=nk, nf=nf, num_vertices=self.num_vertices)
        self.light_enc = LightEncoder(nc=nc, nk=nk)
        # self.feat_enc = FeatEncoder(nc=4, nf=32)
        self.feat_enc = VGG19()

    def forward(self, x):
        device = x.device
        batch_size = x.shape[0]
        input_img = x

        # cameras
        cameras = self.camera_enc(input_img)
        azimuths, elevations, distances = cameras

        # textures
        textures = self.texture_enc(input_img)

        # vertex
        delta_vertices = self.shape_enc(input_img)
        vertices = self.vertices_init[None].to(device) + delta_vertices

        lights = self.light_enc(input_img)

        # image feat
        with torch.no_grad():
            self.feat_enc.eval()
            img_feats = self.feat_enc(input_img)

        # others
        attributes = {
        'azimuths': azimuths,
        'elevations': elevations,
        'distances': distances,
        'vertices': vertices,
        'delta_vertices': delta_vertices,
        'textures': textures,
        'lights': lights,
        'img_feats': img_feats
        }
        return attributes


class MeshFormerEncoder(nn.Module):
    def __init__(self, num_vertices, vertices_init, azi_scope, elev_range, dist_range, nc, nf, nk, use_multi_view_texture=False, refine_pose=False, norm_layer=None, num_refine=opt.num_refine):
        super(MeshFormerEncoder, self).__init__()
        self.num_vertices = num_vertices
        self.vertices_init = vertices_init
        self.use_multi_view_texture = use_multi_view_texture
        self.refine_pose = refine_pose
        self.num_refine = num_refine

        self.camera_enc = CameraEncoder(nc=nc, nk=nk, azi_scope=azi_scope, elev_range=elev_range, dist_range=dist_range)
        self.meshformer = MultiViewMeshFormer(num_vertices, vertices_init, azi_scope, elev_range, dist_range, refine_pose=refine_pose, norm_layer=norm_layer)
        if self.use_multi_view_texture:
            self.texture_enc = MultiViewTextureEncoder(nc=nc, nk=nk, nf=nf, num_vertices=self.num_vertices)
        else:
            self.texture_enc = TextureEncoder(nc=nc, nk=nk, nf=nf, num_vertices=self.num_vertices)
        self.light_enc = LightEncoder(nc=nc, nk=nk)
        # self.feat_enc = FeatEncoder(nc=4, nf=32)
        self.feat_enc = VGG19()

    def forward(self, x, gt_poses, use_gt_pose=False):
        device = x.device
        batch_size, num_views, C, H, W = x.shape
        input_img = x

        # meshformer
        ret = self.meshformer(input_img, gt_poses, use_gt_pose=use_gt_pose)
        delta_vertices = ret["delta_vertices"]
        lights = ret["lights"]
        textures = ret["textures"]
        pred_cameras = ret["pred_cameras"]
        refined_cameras_dict = ret["refined_cameras_dict"]
        shape_memory = ret["shape_memory"]

        # # cameras
        # cameras = self.camera_enc(input_img)
        # azimuths, elevations, distances = cameras
        # azimuths, elevations, distances = pred_cameras_sphere

        # textures
        l = np.array(range(0, batch_size*num_views, num_views))
        if self.use_multi_view_texture:
            textures = self.texture_enc(input_img, shape_memory)
        else:
            textures = self.texture_enc(input_img.view(batch_size * num_views, C, H, W)[l])
        textures = textures.repeat_interleave(num_views, dim=0)

        # vertex
        vertices = self.vertices_init[None].to(device) + delta_vertices

        # lights
        lights = self.light_enc(input_img.view(batch_size * num_views, C, H, W))

        # image feat
        with torch.no_grad():
            self.feat_enc.eval()
            img_feats = self.feat_enc(input_img.view(batch_size * num_views, C, H, W))

        # others
        attributes = {
            'vertices': vertices,
            'delta_vertices': delta_vertices,
            'textures': textures,
            'lights': lights,
            'img_feats': img_feats,
            'pred_cameras': pred_cameras,
            'refined_cameras_dict': refined_cameras_dict
        }
        return attributes


class Discriminator(nn.Module):
    def __init__(self, nc, nf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, nf, 7, 1, 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 -> 64
            nn.Conv2d(nf, nf * 2, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 -> 32
            nn.Conv2d(nf * 2, nf * 4, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf * 4, nf * 4, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 -> 16
            nn.Conv2d(nf * 4, nf * 8, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 -> 8
            nn.Conv2d(nf * 8, nf * 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(nf * 16, nf * 8, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf * 8, 1, 1, 1, 0, bias=False)
        )

    def forward(self, input):
        output = self.main(input).mean([2, 3])
        return output


# custom weights initialization called on netE and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def forward_pass(epoch, rank, model, optimizer, scheduler, diffRender, dataloader, summary_writer, mode="Train"):
    input_key = "original_images"
    lossR_all = []
    lossR_reg_all = []
    lossR_data_all = []
    lossR_flip_all = []
    lossR_cam_quat_all = []
    loss_pose_all = []
    loss_trans_all = []
    loss_refined_pose_all = []
    loss_refined_trans_all = []
    loss_refined_pose_dict_all = defaultdict(list)
    loss_refined_trans_dict_all = defaultdict(list)
    lossR_refined_cam_quat_all = []
    rot_error_all = defaultdict(list)
    refined_rot_error_dict_all = defaultdict(lambda: defaultdict(list))
    refined_trans_error_dict_all = defaultdict(lambda: defaultdict(list))
    rot_gt_all = defaultdict(list)
    trans_error_all = defaultdict(list)
    refined_rot_error_all = defaultdict(list)
    refined_trans_error_all = defaultdict(list)

    rot_error_all_list = defaultdict(list)
    rot_gt_all_list = defaultdict(list)
    trans_error_all_list = defaultdict(list)
    refined_rot_error_all_list = defaultdict(list)
    refined_trans_error_all_list = defaultdict(list)

    for iter, data in enumerate(dataloader):
        Xa = Variable(data['data']['images']).cuda()
        # Ea = Variable(data['data']['edge']).cuda()
        absolute_poses = data['data']['absolute_poses'].cuda()
        relative_poses = data['data']['relative_poses'].cuda()
        cam_quat_gt = data['data']['relative_poses_quat'].cuda()
        focal_length = data['data']['focal_length'].cuda()
        center = data["data"]["center"].cuda()
        category = data["data"]["category"]
        batch_size = Xa.shape[0]
        num_views = Xa.shape[1]

        Xb = Variable(data['data'][input_key]).cuda()
        batch_size, num_views, C, H, W = Xb.shape

        # encode real
        Ae = model(
            Xb, 
            {
                "gt_poses": cam_quat_gt,
                "center": center,
                "focal_length": focal_length,
                "absolute_poses": absolute_poses
            }, 
            use_gt_pose=True
        )
        if torch.isnan(absolute_poses).sum() > 0:
            print("=========after model forward, absolute_poses isnan")
            print(absolute_poses)

        Ae["absolute_poses"] = absolute_poses
        Ae["focal_length"] = focal_length
        Ae["center"] = center
        Ae["num_views"] = num_views
        Xer, Ae = diffRender.render_w_camera(**Ae)

        for k in range(len(Xer)):
            d = Xer[k]
            if torch.isnan(d).sum() > 0:
                print(k)
                print("lossR_data=====================")
                print(focal_length)
                print(data["data"]["path"][k % num_views][k // num_views])
                print(d)

        optimizer.zero_grad()
        lossR_data = opt.lambda_data * diffRender.recon_data(Xer, Xa.view(batch_size * num_views, C, H, W))

        # mesh regularization
        lossR_reg = opt.lambda_reg * diffRender.calc_reg_loss(Ae)
        
        # overall loss
        lossR_fake = torch.zeros(1).to(lossR_data.device)
        lossR_LC = torch.zeros(1).to(lossR_data.device)
        lossR_IC = torch.zeros(1).to(lossR_data.device)

        device = Xb.device
        poses_pred = Ae["pred_cameras"]
        refined_poses_pred_dict = Ae["refined_cameras_dict"]
        
        tmp = torch.zeros_like(poses_pred)
        tmp[:,:4] = F.normalize(poses_pred[:,:4])
        tmp[:,4:] = poses_pred[:,4:]
        poses_pred = tmp

        loss = 0.0
        # Merge the batch_size and num_views dim
        cam_quat_gt = cam_quat_gt.view(batch_size * (num_views - 1), -1)

        # some gt trans are inf, ignore these entry
        loss_pose = F.mse_loss(poses_pred[:,:4], cam_quat_gt[:,:4], reduction='none')
        loss_pose = loss_pose[loss_pose < 10].mean()
        # loss_pose = F.mse_loss(geo_utils.quat2mat(poses_pred)[:, :3, :3], relative_poses.view(batch_size * (num_views - 1), 4, 4)[:, :3, :3], reduction='none')
        # loss_pose_1 = F.mse_loss(poses_pred[:,:4], cam_quat_gt[:,:4], reduction='none')
        # loss_pose_2 = F.mse_loss(geo_utils.quat2mat(poses_pred)[:, :3, :3], relative_poses.view(batch_size * (num_views - 1), 4, 4)[:, :3, :3], reduction='none')
        # loss_pose_1 = loss_pose_1[loss_pose_1 < 10].mean()
        # loss_pose_2 = loss_pose_2[loss_pose_2 < 10].mean()
        # loss_pose = loss_pose_1 + loss_pose_2
        loss_trans = F.mse_loss(poses_pred[:,4:], cam_quat_gt[:,4:], reduction='none')

        loss_trans = loss_trans[loss_trans < 10].mean()
        lossR_cam_quat = loss_pose + loss_trans

        if opt.refine_pose:
            loss_refined_pose_dict = {}
            loss_refined_trans_dict = {}
            for refine_iter, refined_poses_pred in refined_poses_pred_dict.items():
                loss_refined_pose_iter = F.mse_loss(refined_poses_pred[:, :3, :3], relative_poses[:, :, :3, :3].view(batch_size * (num_views - 1), 3, 3), reduction='none')
                loss_refined_trans_iter = F.mse_loss(refined_poses_pred[:, 3, :3], cam_quat_gt[:, 4:], reduction='none')
                loss_refined_pose_iter = loss_refined_pose_iter[loss_refined_pose_iter < 10].mean()
                loss_refined_trans_iter = loss_refined_trans_iter[loss_refined_trans_iter < 10].mean()
                loss_refined_pose_dict[refine_iter] = loss_refined_pose_iter
                loss_refined_trans_dict[refine_iter] = loss_refined_trans_iter

            loss_refine_iter_name = f"refined_cameras_{opt.num_refine - 1}"
            if loss_refine_iter_name == "all":
                loss_refined_pose = torch.sum([v for _, v in loss_refined_pose_dict.items()])
                loss_refined_trans = torch.sum([v for _, v in loss_refined_trans_dict.items()])
            else:
                loss_refined_pose = loss_refined_pose_dict[loss_refine_iter_name]
                loss_refined_trans = loss_refined_trans_dict[loss_refine_iter_name]
        else:
            loss_refined_pose = torch.zeros_like(loss_pose)
            loss_refined_trans = torch.zeros_like(loss_trans)

        lossR_refined_cam_quat = loss_refined_pose + loss_refined_trans

        lossR = lossR_fake + lossR_reg + lossR_data + lossR_IC +  lossR_LC + lossR_cam_quat + lossR_refined_cam_quat
        # lossR = lossR_fake + lossR_reg + lossR_data + lossR_IC +  lossR_LC + lossR_cam_quat

        rot_error_batch = []
        rot_error, trans_error = defaultdict(list), defaultdict(list)
        refined_rot_error, refined_trans_error = defaultdict(list), defaultdict(list)
        refined_rot_error_dict, refined_trans_error_dict = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
        rot_all = defaultdict(list)
        canonical = torch.FloatTensor([1, 0, 0, 0, 0, 0, 0]).cuda()

        # Convert to quaternion
        if opt.refine_pose:
            refined_poses_pred_dict = {k: geo_utils.mat2quat(v) for k, v in refined_poses_pred_dict.items()}
        for img_idx in range(len(cam_quat_gt)):
            b_id = img_idx // (num_views - 1)
            cat = category[b_id]
            cur_rot_error, cur_trans_error = compute_pose_metric(poses_pred[img_idx].detach().cpu(), 
                                                                    cam_quat_gt[img_idx].detach().cpu())
            cur_rot, _ = compute_pose_metric(cam_quat_gt[img_idx].detach().cpu(), 
                                            canonical.detach().cpu())
            rot_all[cat].append(cur_rot)
            rot_error[cat].append(cur_rot_error)
            rot_error_batch.append(cur_rot_error)
            trans_error[cat].append(cur_trans_error)

            if opt.refine_pose:
                for refine_iter_name in refined_poses_pred_dict.keys():
                    refined_poses_pred = refined_poses_pred_dict[refine_iter_name] 
                    cur_rot_error, cur_trans_error = compute_pose_metric(refined_poses_pred[img_idx].detach().cpu(), 
                                                                cam_quat_gt[img_idx].detach().cpu())
                    refined_rot_error_dict[refine_iter_name][cat].append(cur_rot_error)
                    refined_trans_error_dict[refine_iter_name][cat].append(cur_trans_error)

                final_refine_iter_name = f"refined_cameras_{opt.num_refine - 1}"
                refined_poses_pred = refined_poses_pred_dict[final_refine_iter_name] 
                cur_rot_error, cur_trans_error = compute_pose_metric(refined_poses_pred[img_idx].detach().cpu(), 
                                                            cam_quat_gt[img_idx].detach().cpu())
                    
            refined_rot_error[cat].append(cur_rot_error)
            refined_trans_error[cat].append(cur_trans_error)

        for k in rot_all.keys():
            rot_error_all_list[k].extend(rot_error[k])
            rot_gt_all_list[k].extend(rot_all[k])
            trans_error_all_list[k].extend(trans_error[k])
            refined_rot_error_all_list[k].extend(refined_rot_error[k])
            refined_trans_error_all_list[k].extend(refined_trans_error[k])

            rot_error[k] = np.nanmean(rot_error[k])
            trans_error[k] = np.nanmean(trans_error[k])
            refined_rot_error[k] = np.nanmean(refined_rot_error[k])
            refined_trans_error[k] = np.nanmean(refined_trans_error[k])
            rot_all[k] = np.nanmean(rot_all[k])

            for refine_iter_name in refined_rot_error_dict.keys():
                refined_rot_error_dict[refine_iter_name][k] = np.nanmean(refined_rot_error_dict[refine_iter_name][k])
                refined_trans_error_dict[refine_iter_name][k] = np.nanmean(refined_trans_error_dict[refine_iter_name][k])

        # record result from each iteration
        lossR_all.append(lossR.item())
        lossR_reg_all.append(lossR_reg.item())
        lossR_cam_quat_all.append(lossR_cam_quat.item())
        lossR_refined_cam_quat_all.append(lossR_refined_cam_quat.item())
        lossR_data_all.append(lossR_data.item())
        loss_pose_all.append(loss_pose.item())
        loss_trans_all.append(loss_trans.item())
        loss_refined_pose_all.append(loss_refined_pose.item())
        loss_refined_trans_all.append(loss_refined_trans.item())

        if opt.refine_pose:
            for k in loss_refined_pose_dict.keys():
                loss_refined_pose_dict_all[k].append(loss_refined_pose_dict[k].item())
                loss_refined_trans_dict_all[k].append(loss_refined_trans_dict[k].item())

        for k in rot_all.keys():
            rot_error_all[k].append(rot_error[k])
            rot_gt_all[k].append(rot_all[k])
            trans_error_all[k].append(trans_error[k])
            refined_rot_error_all[k].append(refined_rot_error[k]) 
            refined_trans_error_all[k].append(refined_trans_error[k])

            for refine_iter_name in refined_rot_error_dict.keys():
                refined_rot_error_dict_all[refine_iter_name][k].append(refined_rot_error_dict[refine_iter_name][k])
                refined_trans_error_dict_all[refine_iter_name][k].append(refined_trans_error_dict[refine_iter_name][k])

        # if rank == 0:
        print('Name: ', opt.outf)
        print('%d [%s][%d/%d][%d/%d]\n'
        'lossR: %.4f lossR_reg: %.4f lossR_data: %.4f, loss_pose: %.4f refined_loss_pose: %.4f loss_trans: %.4f refined_loss_trans: %.4f'
        'rot_error: %.4f refined_rot_error: %.4f trans_error: %.4f refined_trans_error: %.4f rot_all: %.4f \n'
            % (rank, mode, epoch, opt.niter, iter, len(dataloader),
                lossR.item(), lossR_reg.item(), lossR_data.item(),
                loss_pose.item(), loss_refined_pose.item(), loss_trans.item(), loss_refined_trans.item(), 
                np.mean([v for _, v in rot_error.items()]), 
                np.mean([v for _, v in refined_rot_error.items()]), 
                np.mean([v for _, v in trans_error.items()]), 
                np.mean([v for _, v in refined_trans_error.items()]), 
                np.mean([v for _, v in rot_all.items()]), 
                )
        )

        if mode == "Train":
            optimizer.zero_grad()
            lossR.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()

    ret_dict = {
        "lossR": lossR_all,
        "lossR_reg": lossR_reg_all,
        "lossR_data": lossR_data_all,
        "lossR_cam_quat": lossR_cam_quat_all,
        "lossR_refined_cam_quat": lossR_refined_cam_quat_all,
        "loss_pose": loss_pose_all,
        "loss_trans": loss_trans_all,
        "loss_refined_pose": loss_refined_pose_all,
        "loss_refined_trans": loss_refined_trans_all,
        "rot_gt": [np.average(v) for _, v in rot_gt_all.items()],
        "rot_error": [np.average(v) for _, v in rot_error_all.items()],
        "trans_error": [np.average(v) for _, v in trans_error_all.items()],
        "refined_rot_error": [np.average(v) for _, v in refined_rot_error_all.items()],
        "refined_trans_error": [np.average(v) for _, v in refined_trans_error_all.items()],
    }
    for k in loss_refined_pose_dict_all.keys():
        refine_iter = k.split('_')[-1]
        ret_dict[f"loss_refined_pose_{refine_iter}"] = loss_refined_pose_dict_all[k]
        ret_dict[f"loss_refined_trans_{refine_iter}"] = loss_refined_trans_dict_all[k]

        ret_dict[f"refined_rot_error_iter_{refine_iter}"] = [np.average(v) for _, v in refined_rot_error_dict_all[k].items()]
        ret_dict[f"refined_trans_error_iter_{refine_iter}"] = [np.average(v) for _, v in refined_trans_error_dict_all[k].items()]

    if opt.ddp:
        print(f"{rank} total keys {len(ret_dict)}", flush=True)
        keys = ret_dict.keys()
        values = [torch.Tensor(ret_dict[k]).mean() for k in ret_dict.keys()]
        values = torch.Tensor(values).cuda()

        print(f"{rank} synchronize", flush=True)
        dist.barrier(device_ids=[torch.cuda.current_device()])
        dist.all_reduce(values, op=dist.ReduceOp.AVG)
        ret_dict = {k: v.item() for k, v in zip(keys, values)}

    for k in rot_error_all.keys():
        ret_dict[f"rot_gt_{k}"] = rot_gt_all[k]
        ret_dict[f"rot_error_{k}"] = rot_error_all[k]
        ret_dict[f"trans_error_{k}"] = trans_error_all[k]
        ret_dict[f"refined_rot_error_{k}"] = refined_rot_error_all[k]
        ret_dict[f"refined_trans_error_{k}"] = refined_trans_error_all[k]

    if rank == 0 and not opt.eval_only:
        print("Writing summary writer...", flush=True)
        summary_writer.add_scalar(f'{mode}/lr', scheduler.get_last_lr()[0], epoch)
        for k, v in ret_dict.items():
            summary_writer.add_scalar(f'{mode}/{k}', np.average(v), epoch)

    if opt.eval_only:
        header = ["category", "rot_error (angle)", "rot_gt (angle)", "acc@15", "acc@30"]
        df = []
        error_list = refined_rot_error_all_list
        for k in error_list.keys():
            print(len(error_list[k]))
            df.append([
                k,
                np.average(error_list[k]),
                np.average(rot_gt_all_list[k]),
                np.average(np.array(error_list[k]) < 15) * 100,
                np.average(np.array(error_list[k]) < 30) * 100,
            ])
        
        r_e_l = np.concatenate([v for _, v in error_list.items()])
        df.append([
            "all",
            np.average(np.average([np.average(v) for _, v in error_list.items()])),
            np.average(np.average([np.average(v) for _, v in rot_gt_all_list.items()])),
            np.average(r_e_l < 15) * 100,
            np.average(r_e_l < 30) * 100,
        ])
        print(df)
        with open(os.path.join(opt.outf, "eval_result.csv"), 'w') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile) 
                
            # writing the fields 
            csvwriter.writerow(header) 
                
            # writing the data rows 
            csvwriter.writerows(df)

    if epoch % opt.visualization_epoch == 0 and rank == 0 and mode == "Train":
        num_images = Xa.shape[0]
        textures = Ae['textures']

        Xa = (Xa * 255).view(batch_size * num_views, C, H, W).permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
        Xer = (Xer * 255).permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)

        Xa = torch.tensor(Xa, dtype=torch.float32) / 255.0
        Xa = Xa.permute(0, 3, 1, 2)
        Xer = torch.tensor(Xer, dtype=torch.float32) / 255.0
        Xer = Xer.permute(0, 3, 1, 2)

        # visualize projected points
        pred_rel_pose = geo_utils.quat2mat(poses_pred)
        canonical = absolute_poses[0][:1]
        pred_absolute_pose = geo_utils.transform_relative_pose(canonical, pred_rel_pose).view(batch_size, num_views - 1, 4, 4)
        pred_absolute_pose = torch.cat([canonical[None].expand(batch_size, -1, -1, -1), pred_absolute_pose], dim=1)
        pred_absolute_pose = pred_absolute_pose.view(batch_size * num_views, 4, 4)

        R_kaolin = pred_absolute_pose[:, :3, :3]
        T_kaolin = pred_absolute_pose[:, 3, :3]
        R_opencv, T_opencv = geo_utils.kaolin2opencv(R_kaolin, T_kaolin)
        RT_opencv = torch.cat([R_opencv, T_opencv.unsqueeze(-1)], dim=-1)
        vertices = Ae["vertices"]
        focal_length = focal_length
        k = torch.zeros([batch_size, num_views, 3, 3]).to(vertices.device)
        k[:, :, 0, 0] = focal_length[:, :, 0] * min(W, H) // 2
        k[:, :, 1, 1] = focal_length[:, :, 1] * min(W, H) // 2
        k[:, :, 0, 2] = W // 2
        k[:, :, 1, 2] = H // 2
        k[:, :, 2, 2] = 1
        k = k.view(-1, 3, 3)

        projected_pts, dpt = geo_utils.project_points_torch(
            vertices + center.repeat_interleave(num_views, dim=0).unsqueeze(1), 
            RT_opencv, 
            k
        )    
        normalized_pts = projected_pts.clone().detach()        
        normalized_pts[:, :, 0] = normalized_pts[:, :, 0] / W * 2 - 1
        normalized_pts[:, :, 1] = normalized_pts[:, :, 1] / H * 2 - 1

        rgb_feat = F.grid_sample(Xb.view(batch_size * num_views, C, H, W), normalized_pts.unsqueeze(-2)).squeeze()

        Xb_flat = Xb.view(batch_size*num_views, -1, H, W)

        fig, axs = plt.subplots(len(Xb_flat), 3, figsize=(12, len(Xb_flat) * 4))

        for i in range(len(Xb_flat)):
            ax = axs[i][0]
            im = Xb_flat[i, :3].permute(1, 2, 0).detach().cpu()
            ax.imshow(im)
            ax.set_title("Input")

            projected_pts_cpu = projected_pts[i].cpu().data

            ax = axs[i][1]
            im = Xb_flat[i, :3].permute(1, 2, 0).detach().cpu()
            im = np.ones_like(im)
            ax.imshow(im)
            ax.scatter(projected_pts_cpu[:, 0], projected_pts_cpu[:, 1], c=rgb_feat[i][:3].T.cpu().data.numpy(), s=1)
            ax.set_title(f"Using predicted pose {round(rot_error_batch[i - i // num_views - 1], 2) if i % num_views != 0 else 0}")
            
            ax = axs[i][2]
            im = Xb_flat[i, :3].permute(1, 2, 0).detach().cpu()
            ax.imshow(im)
            ax.scatter(projected_pts_cpu[:, 0], projected_pts_cpu[:, 1])
            ax.set_title("Project mesh to input")
            
        plt.savefig('%s/epoch_%03d_Iter_%04d_Xb_projected.jpg' % (opt.outf, epoch, iter))


        randperm_a = torch.randperm(batch_size)
        randperm_b = torch.randperm(batch_size)


        vutils.save_image(Xa[randperm_a, :3],
                '%s/epoch_%03d_Iter_%04d_randperm_Xa.png' % (opt.outf, epoch, iter), normalize=True)
        vutils.save_image(Xa[randperm_a, :3],
                '%s/current_randperm_Xa.png' % (opt.outf), normalize=True)

        vutils.save_image(Xa[randperm_b, :3],
                '%s/epoch_%03d_Iter_%04d_randperm_Xb.png' % (opt.outf, epoch, iter), normalize=True)
        vutils.save_image(Xa[randperm_b, :3],
                '%s/current_randperm_Xb.png' % (opt.outf), normalize=True)

        vutils.save_image(Xa[:, :3],
                '%s/epoch_%03d_Iter_%04d_Xa.png' % (opt.outf, epoch, iter), normalize=True, nrow=num_views)
        vutils.save_image(Xa[:, :3],
                '%s/current_Xa.png' % (opt.outf), normalize=True, nrow=num_views)
        
        vutils.save_image(Xb.view(batch_size * num_views, C, H, W)[:, :3],
                '%s/epoch_%03d_Iter_%04d_Xb.png' % (opt.outf, epoch, iter), normalize=True, nrow=num_views)
        vutils.save_image(Xb.view(batch_size * num_views, C, H, W)[:, :3],
                '%s/current_Xb.png' % (opt.outf), normalize=True, nrow=num_views)

        vutils.save_image(Xer[:, :3].detach(),
                '%s/epoch_%03d_Iter_%04d_Xer.png' % (opt.outf, epoch, iter), normalize=True, nrow=num_views)
        vutils.save_image(Xer[:, :3].detach(),
                '%s/current_Xer.png' % (opt.outf), normalize=True, nrow=num_views)


        vutils.save_image(textures.detach(),
                '%s/current_textures.png' % (opt.outf), normalize=True)

        l = np.array(range(0, batch_size*num_views, num_views))
        original = l.copy()
        Ae = deep_copy(Ae, index=original, detach=True)
        vertices = Ae['vertices']
        faces = diffRender.faces
        textures = Ae['textures']
        # azimuths = Ae['azimuths']
        # elevations = Ae['elevations']
        # distances = Ae['distances']
        lights = Ae['lights']

        texure_maps = to_pil_image(textures[0].detach().cpu())
        texure_maps.save('%s/current_mesh_recon.png' % (opt.outf), 'PNG')
        texure_maps.save('%s/epoch_%03d_mesh_recon.png' % (opt.outf, epoch), 'PNG')

        tri_mesh = trimesh.Trimesh(vertices[0].detach().cpu().numpy(), faces.detach().cpu().numpy())
        tri_mesh.export('%s/current_mesh_recon.obj' % opt.outf)
        tri_mesh.export('%s/epoch_%03d_mesh_recon.obj' % (opt.outf, epoch))

        rotate_path = os.path.join(opt.outf, 'epoch_%03d_rotation.gif' % epoch)
        writer = imageio.get_writer(rotate_path, mode='I')
        loop = tqdm.tqdm(list(range(-int(opt.azi_scope/2), int(opt.azi_scope/2), 10)))
        loop.set_description('Drawing Dib_Renderer SphericalHarmonics')
        for delta_azimuth in loop:
            Ae['azimuths'] = - torch.tensor([delta_azimuth], dtype=torch.float32).repeat(batch_size).cuda()
            Ae['elevations'] = torch.zeros_like(Ae['azimuths']) + 30
            Ae['distances'] = torch.zeros_like(Ae['azimuths']) + 2.7
            predictions, _ = diffRender.render(**Ae)
            predictions = predictions[:, :3]
            image = vutils.make_grid(predictions)
            image = image.permute(1, 2, 0).detach().cpu().numpy()
            image = (image * 255.0).astype(np.uint8)
            writer.append_data(image)
        writer.close()
        current_rotate_path = os.path.join(opt.outf, 'current_rotation.gif')
        shutil.copyfile(rotate_path, current_rotate_path)


    return ret_dict


def train_main(rank, size):
    device = torch.device("cuda:{}".format(rank))
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{rank}"

    if opt.dataset == "bird":
        train_dataset = BirdDataset(opt.dataroot, opt.imageSize, train=True)
        test_dataset = BirdDataset(opt.dataroot, opt.imageSize, train=False)
    elif opt.dataset == "co3d":
        train_dataset = Co3DDataset(opt.dataroot, opt.imageSize, train=True, categories=["bottle"], amodal=opt.amodal)
        test_dataset = Co3DDataset(opt.dataroot, opt.imageSize, train=False, categories=["bottle"], amodal=opt.amodal)
    # elif opt.dataset == "co3d_seq":
    #     train_dataset = Co3DSeqDataset(opt.dataroot, "train", image_size=opt.imageSize, categories=["toyplane"], amodal=opt.amodal)
    #     test_dataset = Co3DSeqDataset(opt.dataroot, "train", image_size=opt.imageSize, categories=["toyplane"], amodal=opt.amodal)
    elif opt.dataset == "co3d_seq":
        # train_dataset = Co3DSeqNormalizedDataset(opt.dataroot, "train_frame_split", categories=["toyplane"], sample_mode="near20")
        # test_dataset = Co3DSeqNormalizedDataset(opt.dataroot, "val_frame_split", categories=["toyplane"], sample_mode="near20")
        train_dataset = Co3DSeqNormalizedDataset(opt.dataroot, "train", categories=TRAINING_CATEGORIES)
        test_dataset = Co3DSeqNormalizedDataset(opt.dataroot, "val", categories=TRAINING_CATEGORIES)
        # train_dataset = Co3DSeqNormalizedDataset(opt.dataroot, "train", categories=["toyplane", "teddybear", "bottle", "bowl", 
        #                                                                             "cup", "laptop", "mouse", "remote"])
        # test_dataset = Co3DSeqNormalizedDataset(opt.dataroot, "val", categories=["toyplane", "teddybear", "bottle", "bowl", 
        #                                                                             "cup", "laptop", "mouse", "remote"])
    elif opt.dataset == "shapenet":
        train_dataset = ShapeNetMultiView(opt.dataroot, "train", category_ids=["02691156"])
        test_dataset = ShapeNetMultiView(opt.dataroot, "val", category_ids=["02691156"])
        test_dataset.object_path = test_dataset.object_path[:10]
    else:
        raise NotImplementedError
    
    if opt.ddp:
        training_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, rank=rank, shuffle=True, drop_last=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, rank=rank, shuffle=False)
    else:
        training_sampler = test_sampler = None

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize, sampler=training_sampler,
                                            shuffle=None if opt.ddp else True, drop_last=False if not opt.ddp else True, pin_memory=True, num_workers=int(opt.workers), persistent_workers=True if opt.workers > 0 else False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize, sampler=test_sampler,
                                            shuffle=None if opt.ddp else False, pin_memory=True, num_workers=int(opt.workers), persistent_workers=True if opt.workers > 0 else False)

    print(f"Rank {rank}, dataset: {len(train_dataset)}, dataloader: {len(train_dataloader)}")

    # differentiable renderer
    diffRender = DiffRender(filename_obj=opt.template_path, image_size=opt.imageSize)

    if opt.model == "SMR":
        # netE: 3D attribute encoder: Camera, Light, Shape, and Texture
        netE = AttributeEncoder(num_vertices=diffRender.num_vertices, vertices_init=diffRender.vertices_init, 
                                azi_scope=opt.azi_scope, elev_range=opt.elev_range, dist_range=opt.dist_range, 
                                nc=4, nk=opt.nk, nf=opt.nf)
    elif opt.model == "MeshFormer":
        if opt.norm_layer == "InstanceNorm2d":
            norm_layer = nn.InstanceNorm2d
        elif opt.norm_layer is None:
            norm_layer = None
        else:
            raise NotImplementedError

        netE = MeshFormerEncoder(num_vertices=diffRender.num_vertices, vertices_init=diffRender.vertices_init, 
                                azi_scope=opt.azi_scope, elev_range=opt.elev_range, dist_range=opt.dist_range,
                                nc=4, nk=opt.nk, nf=opt.nf, refine_pose=opt.refine_pose, norm_layer=norm_layer)
    else:
        raise NotImplementedError
    
    netE = netE.cuda()

    # setup optimizer
    optimizerE = optim.Adam(list(netE.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

    # if resume is True, restore from latest_ckpt.path
    start_iter = 0
    start_epoch = 0

    if opt.pretrained_path is not None:
        if rank == 0:
            resume_path = os.path.join(opt.pretrained_path, 'ckpts/latest_ckpt.pth')
            print("=> loading checkpoint '{}'".format(resume_path))
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(resume_path)
            start_iter = 0
            netE.load_state_dict({k.split("module.")[-1]: v for k, v in checkpoint['netE'].items()}, strict=False)

            print("=> loaded pretrained checkpoint '{}' (epoch {})"
                .format(resume_path, checkpoint['epoch']))

    if opt.resume:
        resume_path = os.path.join(opt.outf, 'ckpts/latest_ckpt.pth')
        # if rank == 0:
        if os.path.exists(resume_path):
            print("=> loading checkpoint '{}'".format(opt.resume))
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']
            start_iter = 0
            netE.load_state_dict({k.split("module.")[-1]: v for k, v in checkpoint['netE'].items()})

            optimizerE.load_state_dict(checkpoint['optimizerE'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(resume_path, checkpoint['epoch']))
        else:
            start_iter = 0
            start_epoch = 0
            print("=> no checkpoint can be found")

    if opt.ddp:
        netE = DDP(netE, find_unused_parameters=True)

    # setup learning rate scheduler
    schedulerE = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerE, T_max=opt.niter, eta_min=1e-6, last_epoch=start_epoch - 1)
    # schedulerE = torch.optim.lr_scheduler.StepLR(optimizerE, step_size=200, gamma=0.5)

    ori_dir = os.path.join(opt.outf, 'fid/ori')
    rec_dir = os.path.join(opt.outf, 'fid/rec')
    inter_dir = os.path.join(opt.outf, 'fid/inter')
    ckpt_dir = os.path.join(opt.outf, 'ckpts')

    if rank == 0:
        os.makedirs(ori_dir, exist_ok=True)
        os.makedirs(rec_dir, exist_ok=True)
        os.makedirs(inter_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        summary_writer = SummaryWriter(os.path.join(opt.outf + "/logs"))
    else: 
        summary_writer = None

    if opt.eval_only:
        netE.eval()
        with torch.no_grad():
            forward_pass(start_epoch, rank, netE, optimizerE, schedulerE, diffRender, test_dataloader, summary_writer, mode="Val")
    else:
        for epoch in range(start_epoch, opt.niter):
            if training_sampler is not None:
                training_sampler.set_epoch(epoch)
            netE.train()
            forward_pass(epoch, rank, netE, optimizerE, schedulerE, diffRender, train_dataloader, summary_writer, mode="Train")

            if epoch % 2 == 0 and rank == 0:
                epoch_name = os.path.join(ckpt_dir, 'epoch_%05d.pth' % epoch)
                latest_name = os.path.join(ckpt_dir, 'latest_ckpt.pth')
                state_dict = {
                    'epoch': epoch,
                    'netE': netE.state_dict(),
                    'optimizerE': optimizerE.state_dict(),
                }
                torch.save(state_dict, latest_name)

            if epoch % opt.val_epoch == 0 and epoch > 0:
                netE.eval()
                with torch.no_grad():
                    forward_pass(epoch, rank, netE, optimizerE, schedulerE, diffRender, test_dataloader, summary_writer, mode="Val")

            schedulerE.step()


def init_process(rank, local_rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend, rank=rank, world_size=size)
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)
    fn(rank, size)

if __name__ == '__main__':
    if opt.ddp:
        rank          = int(os.environ["SLURM_PROCID"])
        world_size    = int(os.environ["WORLD_SIZE"])
        gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        assert gpus_per_node == torch.cuda.device_count()
        print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
            f" {gpus_per_node} allocated GPUs per node.", flush=True)
        mp.set_start_method("spawn")
        local_rank = rank - gpus_per_node * (rank // gpus_per_node)
        p = mp.Process(target=init_process, args=(rank, local_rank, world_size, train_main))
        p.start()
        p.join()
    else:
        train_main(0, 1)
from typing import List

import torch
from torch import nn
from .position_encoding import PositionEmbeddingSine
from .backbone import Backbone, Joiner
from .transformer import Transformer, TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder
from .misc import NestedTensor
import math
import numpy as np
from .pose_estimator_2D import PoseEstimator2D, PoseEstimator2DFeat
import geo_utils
from utils import camera_position_from_spherical_angles, generate_transformation_matrix, compute_gradient_penalty, Timer, spherical_angles_from_camera_position
import torch.nn.functional as F


class CameraDecoder(nn.Module):
    def __init__(self, transformer_hidden_dim, decoder, azi_scope, elev_range, dist_range):
        super(CameraDecoder, self).__init__()

        self.azi_scope = float(azi_scope)

        elev_range = elev_range.split('~')
        elev_offset = -30
        self.elev_min = float(elev_range[0]) + elev_offset
        self.elev_max = float(elev_range[1]) + elev_offset

        dist_range = dist_range.split('~')
        dist_offset = -2
        self.dist_min = float(dist_range[0]) + dist_offset
        self.dist_max = float(dist_range[1]) + dist_offset
        self.camera_query_embedding = nn.Embedding(1, transformer_hidden_dim)
        self.decoder = decoder
        linear1 = self.linearblock(transformer_hidden_dim, 1024)
        linear2 = self.linearblock(1024, 1024)
        all_blocks = linear1 + linear2
        self.encoder2 = nn.Sequential(*all_blocks)
        self.linear = nn.Linear(transformer_hidden_dim, 4)
        self.batch_norm = nn.BatchNorm1d(transformer_hidden_dim)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) \
            or isinstance(m, nn.Linear) \
            or isinstance(object, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, mean=0, std=0.001)

    def atan2(self, y, x):
        r = torch.sqrt(x**2 + y**2 + 1e-12) + 1e-6
        phi = torch.sign(y) * torch.acos(x / r) * 180.0 / math.pi
        return phi
    
    def linearblock(self, indim, outdim):
        block2 = [
            nn.Linear(indim, outdim),
            nn.BatchNorm1d(outdim),
            nn.ReLU()
        ]
        return block2

    def forward(self, tgt, memory, pos_embed, mask):
        batch_size = memory.shape[1]
        # memory = memory.permute(1, 2, 0).contiguous().view(batch_size, 256, 8, 8)
        # camera_output = self.avg_pool(memory).view(batch_size, -1)
        # camera_output = self.encoder2(camera_output)
        # camera_output = self.linear(camera_output)

        query_embed = self.camera_query_embedding.weight

        query_embed = query_embed.unsqueeze(1).repeat(
            1, batch_size, 1
        )
        if tgt is None:
            tgt = torch.zeros_like(query_embed)
        decoder_output = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )
        decoder_output = decoder_output[0][0]
        # decoder_output = self.batch_norm(decoder_output)
        # decoder_output = nn.ReLU()(decoder_output)
        # decoder_output = self.encoder2(decoder_output)
        camera_output = self.linear(decoder_output)

        # cameras
        distances = self.dist_min + torch.sigmoid(camera_output[:, 0]) * (self.dist_max - self.dist_min)
        elevations = self.elev_min + torch.sigmoid(camera_output[:, 1]) * (self.elev_max - self.elev_min)
        # azimuths = -180 + torch.sigmoid(camera_output[:, 2]) * self.azi_scope

        azimuths_x = camera_output[:, 2]
        azimuths_y = camera_output[:, 3]
        # azimuths = 90.0 - self.atan2(azimuths_y, azimuths_x)
        azimuths = - self.atan2(azimuths_y, azimuths_x) / 360.0 * self.azi_scope

        cameras = [azimuths, elevations, distances]
        return cameras


class Attention(nn.Module):
    def __init__(self, dim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        #self.scale = 10 #head_dim ** -0.5

    def get_attn(self, query, key):
        B, N, C = query.shape
        attn = torch.matmul(query, key.transpose(-2, -1)) #* self.scale
        attn = attn.softmax(dim=-1)
        #__import__('pdb').set_trace()
        return attn
    
    def forward(self, query, key, value):
        B, N, C = query.shape
        query = query.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        key = key.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        value = value.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = torch.matmul(query, key.transpose(-2, -1)) #* self.scale
        attn = attn.softmax(dim=-1)
        x = torch.matmul(attn, value).transpose(1, 2).reshape(B, N, C)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)


class Block(nn.Module):
    def __init__(self, dim, num_heads=1, mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm, return_attn=False):
        super().__init__()

        self.channels = dim

        self.encode_query = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)
        self.encode_key = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)

        self.attn = Attention(dim, num_heads=num_heads)
        
        self.encode_value = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        self.norm = norm_layer(dim)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor)
    
    def get_attn(self, query, key, query_embed=None, key_embed=None):
        b, c, n = query.shape

        q = self.with_pos_embed(query, query_embed)
        k = self.with_pos_embed(key, key_embed)

        q = self.norm(q.permute(0, 2, 1)).permute(0, 2, 1)
        k = self.norm(k.permute(0, 2, 1)).permute(0, 2, 1)

        q = self.encode_query(q).view(b, self.channels, -1)
        q = q.permute(0, 2, 1)
        k = self.encode_key(k).view(b, self.channels, -1)
        k = k.permute(0, 2, 1)
        return self.attn.get_attn(query=q, key=k)   # [b,n,n]
    
    def forward(self, query, key, query_embed=None, key_embed=None):
        b, c, n = query.shape

        q = self.with_pos_embed(query, query_embed)
        k = self.with_pos_embed(key, key_embed)

        q = self.norm(q.permute(0, 2, 1)).permute(0, 2, 1)
        k = self.norm(k.permute(0, 2, 1)).permute(0, 2, 1)

        v = self.encode_value(key).view(b, self.channels, -1)
        v = v.permute(0, 2, 1)

        q = self.encode_query(q).view(b, self.channels, -1)
        q = q.permute(0, 2, 1)

        k = self.encode_key(k).view(b, self.channels, -1)
        k = k.permute(0, 2, 1)

        query = query.view(b, self.channels, -1).permute(0, 2, 1)
        query = query + self.attn(query=q, key=k, value=v)

        query = query + self.mlp(self.norm2(query))
        query = query.permute(0, 2, 1).contiguous().view(b, self.channels, -1)

        return query
    

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return torch.from_numpy(pos_embed)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


class PositionalEncoding(nn.Module):
    def __init__(self, num_octaves=8, start_octave=0):
        super().__init__()
        self.num_octaves = num_octaves
        self.start_octave = start_octave

    def forward(self, coords, rays=None):
        embed_fns = []
        batch_size, num_points, dim = coords.shape

        octaves = torch.arange(self.start_octave, self.start_octave + self.num_octaves)
        octaves = octaves.float().to(coords)
        multipliers = 2**octaves * math.pi
        coords = coords.unsqueeze(-1)
        while len(multipliers.shape) < len(coords.shape):
            multipliers = multipliers.unsqueeze(0)

        scaled_coords = coords * multipliers

        sines = torch.sin(scaled_coords).reshape(batch_size, num_points, dim * self.num_octaves)
        cosines = torch.cos(scaled_coords).reshape(batch_size, num_points, dim * self.num_octaves)

        result = torch.cat((sines, cosines), -1)
        return result


class PoseTransformer(nn.Module):
    def __init__(self, inp_res=32, dim=64, mlp_ratio=1, coord_dim=64):
        super().__init__()
        
        self.coord_dim = coord_dim

        self.cross_transformer = Block(dim=dim, mlp_ratio=mlp_ratio, return_attn=True)
        self.self_transformer = Block(dim=dim, mlp_ratio=mlp_ratio, return_attn=False)
        
        # self.penc3d_module = PositionalEncoding3D(channels=self.coord_dim)
        # _ = self.penc3d_module(torch.zeros(1,inp_res,inp_res,inp_res,self.coord_dim))
        # self.pos_embed_3d_coord = self.penc3d_module.cached_penc / 6     # [1,D,H,W,32]
        # del self.penc3d_module
        self.pos_embed_2d_coord = get_2d_sincos_pos_embed(coord_dim, inp_res) * 0.1  # [D,H,W,32]

        self.pos_embed_2d_coord = self.pos_embed_2d_coord.reshape(1,-1,self.coord_dim)


    def forward(self, q, k, q_pe=None, k_pe=None):
        '''
        q and k: volume features in shape [B,16,N]
        '''
        B,C,N = q.shape
        pe = self.pos_embed_2d_coord.to(q)                      # [1,N,C]
        attn = self.cross_transformer.get_attn(query=q, key=k)  # [B,N,N]
        coord = torch.matmul(attn, pe)                          # [B,N,C]
        coord = coord.permute(0,2,1)                            # [B,C,N]
        coord = self.self_transformer(query=coord, key=coord)   # [B,C,N]
        return coord


class PairwisePoseEstimator(nn.Module):
    def __init__(self, transformer_hidden_dim, encoder, azi_scope, elev_range, dist_range):
        super(PairwisePoseEstimator, self).__init__()

        self.azi_scope = float(azi_scope)

        elev_range = elev_range.split('~')
        elev_offset = -30
        self.elev_min = float(elev_range[0]) + elev_offset
        self.elev_max = float(elev_range[1]) + elev_offset

        dist_range = dist_range.split('~')
        dist_offset = -2
        self.dist_min = float(dist_range[0]) + dist_offset
        self.dist_max = float(dist_range[1]) + dist_offset
        self.encoder = PoseTransformer(inp_res=8, dim=256, coord_dim=256)

        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 512, 3, padding=1, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
        )

        self.conv2d_3 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1, stride=2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
        )

        self.pose_head_1 = nn.Sequential(*[
            nn.Conv2d(1024, 2048, 3, padding=1, stride=2),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(2048, 2048, 3, padding=1),]
        )
        self.pose_head_2 = nn.Sequential(*[
            #nn.BatchNorm3d(1024),
            nn.LayerNorm(2048),
            nn.LeakyReLU(inplace=True),]
        )
        self.out = nn.Sequential(*[
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 4)
        ])

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) \
            or isinstance(m, nn.Linear) \
            or isinstance(object, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, mean=0, std=0.001)
    
    def linearblock(self, indim, outdim):
        block2 = [
            nn.Linear(indim, outdim),
            nn.BatchNorm1d(outdim),
            nn.ReLU()
        ]
        return block2

    def forward(self, features, mask, pos_embed):
        b,t,C1,H1,W1 = features.shape                                        # spatial size 8

        features = features.reshape(b*t,C1,H1,W1)
        _,C,H,W = features.shape
        N = H * W

        features = features.reshape(b,t,C,N)                                    # [b,t,C,N]
        features_ref = features[:,0:1].repeat(1,t-1,1,1).reshape(b*(t-1),C,N)   # [b*(t-1),C,N]
        features_cur = features[:,1:].reshape(b*(t-1),C,N)

        x = self.encoder(q=features_ref, k=features_cur)

        x = x.reshape(b*(t-1), C, H, W)
        
        x = self.conv2d_2(x)
        x = self.conv2d_3(x)     # [B,1024,4,4,4]
        # x = self.pose_head(x).view(b*(t-1), self.pose_dim + 1)
        x = self.pose_head_2(self.pose_head_1(x).squeeze())       # [b*(t-1),1024]
        camera_output = self.out(x) 

        # cameras
        distances = self.dist_min + torch.sigmoid(camera_output[:, 0]) * (self.dist_max - self.dist_min)
        elevations = self.elev_min + torch.sigmoid(camera_output[:, 1]) * (self.elev_max - self.elev_min)
        azimuths = -180 + torch.sigmoid(camera_output[:, 2]) * self.azi_scope

        # azimuths_x = camera_output[:, 2]
        # azimuths_y = camera_output[:, 3]
        # # azimuths = 90.0 - self.atan2(azimuths_y, azimuths_x)
        # azimuths = - self.atan2(azimuths_y, azimuths_x) / 360.0 * self.azi_scope

        cameras = [azimuths, elevations, distances]
        return cameras

class ShapeDecoder(nn.Module):
    def __init__(self, transformer_hidden_dim, decoder, num_vertices):
        super(ShapeDecoder, self).__init__()
        # self.vertices_encoder = nn.Linear(3, transformer_hidden_dim)
        # self.self_attn = nn.MultiheadAttention(transformer_hidden_dim, 2, dropout=0.1)
        # self.linear = nn.Linear(transformer_hidden_dim, 3)

        self.decoder = decoder
        self.num_vertices = num_vertices
        self.shape_embed = nn.Embedding(1, transformer_hidden_dim)
        self.linear = nn.Linear(transformer_hidden_dim, num_vertices * 3)


        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) \
            or isinstance(m, nn.Linear) \
            or isinstance(object, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, mean=0, std=0.001)

    def linearblock(self, indim, outdim):
        block2 = [
            nn.Linear(indim, outdim),
            nn.BatchNorm1d(outdim),
            nn.ReLU()
        ]
        return block2

    def forward(self, memory, vertices_init, pos_embed, mask):
        batch_size = memory.shape[1]
        # query_embed = self.vertices_encoder(vertices_init)
        # query_embed = self.self_attn(query_embed, query_embed, query_embed)[0]
        query_embed = self.shape_embed.weight

        query_embed = query_embed.unsqueeze(1).repeat(
            1, batch_size, 1
        )
        tgt = torch.zeros_like(query_embed)
        decoder_output = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )
        shape_output = self.linear(decoder_output)[0]

        delta_vertices = shape_output.view(batch_size, self.num_vertices, 3)
        # delta_vertices = shape_output.permute(1, 0, 2).contiguous()
        # delta_vertices = torch.tanh(delta_vertices)

        return delta_vertices
    

class LightDecoder(nn.Module):
    def __init__(self, transformer_hidden_dim, decoder):
        super(LightDecoder, self).__init__()

        self.decoder = decoder
        self.light_query_embedding = nn.Embedding(1, transformer_hidden_dim)
        self.linear = nn.Linear(transformer_hidden_dim, 9)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) \
            or isinstance(m, nn.Linear) \
            or isinstance(object, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, mean=0, std=0.001)

    def forward(self, memory, pos_embed, mask):
        batch_size = memory.shape[1]
        query_embed = self.light_query_embedding.weight

        query_embed = query_embed.unsqueeze(1).repeat(
            1, batch_size, 1
        )
        tgt = torch.zeros_like(query_embed)
        decoder_output = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )        
        light_output = self.linear(decoder_output)[0][0]

        lightparam = torch.tanh(light_output)
        scale = torch.tensor([[0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]], dtype=torch.float32).cuda()
        bias = torch.tensor([[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32).cuda()
        lightparam = lightparam * scale + bias
    
        return lightparam
    

class TextureDecoder(nn.Module):
    def __init__(self, transformer_hidden_dim, decoder):
        super(TextureDecoder, self).__init__()
        self.decoder = decoder
        self.transformer_hidden_dim = transformer_hidden_dim
        self.texture_embed = nn.Embedding(1, transformer_hidden_dim)
        self.linear = nn.Linear(transformer_hidden_dim, 3)
        self.uv_map_resolution = 32
        self.position_embedding_sine = PositionEmbeddingSine(transformer_hidden_dim // 2)

        self.texture_flow = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv2d(transformer_hidden_dim, transformer_hidden_dim, 3, 1, 1, padding_mode='reflect'),
            nn.BatchNorm2d(transformer_hidden_dim),
            nn.ReLU(True),

            # state size. (nf*8) x 8 x 8
            nn.Upsample(scale_factor=2),
            nn.Conv2d(transformer_hidden_dim, 3, 3, 1, 1, padding_mode='reflect'),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )

        # Initialize with Xavier Glorot
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) \
            or isinstance(m, nn.Linear) \
            or isinstance(object, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, mean=0, std=0.001)

    def forward(self, memory, pos_embed, mask):
        batch_size = memory.shape[1]
        query_embed = self.texture_embed.weight

        query_embed = query_embed.unsqueeze(1).repeat(
            1, batch_size, 1
        )

        t = torch.zeros([batch_size, self.transformer_hidden_dim, self.uv_map_resolution, self.uv_map_resolution]).to(memory.device)
        m = torch.zeros_like(t)[:, 0, :, :].bool()
        pixel_pos_embed = self.position_embedding_sine(NestedTensor(t, m))
        pixel_pos_embed = pixel_pos_embed.view(batch_size, self.transformer_hidden_dim, -1)
        pixel_pos_embed = pixel_pos_embed.permute(2, 0, 1)

        query_embed = query_embed + pixel_pos_embed
        tgt = torch.zeros_like(query_embed)
        decoder_output = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )        
        textures = decoder_output[0]
        textures = textures.permute(1, 2, 0).contiguous().view(batch_size, self.transformer_hidden_dim, self.uv_map_resolution, self.uv_map_resolution)
        textures = self.texture_flow(textures)

        textures_flip = textures.flip([2])
        textures = torch.cat([textures, textures_flip], dim=2)
        return textures


class MeshFormer(nn.Module):
    """This is the base class for Transformer based mesh reconstruction"""

    def __init__(self, num_vertices, vertices_init, azi_scope, elev_range, dist_range):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_vertices = num_vertices
        self.vertices_init = vertices_init

        self.backbone = build_backbone("resnet18")
        # self.transformer = build_transformer(nheads=4, enc_layers=2, dec_layers=2)
        hidden_dim = 256
        pre_norm = False

        d_model=hidden_dim + 3
        dropout=0.1
        nhead=2
        dim_feedforward=2048
        num_encoder_layers = 2
        num_decoder_layers = 2
        normalize_before=pre_norm,
        return_intermediate_dec=False
        divide_norm=False
        activation="relu"
        encoder_layer = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            divide_norm=divide_norm,
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        self.bottleneck = nn.Conv2d(
            self.backbone.num_channels, hidden_dim, kernel_size=1
        ) 
        self._reset_parameters()

        # shape decoder
        decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            divide_norm=divide_norm,
        )
        decoder_norm = nn.LayerNorm(d_model)

        decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        self.shape_decoder = ShapeDecoder(d_model, decoder, num_vertices)

        # camera decoder
        decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            divide_norm=divide_norm,
        )
        decoder_norm = nn.LayerNorm(d_model)

        decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        self.camera_decoder = CameraDecoder(d_model, decoder, azi_scope, elev_range, dist_range)

        # light decoder
        decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            divide_norm=divide_norm,
        )
        decoder_norm = nn.LayerNorm(d_model)

        decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        self.light_decoder = LightDecoder(d_model, decoder)

        # texture decoder
        decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            divide_norm=divide_norm,
        )
        decoder_norm = nn.LayerNorm(d_model)

        decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        self.texture_decoder = TextureDecoder(d_model, decoder)


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def adjust(self, output_back: List, pos_embed: List):
        """ """
        src_feat, mask = output_back[-1].decompose()
        assert mask is not None
        # reduce channel
        feat = self.bottleneck(src_feat)  # (B, C, H, W)
        # adjust shapes
        feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec = pos_embed[-1].flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec = mask.flatten(1)  # BxHW
        return {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}

    def forward(self, x):
        device = x.device
        if not isinstance(x, NestedTensor):
            m = torch.zeros_like(x)
            m = m[:, 0, :, :]
            x = NestedTensor(x, m)
            # only need it for each pixel
            
        output_back, pos = self.backbone(
            x 
        )
        output = self.adjust(output_back, pos)
        feat = output["feat"]
        mask = output["mask"]
        pos_embed = output["pos"]
        memory = self.encoder(feat, src_key_padding_mask=mask, pos=pos_embed)

        delta_vertices = self.shape_decoder(memory, self.vertices_init.to(device), pos_embed, mask)
        cameras = self.camera_decoder(memory, pos_embed, mask)
        lights = self.light_decoder(memory, pos_embed, mask)
        textures = None
        # textures = self.texture_decoder(memory, pos_embed, mask)
        return delta_vertices, cameras, lights, textures


class MultiViewMeshFormer(nn.Module):
    """This is the base class for Transformer based mesh reconstruction"""

    def __init__(self, num_vertices, vertices_init, azi_scope, elev_range, dist_range):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_vertices = num_vertices
        self.vertices_init = vertices_init

        self.backbone = build_backbone("resnet18")
        # self.transformer = build_transformer(nheads=4, enc_layers=2, dec_layers=2)
        hidden_dim = 256
        pre_norm = False
        self.pose_embed_size = 7

        d_model=hidden_dim
        dropout=0.1
        nhead=1
        dim_feedforward=2048
        num_encoder_layers = 2
        num_decoder_layers = 2
        normalize_before=pre_norm,
        return_intermediate_dec=False
        divide_norm=False
        activation="relu"
        encoder_layer = TransformerEncoderLayer(
            d_model,   # gt_pose
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            divide_norm=divide_norm,
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        self.bottleneck = nn.Conv2d(
            self.backbone.num_channels, hidden_dim, kernel_size=1
        ) 

        # fuse layer
        encoder_layer = TransformerEncoderLayer(
            d_model + self.pose_embed_size,   # gt_pose
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            divide_norm=divide_norm,
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        encoder_norm = nn.LayerNorm(d_model + self.pose_embed_size) if normalize_before else None
        self.fuse = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        # pos_estimator
        encoder_layer = TransformerEncoderLayer(
            d_model + 3,   # gt_pose
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            divide_norm=divide_norm,
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        encoder_norm = nn.LayerNorm(d_model + 3) if normalize_before else None
        encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )
        # self.pose_estimator = PairwisePoseEstimator(d_model, encoder, azi_scope, elev_range, dist_range)
        # self.pose_estimator = PoseEstimator2D()
        self.pose_estimator = PoseEstimator2DFeat()

        self._reset_parameters()

        # shape decoder
        decoder_layer = TransformerDecoderLayer(
            d_model + self.pose_embed_size,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            divide_norm=divide_norm,
        )
        decoder_norm = nn.LayerNorm(d_model + self.pose_embed_size)

        decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        self.shape_decoder = ShapeDecoder(d_model + self.pose_embed_size, decoder, num_vertices)

        # camera decoder
        decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            divide_norm=divide_norm,
        )
        decoder_norm = nn.LayerNorm(d_model)

        decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        self.camera_decoder = CameraDecoder(d_model, decoder, azi_scope, elev_range, dist_range)

        # light decoder
        decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            divide_norm=divide_norm,
        )
        decoder_norm = nn.LayerNorm(d_model)

        decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        self.light_decoder = LightDecoder(d_model, decoder)

        # texture decoder
        decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            divide_norm=divide_norm,
        )
        decoder_norm = nn.LayerNorm(d_model)

        decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        self.texture_decoder = TextureDecoder(d_model, decoder)
        # self.camera_pos_embeding = PositionalEncoding(num_octaves=8)


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def adjust(self, output_back: List, pos_embed: List):
        """ """
        src_feat, mask = output_back[-1].decompose()
        assert mask is not None
        # reduce channel
        feat = self.bottleneck(src_feat)  # (B, C, H, W)
        # adjust shapes
        feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec = pos_embed[-1].flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec = mask.flatten(1)  # BxHW
        return {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}

    def forward(self, x, gt_poses):
        device = x.device
        batch_size, num_views, C, H, W = x.shape
        img = x

        x = x.view(batch_size * num_views, C, H, W)
        if not isinstance(x, NestedTensor):
            m = torch.zeros_like(x)
            m = m[:, 0, :, :]
            x = NestedTensor(x, m)
            # only need it for each pixel
            
        output_back, pos = self.backbone(
            x 
        )
        output = self.adjust(output_back, pos)
        feat = output["feat"]
        mask = output["mask"]
        pos_embed = output["pos"]
        shape_feat = feat
        shape_pos_embed = pos_embed

        # feat_h, feat_w = 8, 8
        # cameras = self.pose_estimator(feat.permute(1, 2, 0).contiguous().view(batch_size, num_views, -1, feat_h, feat_w), mask.view(batch_size, num_views, feat_h, feat_w), pos_embed.permute(1, 2, 0).view(batch_size, num_views, -1, feat_h, feat_w))
        # gt_poses = torch.stack(cameras, dim=0).permute(1, 0).contiguous().view(batch_size, num_views - 1, -1)
        # canonical = torch.FloatTensor([0, 30, 3]).to(gt_poses.device).repeat(batch_size, 1, 1)
        # gt_poses = gt_poses + canonical
        # gt_poses = torch.cat([canonical, gt_poses], dim=1)

        # feat_h, feat_w = 8, 8
        # pred_cameras = self.pose_estimator(feat.permute(1, 2, 0).contiguous().view(batch_size, num_views, -1, feat_h, feat_w), mask.view(batch_size, num_views, feat_h, feat_w), pos_embed.permute(1, 2, 0).view(batch_size, num_views, -1, feat_h, feat_w))
        # pred_cameras = torch.stack(pred_cameras, dim=0).permute(1, 0).contiguous().view(batch_size, num_views - 1, -1)
        # canonical = torch.FloatTensor([0, 30, 3]).to(gt_poses.device).repeat(batch_size, 1, 1)
        # pred_cameras = pred_cameras + canonical
        # pred_cameras = torch.cat([canonical, pred_cameras], dim=1)
        # cameras = gt_poses.view(batch_size * num_views, -1)
        # cameras = [cameras[:, 0], cameras[:, 1], cameras[:, 2]]

        lights = None
        
        # camera prediction
        HW, _, dim = feat.shape
        tgt = feat.permute(1, 2, 0).contiguous()
        tgt = self.avg_pool(tgt)
        tgt = tgt.permute(2, 0, 1).contiguous()
        feat = feat.view(HW, batch_size, num_views, dim).permute(0, 2, 1, 3).contiguous().view(HW * num_views, batch_size, dim)
        mask = mask.view(batch_size, num_views * HW)
        # pos_embed = torch.nn.functional.pad(pos_embed, (0, 3, 0, 0, 0, 0))  # pad for the added gt_pose
        pos_embed = pos_embed.view(HW, batch_size, num_views, dim).permute(0, 2, 1, 3).contiguous().view(HW * num_views, batch_size, dim)

        memory = self.encoder(feat, src_key_padding_mask=mask, pos=pos_embed)
        # shape_memory = memory
        # shape_pos_embed = pos_embed
        # shape_memory = torch.cat([memory, gt_poses.permute(1, 0, 2).repeat_interleave(HW, dim=0)], dim=2)
        # shape_pos_embed = torch.nn.functional.pad(pos_embed, (0, 3, 0, 0, 0, 0))  # pad for the added gt_pose

        # pred_cameras = self.camera_decoder(tgt, memory.repeat_interleave(num_views, dim=1), pos_embed.repeat_interleave(num_views, dim=1), mask.repeat_interleave(num_views, dim=0))
        # pred_cameras = torch.stack(pred_cameras, dim=0).permute(1, 0).contiguous().view(batch_size, num_views, -1)
        # canonical = torch.FloatTensor([0, 30, 3]).to(gt_poses.device)
        # pred_cameras[:, 0] = canonical
        # pred_cameras[:, 1:] = pred_cameras[:, 1:] + canonical
        # gt_poses = torch.stack(cameras, dim=0).permute(1, 0).contiguous().view(batch_size, num_views, -1)
        # canonical = torch.FloatTensor([0, 30, 3]).to(gt_poses.device)
        # gt_poses[:, 0] = canonical
        # gt_poses[:, 1:] = gt_poses[:, 1:] + canonical
        cameras = gt_poses.view(batch_size * num_views, -1)
        cameras = [cameras[:, 0], cameras[:, 1], cameras[:, 2]]

        # pred_cameras = self.pose_estimator(img[:, :, :3, :, :])
        pred_cameras = self.pose_estimator(output["feat"].permute(1, 2, 0).contiguous().view(batch_size, num_views, dim, 8, 8))

        # pred camera pose to spherical coordinate
        canonical = torch.FloatTensor([[ 1.0000,  0.0000,  0.0000,  0.0000],
            [ 0.0000,  0.8660, -0.5000,  0.0000],
            [ 0.0000,  0.5000,  0.8660, -3.0000],
            [ 0.0000,  0.0000,  0.0000,  1.0000]]
        ).to(pred_cameras.device)
        canonical_quat = torch.FloatTensor([ 0.9659,  0.2588,  0.0000,  0.0000,  0.0000,  0.0000, -3.0000]).to(pred_cameras.device)
        tmp = torch.zeros_like(pred_cameras)
        tmp[:,:4] = F.normalize(pred_cameras[:,:4])
        tmp[:,4:] = pred_cameras[:,4:]
        poses_pred = tmp

        pred_cam_quat = poses_pred.view(batch_size, num_views - 1, -1)
        canonical_quat = canonical_quat.unsqueeze(0).repeat(batch_size, 1).unsqueeze(1)
        pred_cam_quat = torch.cat([canonical_quat, pred_cam_quat], dim=1)

        camPoseRel_cv2 = geo_utils.quat2mat(poses_pred)              # [b*(t-1),4,4], relative cam pose in cv2 frame
        camPoseAbsolute = canonical.unsqueeze(0) @ camPoseRel_cv2

        rot = camPoseAbsolute[:, :3, :3]
        trans = camPoseAbsolute[:, :3, 3]
        pred_camera_pos = (- trans.unsqueeze(1) @ rot).squeeze(1)
        pred_cam_sphere = spherical_angles_from_camera_position(pred_camera_pos)
        pred_cam_sphere = torch.stack(pred_cam_sphere, dim=1)
        canonical_sphere = torch.FloatTensor([0, 30, 3]).to(pred_cam_sphere.device).unsqueeze(0).repeat(batch_size, 1).unsqueeze(1)
        pred_cam_sphere = pred_cam_sphere.view(batch_size, num_views - 1, -1)
        pred_cam_sphere = torch.cat([canonical_sphere, pred_cam_sphere], dim=1)


        # shape fuse and prediction
        # camera_pos_embed = self.camera_pos_embeding(gt_poses.view(batch_size * num_views, -1).unsqueeze(0))[0].view(batch_size, num_views, -1)
        shape_feat = torch.cat([shape_feat, pred_cam_quat.view(num_views*batch_size, -1).unsqueeze(0).repeat(64, 1, 1)], dim=2)
        shape_pos_embed = pos_embed
        HW, _, dim = shape_feat.shape
        tgt = shape_feat.permute(1, 2, 0).contiguous()
        tgt = self.avg_pool(tgt)
        tgt = tgt.permute(2, 0, 1).contiguous()
        shape_feat = shape_feat.view(HW, batch_size, num_views, dim).permute(0, 2, 1, 3).contiguous().view(HW * num_views, batch_size, dim)
        mask = mask.view(batch_size, num_views * HW)
        shape_pos_embed = torch.nn.functional.pad(shape_pos_embed, (0, self.pose_embed_size, 0, 0, 0, 0))  # pad for the added gt_pose
        shape_pos_embed = shape_pos_embed.view(HW, batch_size, num_views, dim).permute(0, 2, 1, 3).contiguous().view(HW * num_views, batch_size, dim)
        shape_memory = self.fuse(shape_feat, src_key_padding_mask=mask, pos=shape_pos_embed)

        delta_vertices = self.shape_decoder(shape_memory, self.vertices_init.to(device), shape_pos_embed, mask)
        

        # lights = self.light_decoder(memory, pos_embed, mask)
        # lights = lights.repeat_interleave(num_views, dim=0)

        textures = None
        # textures = self.texture_decoder(memory, pos_embed, mask)

        # repeat shape prediction for every item
        delta_vertices = delta_vertices.repeat_interleave(num_views, dim=0)

        return delta_vertices, cameras, lights, textures, pred_cameras, [pred_cam_sphere.view(num_views*batch_size, -1)[:, 0], pred_cam_sphere.view(num_views*batch_size, -1)[:, 1], pred_cam_sphere.view(num_views*batch_size, -1)[:, 2]]


def build_backbone(backbone_type="resnet50", train_backbone=True):
    position_embedding = build_position_encoding()
    backbone = Backbone(
        backbone_type,
        train_backbone,
        False,
        False,
        False, # since the input channel is 4 instead of 3, we need to train the entire network
        None,
        pretrained=False,
        input_dim=4
    )
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels

    return model

def build_position_encoding(hidden_dim=256):
    N_steps = hidden_dim // 2
    position_embedding = PositionEmbeddingSine(N_steps, normalize=True)

    return position_embedding

def build_transformer(hidden_dim=256, nheads=8, dim_feedforward=2048, enc_layers=6, dec_layers=6, pre_norm=False):
    return Transformer(
        d_model=hidden_dim,
        dropout=0.1,
        nhead=nheads,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        normalize_before=pre_norm,
        return_intermediate_dec=False,  # we use false to avoid DDP error,
        divide_norm=False,
    )

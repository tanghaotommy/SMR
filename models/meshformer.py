from typing import List

import torch
from torch import nn
from .position_encoding import PositionEmbeddingSine
from .backbone import Backbone, Joiner
from .transformer import Transformer, TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder
from .misc import NestedTensor
import math


class CameraDecoder(nn.Module):
    def __init__(self, transformer_hidden_dim, decoder, azi_scope, elev_range, dist_range):
        super(CameraDecoder, self).__init__()

        self.azi_scope = float(azi_scope)

        elev_range = elev_range.split('~')
        self.elev_min = float(elev_range[0])
        self.elev_max = float(elev_range[1])

        dist_range = dist_range.split('~')
        self.dist_min = float(dist_range[0])
        self.dist_max = float(dist_range[1])
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

        azimuths_x = camera_output[:, 2]
        azimuths_y = camera_output[:, 3]
        # azimuths = 90.0 - self.atan2(azimuths_y, azimuths_x)
        azimuths = - self.atan2(azimuths_y, azimuths_x) / 360.0 * self.azi_scope

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

        d_model=hidden_dim
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
        self.shape_decoder = ShapeDecoder(hidden_dim, decoder, num_vertices)

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
        self.camera_decoder = CameraDecoder(hidden_dim, decoder, azi_scope, elev_range, dist_range)

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
        self.light_decoder = LightDecoder(hidden_dim, decoder)

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
        self.texture_decoder = TextureDecoder(hidden_dim, decoder)


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

        d_model=hidden_dim
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
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
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
        self.shape_decoder = ShapeDecoder(hidden_dim, decoder, num_vertices)

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
        self.camera_decoder = CameraDecoder(hidden_dim, decoder, azi_scope, elev_range, dist_range)

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
        self.light_decoder = LightDecoder(hidden_dim, decoder)

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
        self.texture_decoder = TextureDecoder(hidden_dim, decoder)


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
        batch_size, num_views, C, H, W = x.shape

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
        
        HW, _, dim = feat.shape
        tgt = feat.permute(1, 2, 0).contiguous()
        tgt = self.avg_pool(tgt)
        tgt = tgt.permute(2, 0, 1).contiguous()
        feat = feat.view(HW, batch_size, num_views, dim).permute(0, 2, 1, 3).contiguous().view(HW * num_views, batch_size, dim)
        mask = mask.view(batch_size, num_views * HW)
        pos_embed = pos_embed.view(HW, batch_size, num_views, dim).permute(0, 2, 1, 3).contiguous().view(HW * num_views, batch_size, dim)

        memory = self.encoder(feat, src_key_padding_mask=mask, pos=pos_embed)

        delta_vertices = self.shape_decoder(memory, self.vertices_init.to(device), pos_embed, mask)
        cameras = self.camera_decoder(tgt, memory.repeat_interleave(num_views, dim=1), pos_embed.repeat_interleave(num_views, dim=1), mask.repeat_interleave(num_views, dim=0))
        lights = self.light_decoder(memory, pos_embed, mask)
        textures = None
        # textures = self.texture_decoder(memory, pos_embed, mask)

        # repeat shape prediction for every item
        delta_vertices = delta_vertices.repeat_interleave(num_views, dim=0)
        lights = lights.repeat_interleave(num_views, dim=0)

        return delta_vertices, cameras, lights, textures


def build_backbone(backbone_type="resnet50", train_backbone=True):
    position_embedding = build_position_encoding()
    backbone = Backbone(
        backbone_type,
        train_backbone,
        False,
        False,
        False, # since the input channel is 4 instead of 3, we need to train the entire network
        None,
        pretrained=False
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

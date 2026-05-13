# calculate the center of the patch, then calcualte the positional embedding
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import repeat
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from einops.layers.torch import Reduce
from chamfer_dist import ChamferDistanceL1
import copy
import numpy as np
import math
from functools import partial

import torch
import timm.models.vision_transformer

from timm.models.vision_transformer import PatchEmbed

from timm.models.vision_transformer import Block as VitBlock

# from zeta.nn import SSM
from typing import Optional
from timm.models.layers import DropPath, to_2tuple
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
# from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.5,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        # import ipdb; ipdb.set_trace()
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
        
    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
                
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype)) #self.norm(residual.to(dtype=self.norm.weight.dtype))
            
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )    
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    d_state=16,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=torch.float32,
    if_bimamba=False,
    bimamba_type="none",
    if_devide_out=False,
    init_layer_scale=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    # import ipdb; ipdb.set_trace()
    print(f"drop:{drop_path}")
    mixer_cls = partial(Mamba, d_state=d_state, layer_idx=layer_idx, bimamba_type=bimamba_type, if_devide_out=if_devide_out, init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block



class Mesh_mamba_seg(nn.Module):
    def __init__(self, masking_ratio=0.75, channels=13, num_heads=12, encoder_depth=12, embed_dim=768,
                 decoder_num_heads=16, decoder_depth=6, decoder_embed_dim=512,
                 patch_size=64, norm_layer=nn.LayerNorm, seg_part=4, drop_path=0.5, fpn=False, face_pos=False):
        super(Mesh_mamba_seg, self).__init__()
        patch_dim = channels
        self.num_patches = 256
        self.face_pos = face_pos
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h p -> b h (p c)', p=patch_size),
            nn.Linear((patch_dim) * patch_size, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.to_face_embedding = nn.Sequential(
            Rearrange('b c h p -> b h p c', p=patch_size),
            nn.Linear(patch_dim if not self.face_pos else patch_dim+3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.masking_ratio = masking_ratio
        # MAE encoder specifics
        
        self.blocks = nn.ModuleList([
            create_block(
                d_model=embed_dim,
                drop_path=drop_path,
                norm_epsilon=1e-5,
                rms_norm=False,
                residual_in_fp32=True,
                fused_add_norm=False,
                layer_idx=i,
                device=None,
                dtype=torch.float32,
                bimamba_type="None", #"v2"
                if_devide_out=False,
                init_layer_scale=None,
            ) for i in range(encoder_depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        self.pos_embedding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )
        self.fpn = fpn
        if self.fpn:
            self.linears = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim)), 
                                        nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim)),
                                        nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))])
            
        self.max_pooling = nn.MaxPool2d((64, 1))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, seg_part)
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, seg_part)  # Updated for 50 segmentation labels
        )
        self.head1 = nn.Sequential(
            nn.Linear(embed_dim * 2, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, seg_part)  # Updated for 50 segmentation labels
        )
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, faces, feats, centers, Fs, cordinates):
        feats_patches = feats
        centers_patches = centers
        batch, channel, num_patches, *_ = feats_patches.shape
        cordinates_patches = cordinates
        center_of_patches = torch.sum(centers_patches, dim=2) / 64
        pos_emb = self.pos_embedding(center_of_patches)
        center_of_patches = center_of_patches.unsqueeze(2).repeat(1,1,64,1)
        feats_patches = feats.permute(0,3,1,2)

        tokens = self.to_patch_embedding(feats_patches)
        if not self.face_pos:
            tokens_seg = self.to_face_embedding(feats_patches)
        else:
            face_pos = (centers_patches - center_of_patches).permute(0,3,1,2)
            tokens_seg = self.to_face_embedding(torch.cat([feats_patches, face_pos], dim=1))

        cls_tokens = self.cls_token.expand(feats_patches.shape[0], -1, -1)

        tokens = tokens + pos_emb
        tokens = torch.cat((tokens, cls_tokens), dim=1)
        # patch to encoder tokens

        tokens_s = []
        residual = tokens
        for i, blk in enumerate(self.blocks):
            tokens, residual = blk(tokens, residual)
            if i % 4 == 3:
                tokens_s.append(tokens)
        
        if self.fpn:
            tokens = 0
            for l, t in zip(self.linears, tokens_s):
                tokens = tokens + l(t)
        
        x = self.norm(tokens)
        outcome = x[:, 0:-1]
        outcome = outcome.unsqueeze(2).repeat(1, 1, 64, 1)
        x = self.head(outcome)
        tokens_seg = torch.cat((tokens_seg, outcome), dim=3)
        x_seg = self.head1(tokens_seg)
        return x, x_seg

class Mesh_mae_mamba(nn.Module):
    def __init__(self, masking_ratio=0.75, channels=13, num_heads=12, encoder_depth=12, embed_dim=768,
                 decoder_num_heads=16, decoder_depth=6, decoder_embed_dim=512,
                 patch_size=64, norm_layer=nn.LayerNorm, weight=0.2, drop_path=0.2):
        super(Mesh_mae_mamba, self).__init__()
        patch_dim = channels
        self.num_patches = 256
        self.weight = weight
        self.pos_embedding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )
        self.embed_dim = embed_dim
        self.decoer_pos_embedding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, decoder_embed_dim)
        )
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h p -> b h (p c)', p=patch_size),
            nn.Linear(patch_dim * patch_size, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.masking_ratio = masking_ratio
        # MAE encoder specifics
        self.blocks = nn.ModuleList([
            create_block(
                d_model=embed_dim,
                drop_path=drop_path,
                norm_epsilon=1e-5,
                rms_norm=False,
                residual_in_fp32=False,
                fused_add_norm=False,
                layer_idx=i,
                device=None,
                dtype=torch.float32,
                if_bimamba=False,
                bimamba_type="none",
                if_devide_out=False,
                init_layer_scale=None,
            ) for i in range(encoder_depth)
        ])
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            VitBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        # --------------------------------------------------------------------------

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.to_points = nn.Linear(decoder_embed_dim, 64 * 9)
        self.to_pointsnew = nn.Linear(decoder_embed_dim, 45 * 3)
        self.to_points_seg = nn.Linear(decoder_embed_dim, 9)
        self.to_features = nn.Linear(decoder_embed_dim, 64 * (channels))
        self.to_features_seg = nn.Linear(decoder_embed_dim, channels)
        self.build_loss_func()
        self.initialize_weights()
        self.decoder_cls_token_pos = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.encoder_cls_token_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.max_pooling = nn.MaxPool2d((256, 1))

    def build_loss_func(self):
        self.loss_func_cdl1 = ChamferDistanceL1().cuda()

    def initialize_weights(self):

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, faces, feats, centers, Fs, cordinates):

        minFs = min(Fs)
        min_patch_number = minFs / 64

        min_patch_number = int(min_patch_number.detach().cpu().numpy())
        feats_patches = feats
        centers_patches = centers
        center_of_patches = torch.sum(centers_patches, dim=2) / 64
        batch, channel, num_patches, *_ = feats_patches.shape

        cordinates_patches = cordinates
        pos_emb = self.pos_embedding(center_of_patches)

        encoder_cls_token_pos = self.encoder_cls_token_pos.repeat(batch, 1, 1)

        tokens = self.to_patch_embedding(feats_patches)

        num_masked = int(self.masking_ratio * min_patch_number)

        rand_indices = torch.rand(batch, min_patch_number).argsort(dim=-1).cuda()

        left_indices = torch.rand(batch, num_patches - min_patch_number).argsort(dim=-1).cuda() + min_patch_number

        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        unmasked_indices = torch.cat((unmasked_indices, left_indices), dim=1)

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch)[:, None]
        tokens_unmasked = tokens[batch_range, unmasked_indices]
        cls_tokens = self.cls_token.expand(feats_patches.shape[0], -1, -1)
        tokens_unmasked = torch.cat((tokens_unmasked, cls_tokens), dim=1)
        pos_emb_a = torch.cat((pos_emb[batch_range, unmasked_indices], encoder_cls_token_pos), dim=1)
        tokens_unmasked = tokens_unmasked + pos_emb_a
        # print(tokens_unmasked.shape)
        # encoded_tokens = self.blocks(tokens_unmasked)
        residual = None
        for blk in self.blocks:
            #tokens_unmasked = blk(tokens_unmasked)
            tokens_unmasked, residual = blk(tokens_unmasked, residual)
        tokens_unmasked = self.norm(tokens_unmasked)
        encoded_tokens = tokens_unmasked

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.decoder_embed(encoded_tokens)
        mask_tokens = self.mask_token.repeat(batch, num_masked, 1)
        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim=1)

        decoder_pos_emb = self.decoer_pos_embedding(center_of_patches)

        decoder_cls_token_pos = self.decoder_cls_token_pos.repeat(batch, 1, 1)
        decoder_pos_emb = torch.cat((decoder_pos_emb[batch_range, masked_indices],
                                     decoder_pos_emb[batch_range, unmasked_indices], decoder_cls_token_pos), dim=1)
        decoder_tokens = decoder_tokens + decoder_pos_emb
        # decoded_tokens = self.decoder_blocks(decoder_tokens)
        for blk in self.decoder_blocks:
            decoder_tokens = blk(decoder_tokens)
        decoded_tokens = decoder_tokens
        decoded_tokens = self.decoder_norm(decoded_tokens)

        # splice out the mask tokens and project to pixel values
        recovered_tokens = decoded_tokens[:, :num_masked]
        pred_vertices_coordinates = self.to_pointsnew(recovered_tokens)
        faces_values_per_patch = feats_patches.shape[-1]
        pred_vertices_coordinates = torch.reshape(pred_vertices_coordinates,
                                                  (batch, num_masked, 45, 3)).contiguous()

        # get the patches to be masked for the final reconstruction loss
        # print(pred_vertices_coordinates.shape, torch.sum(centers_patches[batch_range,masked_indices],dim=2).shape)
        center = torch.sum(centers_patches[batch_range, masked_indices], dim=2) / 64
        pred_vertices_coordinates = pred_vertices_coordinates + center.unsqueeze(2).repeat(1, 1, 45, 1)
        pred_vertices_coordinates = torch.reshape(pred_vertices_coordinates, (batch * num_masked, 45, 3)).contiguous()
        cordinates_patches = cordinates_patches[batch_range, masked_indices]

        cordinates_patches = torch.reshape(cordinates_patches, (batch, num_masked, -1, 3)).contiguous()
        cordinates_unique = torch.unique(cordinates_patches, dim=2)
        cordinates_unique = torch.reshape(cordinates_unique, (batch * num_masked, -1, 3)).contiguous()
        masked_feats_patches = feats_patches[batch_range, :, masked_indices]

        pred_faces_features = self.to_features(recovered_tokens)
        pred_faces_features = torch.reshape(pred_faces_features, (batch, num_masked, channel, faces_values_per_patch))

        # calculate reconstruction loss
        # print(pred_vertices_coordinates.shape, cordinates_unique.shape)

        shape_con_loss, _, _ = self.loss_func_cdl1(pred_vertices_coordinates, cordinates_unique)

        feats_con_loss = F.mse_loss(pred_faces_features, masked_feats_patches)
        # print(shape_con_loss, feats_con_loss)
        loss = feats_con_loss + self.weight * shape_con_loss
        #######################################################################
        # if you are going to show the reconstruct shape, please using the following codes
        # pred_vertices_coordinates = pred_vertices_coordinates.reshape(batch, num_masked, -1, 3)
        #return loss, masked_indices, unmasked_indices, pred_vertices_coordinates, cordinates
        #######################################################################
        return loss
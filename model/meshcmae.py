# calculate the center of the patch, then calcualte the positional embedding
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from chamfer_dist import ChamferDistanceL1
import torch
from timm.models.vision_transformer import PatchEmbed, Block


class Mesh_encoder(nn.Module):
    def __init__(self, masking_ratio=0.75, channels=13, num_heads=12, encoder_depth=12, embed_dim=384,
                 decoder_num_heads=16, decoder_depth=6, decoder_embed_dim=512, drop_path=0.1,
                 patch_size=64, norm_layer=nn.LayerNorm):
        super(Mesh_encoder, self).__init__()
        patch_dim = channels
        self.num_patches = 250
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h p -> b h (p c)', p=patch_size),
            nn.Linear(patch_dim * patch_size, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.dim = embed_dim
        self.masking_ratio = masking_ratio
        # MAE encoder specifics
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer,drop_path=drop_path)
            for i in range(encoder_depth)])
        self.norm = norm_layer(embed_dim)
        self.pos_embedding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )

        self.max_pooling = nn.MaxPool2d((64, 1))
        #self.max_pooling2 = nn.MaxPool2d((256, 1))
        self.max_pooling2 = nn.AdaptiveMaxPool2d((1, 384))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #self.cls_token_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.initialize_weights()

    def initialize_weights(self):

        torch.nn.init.normal_(self.cls_token, std=.02)
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

        feats_patches = feats
        centers_patches = centers

        center_of_patches = torch.sum(centers_patches, dim=2) / 64

        pos_emb = self.pos_embedding(center_of_patches)

        batch, channel, num_patches, *_ = feats_patches.shape
        tokens = self.to_patch_embedding(feats_patches)

        tokens = tokens + pos_emb

        for blk in self.blocks:
            tokens = blk(tokens)

        x = self.norm(tokens)
        zero_tokens = torch.zeros((batch, 0, self.dim), dtype=torch.float32).cuda()
        tokens = torch.cat((x, zero_tokens), dim=1)   # 32, 256, 384
        tokens = self.max_pooling2(tokens).squeeze(1)   # 32, 384
        return tokens


class EnhancedCrossAttention(nn.Module):
    def __init__(self, feat_dim, pos_dim, embed_dim, num_heads=8):
        super().__init__()
        # 更深的投影层增强表达能力
        self.query_proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(embed_dim)
        )
        self.key_proj = nn.Sequential(
            nn.Linear(pos_dim, embed_dim),
            nn.GELU(), 
            nn.Dropout(0.3),
            nn.LayerNorm(embed_dim)
        )
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        # 残差融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + feat_dim, embed_dim),
            nn.Dropout(0.3),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, feats, pos):
        """ 
        feats: (B, H, P, C)
        pos: (B, H, P, 3)
        """
        # 保持空间结构
        query = self.query_proj(feats)
        key = self.key_proj(pos)
        
        # 注意力计算
        attn_output, _ = self.multihead_attn(
            query=query.flatten(1,2),  # (B, H*P, E)
            key=key.flatten(1,2),
            value=key.flatten(1,2),
            need_weights=False
        )
        attn_output = attn_output.view_as(query)  # (B, H, P, E)
        
        # 残差融合原始特征
        combined = torch.cat([feats, attn_output], dim=-1)  # (B, H, P, C+E)
        return self.fusion(combined)  # (B, H, P, E)

class FaceEmbedding(nn.Module):
    def __init__(self, patch_dim, embed_dim, patch_size):
        super().__init__()
        self.rearrange = Rearrange('b c h p -> b h p c', p=patch_size)
        
        # 混合融合路径
        self.attn_path = EnhancedCrossAttention(
            feat_dim=patch_dim, 
            pos_dim=3,
            embed_dim=embed_dim
        )
        self.direct_path = nn.Sequential(
            nn.Linear(patch_dim + 3, embed_dim),
            nn.Dropout(0.3),
            nn.LayerNorm(embed_dim)
        )
        
        # 最终融合
        self.final_fusion = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, feats_patches, face_pos):
        # 原始特征处理
        feats = self.rearrange(feats_patches)  # (B, H, P, C)
        pos = self.rearrange(face_pos)         # (B, H, P, 3)
        
        # 双路径融合
        attn_feats = self.attn_path(feats, pos)
        direct_feats = self.direct_path(torch.cat([feats, pos], dim=-1))
        
        # 加权融合
        return self.final_fusion(torch.cat([attn_feats, direct_feats], dim=-1))

# MeshCMAE主架构
class Mesh_cmae(nn.Module):
    def __init__(self, masking_ratio=0.5, channels=10, num_heads=12, encoder_depth=12, embed_dim=768,
                 decoder_num_heads=16, decoder_depth=6, decoder_embed_dim=512,
                 patch_size=64, norm_layer=nn.LayerNorm, weight=0.2, contrastive_weight=0, ema_decay=0.999):
        super(Mesh_cmae, self).__init__()
        patch_dim = channels
        self.num_patches = 250
        self.weight = weight   # 
        self.contrastive_weight = contrastive_weight   # 总损失中对比损失权重 λ
        self.ema_decay = ema_decay   # EMA动量系数
        # 位置信息
        self.pos_embedding = nn.Sequential(
            nn.Linear(3, 128),   # 输入每个patch的几何中心 [24, 250, 64, 3]->[24, 250, 64, 128]
            nn.GELU(),
            nn.Linear(128, embed_dim)   # [24, 250, 64, 128]->[24, 250, 64, 768]
        )
        self.embed_dim = embed_dim
        self.decoer_pos_embedding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, decoder_embed_dim)
        )
        # patch->token
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h p -> b h (p c)', p=patch_size),   # 输入feats：[24, 13, 250, 64]->[24, 250, 13*64]，将patch中所有面的特征向量拼接
            nn.Linear(patch_dim * patch_size, embed_dim),   # [24, 250, 13*64]->[24, 250, 768]
            nn.LayerNorm(embed_dim)
        )
        self.masking_ratio = masking_ratio
        # MAE encoder specifics
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer)
            for i in range(encoder_depth)])
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer)
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
        #self.max_pooling2 = nn.MaxPool2d((128, 1))
        self.max_pooling2 = nn.AdaptiveMaxPool2d((1, 384))
        
        # Initialize target encoder and projection head
        self.target_encoder = Mesh_encoder(masking_ratio = self.masking_ratio)
        self.hidden_dim = 2048  # 增大隐层维度
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),  # 增加隐层
            nn.LayerNorm(self.hidden_dim),   # 添加LayerNorm
            nn.Linear(self.hidden_dim, embed_dim)
        )
        self.target_projection = nn.Sequential(
            nn.Linear(embed_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, embed_dim)
        )
        self.temperature = nn.Parameter(torch.tensor(0.07))   # τ

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

    def update_target_encoder(self):  # 动量更新机制EMA
        """动量更新目标网络"""
        with torch.no_grad():
            for param_online, param_target in zip(self.parameters(), self.target_encoder.parameters()):
                param_target.data = param_target.data * self.ema_decay + param_online.data * (1. - self.ema_decay)

    def forward(self, faces1, feats1, centers1, Fs1, cordinates1, faces2, feats2, centers2, Fs2, cordinates2):

        # -------------------------在线分支-----------------------------

        minFs = min(Fs1)   # 获取batch个样本中最少的面数   Fs1.shape=(21)，batch个样本的面数
        min_patch_number = minFs / 64   # 最少的patch数量，保证一个batch里，所有样本使用相同数量的patch
        min_patch_number = int(min_patch_number.detach().cpu().numpy())  # tensor张量->整数   250.->250

        feats1_patches = feats1
        centers1_patches = centers1
        centers1_of_patches = torch.sum(centers1_patches, dim=2) / 64
        batch, channel, num_patches, *_ = feats1_patches.shape
        cordinates1_patches = cordinates1

        pos_emb = self.pos_embedding(centers1_of_patches)
        encoder_cls_token_pos = self.encoder_cls_token_pos.repeat(batch, 1, 1)
        tokens = self.to_patch_embedding(feats1_patches)   # 生成patch token，tokens.shape=[24, 250, 768]
        num_masked = int(self.masking_ratio * min_patch_number)   # 根据掩码率计算掩码的patch数量

        # 将要掩码的patch遮住，其余的用来编码
        rand_indices = torch.rand(batch, min_patch_number).argsort(dim=-1).cuda()
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        left_indices = torch.rand(batch, num_patches - min_patch_number).argsort(dim=-1).cuda() + min_patch_number
        unmasked_indices = torch.cat((unmasked_indices, left_indices), dim=1)

        # get the unmasked tokens to be encoded 编码器只看未掩码的patch
        batch_range = torch.arange(batch)[:, None]
        tokens_unmasked = tokens[batch_range, unmasked_indices]
        tokens_unmasked = tokens_unmasked + pos_emb[batch_range, unmasked_indices]
        # print(tokens_unmasked.shape)
        # encoded_tokens = self.blocks(tokens_unmasked)

        #houyan
        # cnt=0
        for blk in self.blocks:   # 共12层，输入输出大小都是[32, 128, 384]
            # print(tokens_unmasked.shape)
            tokens_unmasked = blk(tokens_unmasked)
        #     print(tokens_unmasked.shape)
        #     cnt+=1
        #     print(cnt)
        # adak
        
        # Transformer编码：未掩码
        tokens_unmasked = self.norm(tokens_unmasked)
        encoded_tokens = tokens_unmasked

        # 解码：掩码+未掩码，补回被mask的patch
        decoder_tokens = self.decoder_embed(encoded_tokens)
        mask_tokens = self.mask_token.repeat(batch, num_masked, 1)
        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim=1)

        decoder_pos_emb = self.decoer_pos_embedding(centers1_of_patches)
        decoder_cls_token_pos = self.decoder_cls_token_pos.repeat(batch, 1, 1)
        decoder_pos_emb = torch.cat((decoder_pos_emb[batch_range, masked_indices],
                                     decoder_pos_emb[batch_range, unmasked_indices]), dim=1)
        decoder_tokens = decoder_tokens + decoder_pos_emb  # 32, 256, 512 加位置编码

        # Transformer解码
        for blk in self.decoder_blocks:
            decoder_tokens = blk(decoder_tokens)
        decoded_tokens = decoder_tokens
        decoded_tokens = self.decoder_norm(decoded_tokens)

        # splice out the mask tokens and project to pixel values
        recovered_tokens = decoded_tokens[:, :num_masked]
        pred_vertices_coordinates = self.to_pointsnew(recovered_tokens)   # 重建几何坐标 [batch * num_masked, 45, 3]
        faces_values_per_patch = feats1_patches.shape[-1]
        pred_vertices_coordinates = torch.reshape(pred_vertices_coordinates,
                                                  (batch, num_masked, 45, 3)).contiguous()

        # get the patches to be masked for the final reconstruction loss
        # print(pred_vertices_coordinates.shape, torch.sum(centers_patches[batch_range,masked_indices],dim=2).shape)
        center = torch.sum(centers1_patches[batch_range, masked_indices], dim=2) / 64
        pred_vertices_coordinates = pred_vertices_coordinates + center.unsqueeze(2).repeat(1, 1, 45, 1)
        pred_vertices_coordinates = torch.reshape(pred_vertices_coordinates, (batch * num_masked, 45, 3)).contiguous()
        
        cordinates1_patches = cordinates1_patches[batch_range, masked_indices]
        cordinates1_patches = torch.reshape(cordinates1_patches, (batch, num_masked, -1, 3)).contiguous()
        cordinates_unique = torch.unique(cordinates1_patches, dim=2)
        cordinates_unique = torch.reshape(cordinates_unique, (batch * num_masked, -1, 3)).contiguous()
        # print(pred_vertices_coordinates.shape, cordinates_unique.shape)

        masked_feats_patches = feats1_patches[batch_range, :, masked_indices]

        pred_faces_features = self.to_features(recovered_tokens)   # 重建特征
        pred_faces_features = torch.reshape(pred_faces_features, (batch, num_masked, channel, faces_values_per_patch))

        # 计算重建损失
        shape_con_loss = self.loss_func_cdl1(pred_vertices_coordinates, cordinates_unique)    # 形状重建损失 Chamfer Distance
        feats_con_loss = F.mse_loss(pred_faces_features, masked_feats_patches)    # 特征重建损失 MSE
        # print(shape_con_loss, feats_con_loss)
        loss = feats_con_loss + self.weight * shape_con_loss   # 重建损失
        #######################################################################
        # if you are going to show the reconstruct shape, please using the following codes
        # pred_vertices_coordinates = pred_vertices_coordinates.reshape(batch, num_masked, -1, 3)
        # return loss, masked_indices, unmasked_indices, pred_vertices_coordinates, cordinates
        #######################################################################
        
        # -------------------------目标分支-----------------------------

        target_tokens = self.target_encoder.forward(faces2, feats2, centers2, Fs2, cordinates2)
        
        encoded_tokens = self.max_pooling2(encoded_tokens)   # Shape will be (24, 250, 384)
        encoded_tokens = encoded_tokens.squeeze(1)   # Shape becomes (24, 384)

        #print(f"encod shpe:{encoded_tokens.shape}")  
        #print(f"tar shpe:{target_tokens.shape}")

        # -------------------------双分支-----------------------------

        # 投影头处理
        projected_online = self.projection(encoded_tokens)
        projected_target = self.target_projection(target_tokens)
        # L2规范化
        projected_online = F.normalize(projected_online, dim=-1)
        projected_target = F.normalize(projected_target, dim=-1)
        # 对比损失计算 InfoNCE
        logits = torch.einsum('bd,bd->b', projected_online, projected_target)  # 计算余弦相似度
        logits /= self.temperature  # 温度缩放
        contrastive_loss = -logits.mean()

        # 计算预训练总损失
        loss += self.contrastive_weight * contrastive_loss

        # Update target encoder
        # self.update_target_encoder()
        
        return loss

# 微调分割主架构
class Mesh_baseline_seg(nn.Module):
    def __init__(self, masking_ratio=0.5, channels=13, num_heads=12, encoder_depth=12, embed_dim=384,
                 decoder_num_heads=16, decoder_depth=6, decoder_embed_dim=512,
                 patch_size=64, norm_layer=nn.LayerNorm, seg_part=50, drop_path =0.4, fpn=False, face_pos=True):
        super(Mesh_baseline_seg, self).__init__()
        patch_dim = channels
        self.embed_dim = embed_dim
        self.num_patches = 250
        self.face_pos = face_pos
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h p -> b h (p c)', p=patch_size),
            nn.Linear((patch_dim) * patch_size, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.to_face_embedding = FaceEmbedding(
            patch_dim=patch_dim,
            embed_dim=embed_dim,
            patch_size=patch_size
        )
        self.masking_ratio = masking_ratio
        # MAE encoder specifics
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer)
            for i in range(encoder_depth)])
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
            nn.Dropout(p=0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, seg_part)  # Updated for 50 segmentation labels
        )
        self.head1 = nn.Sequential(
            nn.Linear(embed_dim * 2, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, seg_part)  # Updated for 50 segmentation labels
        )
        self.initialize_weights()
        self.cross_attn = nn.MultiheadAttention(embed_dim, decoder_num_heads, batch_first=True)
        
        self.convhead = nn.Conv2d(embed_dim, seg_part, kernel_size=1, stride=1, padding=0)
    def initialize_weights(self):

        torch.nn.init.normal_(self.cls_token, std=.02)
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
            tokens_seg = self.to_face_embedding(feats_patches, face_pos)

        cls_tokens = self.cls_token.expand(feats_patches.shape[0], -1, -1)

        tokens = tokens + pos_emb
        tokens = torch.cat((tokens, cls_tokens), dim=1)
        # patch to encoder tokens

        tokens_s = []
        for i, blk in enumerate(self.blocks):
            tokens = blk(tokens)
            if i % 4 == 3:
                tokens_s.append(tokens)
        
        if self.fpn:
            fused_features = []
            for t, l in zip(tokens_s, self.linears):
                fused_features.append(l(t))
            fused_features = torch.stack(fused_features, dim=-1)  # [B, N, C, L]
            fused_features = fused_features.mean(dim=-1)  # 改为平均融合（可选其他方式）
            # 或者使用注意力加权融合
            # attention_weights = self.fpn_attention(fused_features)
            # fused_features = (fused_features * attention_weights).sum(dim=-1)
            tokens = fused_features
        
        x = self.norm(tokens)
        outcome = x[:, 0:-1]
        
        outcome = outcome.unsqueeze(2).repeat(1, 1, 64, 1)
        
        x = self.head(outcome)
        tokens_seg = torch.cat((tokens_seg, outcome), dim=3)
        x_seg = self.head1(tokens_seg)
        return x, x_seg

"""
Model architecture for asymmetric lymph node matching.

This module implements the three main components:
- SE (Specificity Extraction): Multi-view 2D-3D feature extraction
- TE (Temporal Enhanced): Temporal contrastive learning
- SEAG (Spatial Enhanced Adaptive Graph): Graph neural network for spatial correlations
- SCB (Specificity-Correlation Balancing): Multi-view fusion and thresholded similarity
"""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from sconv_archs import SiameseNodeFeaturesToEdgeFeatures, SiameseSConvOnNodes
from src.utils.pad_tensor import pad_tensor
from config import cfg
from resnet50 import ResNet50
from moco.moco.builder3D_2D import MoCo

# ==================== Constants ====================
SOFTMAX_TEMP = 0.07
LOGIT_SCALE_MAX = 4.6052  # ln(100) ≈ 4.6052, corresponds to temperature 0.01
LOGIT_SCALE_MIN = 0.0     # corresponds to temperature 1.0
DROPOUT_RATE = 0.1
NUM_TOP_VIEWS = 2  # Number of top-2 views selected in SCB module


# ==================== Utility Functions ====================
def lexico_iter(lex):
    """Generate all pairs from a list."""
    from itertools import combinations
    return combinations(lex, 2)

class InnerProduct(nn.Module):
    def __init__(self, output_dim):
        super(InnerProduct, self).__init__()
        self.d = output_dim

    def _forward(self, X, Y):
        assert X.shape[1] == Y.shape[1] == self.d, (X.shape[1], Y.shape[1], self.d)
        X = torch.nn.functional.normalize(X, dim=-1)
        Y = torch.nn.functional.normalize(Y, dim=-1)
        res = torch.matmul(X, Y.transpose(0, 1))
        return res

    def forward(self, Xs, Ys):
        return [self._forward(X, Y) for X, Y in zip(Xs, Ys)]


class InnerProduct_encoder(nn.Module):
    def __init__(self, output_dim):
        super(InnerProduct_encoder, self).__init__()
        self.d = output_dim

    def _forward(self, X, Y):
        assert X.shape[1] == Y.shape[1] == self.d, (X.shape[1], Y.shape[1], self.d)
        # X = torch.nn.functional.normalize(X, dim=-1)
        # Y = torch.nn.functional.normalize(Y, dim=-1)
        res = torch.matmul(X, Y.transpose(0, 1))
        return res

    def forward(self, Xs, Ys):
        return [self._forward(X, Y) for X, Y in zip(Xs, Ys)]


class GMNet(nn.Module):
    """
    Main graph matching network implementing the asymmetric matching framework.
    
    According to the paper, this network consists of:
    - SE (Specificity Extraction): Multi-view 2D-3D feature extraction
    - TE (Temporal Enhanced): Temporal contrastive learning
    - SEAG (Spatial Enhanced Adaptive Graph): Graph neural network for spatial correlations
    - SCB (Specificity-Correlation Balancing): Multi-view fusion and thresholded similarity
    """
    def __init__(self, encoder, dim=128, encoder_dim=256):
        super(GMNet, self).__init__()
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=dim)
        self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(
            total_num_nodes=self.message_pass_node_features.num_node_features
        )
        self.vertex_affinity = InnerProduct(dim)
        self.vertex_affinity_encoder = InnerProduct_encoder(encoder_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / SOFTMAX_TEMP))
        self.encoder = MoCo().cuda()
        # Note: Paper specifies ResNet-34 for 2D, but code uses ResNet50
        # ResNet50 is used here for compatibility with existing trained models
        self.encoder_2d = ResNet50()
        # Transform layer: ModifiedResNet attention pool output dimension
        # Note: The actual dimension may vary based on ModifiedResNet implementation
        # Keeping 24*128 for compatibility with existing trained models
        self.transform_layer = nn.Linear(24 * 128, 2048)
        self.transform_layer_encode = nn.Linear(2048, encoder_dim)
        self.transform_layer_node = nn.Linear(2048, dim)
        # SCB: Gate network for discriminative view selection
        self.gate_2d = nn.Linear(dim, 1)
        
        # SCB: Gate network for weighted fusion
        self.gate = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
        # SCB: Cross-attention for feature fusion
        self.cross_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=1, batch_first=True)
        
        # SCB: Classifier for similarity scoring
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.BatchNorm1d(num_features=8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
        
        # SE: 3D encoder
        # Note: Paper specifies ResNet-151 for 3D, but code uses ModifiedResNet with layers=(3,4,6,3)
        # which is equivalent to ResNet-101, not ResNet-151 (which would be layers=(3,8,36,3))
        # This is kept for compatibility with existing trained models
        self.encoder_3d = ModifiedResNet(
            layers=(3, 4, 6, 3),  # ResNet-101 equivalent (paper specifies ResNet-151)
            output_dim=dim,
            heads=64 * 32 // 64,
            input_resolution=24,
            width=64
        )
        
        self.dim = dim
    
    def forward(self, data_dict):
        with torch.no_grad():
            self.logit_scale.clamp_(LOGIT_SCALE_MIN, LOGIT_SCALE_MAX)
        orig_graph_list = []
        orig_encoder_list = []
        orig_encoder_dropout = []
        # Extract input data
        images = data_dict['imgs']
        points = data_dict['Ps']
        n_points = data_dict['ns']
        graphs = data_dict['pyg_graphs']
        batch_size = data_dict['batch_size']
        images_2d = data_dict['imgs_2d']
        
        # Initialize storage for features
        time_node_features = []
        loss = 0
        t12_list_3d = []  # 3D features for T1 and T2
        t12_list_2d = []  # 2D features for T1 and T2
        t12_list_2d_top2 = []  # Top-2 2D features for SCB
        all_list1_3d = []  # Store all 3D features for return
        
        # Track which time point we're processing (0 for T1, 1 for T2)
        time_flag = 0
        
        for image, p, n_p, graph, image_2d in zip(images, points, n_points, graphs, images_2d):
            list1_3d = []
            list1_2d = []
            list1_2d_top2 = []
            encode_node_features = []
            encode_dropout_features = []
            
            for j in range(image.shape[0]):
                img = image[j]
                img_2d = image_2d[j]
                image_3d = []
                image1_2d = []
                image1_2d_top2 = []
                encode_list = []
                encode_dropout = []
                for i in range(n_p[j]):
                    # Extract 3D features using ModifiedResNet
                    x_3d = img[i].unsqueeze(0)
                    y_3d = self.encoder_3d(x_3d)
                    node_feat = y_3d.reshape(1, -1)
                    node_feat = self.transform_layer(node_feat)
                    
                    # Encode for temporal contrastive learning
                    encode = self.transform_layer_encode(node_feat).squeeze(0)
                    encode_list.append(encode)
                    encode_dropout.append(torch.dropout(encode, DROPOUT_RATE, self.training))
                    
                    # Transform to node feature dimension
                    node_feat = self.transform_layer_node(node_feat)
                    image_3d.append(F.normalize(node_feat, dim=1))
                    
                    # Extract 2D features (9 views per node)
                    # Note: Paper specifies ResNet-34 for 2D, but code uses ResNet50
                    # ResNet50 is used here for compatibility with existing trained models
                    img2d_feat = F.normalize(self.encoder_2d(img_2d[i]), dim=1)
                    image1_2d.append(img2d_feat)
                    
                    # SCB: Discriminative view selection - select top-2 views
                    scores = self.gate_2d(img2d_feat).sigmoid().squeeze(1)
                    top2_indices = torch.topk(scores, NUM_TOP_VIEWS).indices
                    top2_values = img2d_feat[top2_indices]
                    image1_2d_top2.append(top2_values)
                encode_node_features.append(
                    F.normalize(torch.stack(encode_list), dim=1)
                )
                encode_dropout_features.append(
                    F.normalize(torch.stack(encode_dropout), dim=1)
                )
                list1_3d.append(image_3d)
                list1_2d.append(image1_2d)
                t12_list_2d_top2.append(image1_2d_top2)
            
            t12_list_3d.append(list1_3d)
            t12_list_2d.append(list1_2d)
            orig_encoder_list.append(encode_node_features)
            orig_encoder_dropout.append(encode_dropout_features)
            all_list1_3d.append(list1_3d)
            time_flag = 1
        
        # Process features for contrastive learning and graph matching
        time_flag = 0
        for image, p, n_p, graph, list_3d, list_2d in zip(
            images, points, n_points, graphs, t12_list_3d, t12_list_2d
        ):
            node_features = torch.empty(0, 128)
            for j in range(image.shape[0]):
                for i in range(n_p[j]):
                    node_3d_feat = list_3d[j][i]
                    
                    # Prepare negative samples for contrastive learning (all nodes except current)
                    num_nodes = n_p[j]
                    if i == 0:
                        neg_2d_t1 = list_2d[j][i+1:num_nodes-1]
                        neg_3d_t1 = list_3d[j][i+1:num_nodes-1]
                    elif i == num_nodes - 1:
                        neg_2d_t1 = list_2d[j][:i]
                        neg_3d_t1 = list_3d[j][:i]
                    else:
                        neg_2d_t1 = list_2d[j][:i] + list_2d[j][i+1:num_nodes-1]
                        neg_3d_t1 = list_3d[j][:i] + list_3d[j][i+1:num_nodes-1]
                    
                    # SE: 2D-3D contrastive learning
                    # According to paper: L_se = -log(sum(exp(s+)) / (sum(exp(s+)) + sum(exp(s-))))
                    # where s+ = sim(f_3D, f_2D) / τ for positive pairs, s- for negative pairs
                    logits, labels = self.encoder(
                        list_2d[j][i],  # 9 2D views of current node (positives)
                        node_3d_feat,  # 3D feature of current node (positive)
                        neg_2d_t1,  # 2D views of other nodes in T1 (negatives)
                        t12_list_2d[1-time_flag][j],  # 2D views of nodes in T2 (negatives)
                        neg_3d_t1,  # 3D features of other nodes in T1 (negatives)
                        t12_list_3d[1-time_flag][j],  # 3D features of nodes in T2 (negatives)
                        only_3d=False,
                        only_2d=True
                    )
                    
                    # SE loss: L_se (binary cross-entropy approximation of contrastive loss)
                    if self.training:
                        loss += F.binary_cross_entropy(logits, labels.float(), reduction='mean')
                
                # Aggregate node features for graph processing
                node_feat_batch = torch.stack(list_3d[j]).squeeze(1)
                
                if node_features.numel() == 0:
                    node_features = node_feat_batch
                    time_node_features.append(node_feat_batch)
                else:
                    node_features = node_features.to(labels.device)
                    node_feat_batch = node_feat_batch.to(labels.device)
                    time_node_features.append(node_feat_batch)
                    node_features = torch.cat((node_features, node_feat_batch), dim=0)
            
            # SEAG: Graph neural network processing
            graph.x = node_features.to(image.device)
            graph = self.message_pass_node_features(graph)
            orig_graph = self.build_edge_features_from_node_features(graph)
            orig_graph_list.append(orig_graph)
            time_flag = 1
        # SEAG: Compute similarity matrix using graph features
        unary_affs_list = [
            self.vertex_affinity([item.x for item in g_1], [item.x for item in g_2])
            for (g_1, g_2) in lexico_iter(orig_graph_list)
        ]
        
        # TE: Compute encoder similarity matrix with dropout augmentation
        node_feature_list = [
            self.vertex_affinity_encoder(
                [torch.cat((item, item2), dim=0) for item, item2 in zip(t_1, t_2_dropout)],
                [torch.cat((item, item2), dim=0) for item, item2 in zip(t_1_dropout, t_2)]
            )
            for (t_1, t_2), (t_1_dropout, t_2_dropout) in zip(
                lexico_iter(orig_encoder_list), lexico_iter(orig_encoder_dropout)
            )
        ]
        # TE: Compute encoder similarity matrix
        Kps = torch.stack(pad_tensor(node_feature_list[0]), dim=0)
        node_feature_finallist = torch.sigmoid(Kps)
        
        # SCB: Compute multi-view similarity matrix
        time_result_list = []
        # Get device from first node feature tensor
        device = time_node_features[0].device if time_node_features else images[0].device
        
        for batch_idx in range(batch_size):
            num_nodes_t1 = n_points[0][batch_idx]
            num_nodes_t2 = n_points[1][batch_idx]
            classification_results = torch.zeros(num_nodes_t1, num_nodes_t2, device=device)
            
            for i in range(num_nodes_t1):
                for j in range(num_nodes_t2):
                    # Get 3D and top-2 2D features for node pair
                    feat_t1_3d = time_node_features[batch_idx][i].squeeze()
                    feat_t1_2d = t12_list_2d_top2[batch_idx][i]
                    feat_t1 = torch.cat([feat_t1_3d.unsqueeze(0), feat_t1_2d], dim=0)  # [3, 128]
                    
                    feat_t2_3d = time_node_features[batch_idx + batch_size][j]
                    feat_t2_2d = t12_list_2d_top2[batch_idx + batch_size][j]
                    feat_t2 = torch.cat([feat_t2_3d.unsqueeze(0), feat_t2_2d], dim=0)  # [3, 128]
                    
                    # SCB: Generate 9 feature combinations (3x3: 1 3D + 2 2D views per node)
                    # According to paper: f_{k1,k2}^{(p1,p2)} = Fuse(f_2D^{p1}, f_2D^{p2}, f_3D)
                    pairs = [(a, b) for a in feat_t1 for b in feat_t2]
                    pairs_tensor = torch.stack([torch.cat(pair).unsqueeze(0) for pair in pairs])  # [9, 256]
                    feat_2d_pairs = pairs_tensor[:, :, :self.dim]  # [9, 128] - 2D features
                    feat_3d_pairs = pairs_tensor[:, :, self.dim:]  # [9, 128] - 3D features
                    
                    # SCB: Cross-attention fusion for multi-view feature integration
                    attended_3d, _ = self.cross_attention(query=feat_3d_pairs, key=feat_2d_pairs, value=feat_2d_pairs)
                    attended_2d, _ = self.cross_attention(query=feat_2d_pairs, key=feat_3d_pairs, value=feat_3d_pairs)
                    attn_output = (attended_3d.squeeze(0) + attended_2d.squeeze(0)) / 2
                    fused_pairs = attn_output.squeeze(1)  # [9, 128]
                    
                    # SCB: Similarity scoring with threshold evaluation
                    # Note: Paper specifies: s = σ(Wf + b) · I[σ(Wf + b) ≥ 0.5]
                    # and final similarity = (1/9) * sum(s) across all 9 combinations
                    # Current implementation uses weighted classification instead of direct threshold
                    weights = torch.sigmoid(self.gate(fused_pairs))
                    weights = weights / (weights.sum() + 1e-8)  # Normalize with epsilon for stability
                    logits = self.classifier(fused_pairs)
                    classifications = torch.argmax(torch.sigmoid(logits), dim=1).unsqueeze(dim=1)
                    final_output = (weights * classifications.float()).sum(dim=0)
                    classification_results[i][j] = final_output.item()

            time_result_list.append(classification_results)
        
        time_result_finallist = torch.stack(pad_tensor(time_result_list), dim=0)
        
        # SEAG: Final similarity matrix
        Kp = torch.sigmoid(torch.stack(pad_tensor(unary_affs_list[0]), dim=0))
        Kp_final = torch.sigmoid(torch.stack(pad_tensor(unary_affs_list[0]), dim=0))
        
        # Clip values to valid range for numerical stability
        EPSILON = 1e-6
        node_feature_finallist = torch.clamp(node_feature_finallist, 0.0, 1.0)
        time_result_finallist = torch.clamp(time_result_finallist, 0.0, 1.0)
        Kp = torch.clamp(Kp, 0.0, 1.0)
        Kp_final = torch.clamp(Kp_final, EPSILON, 1.0 - EPSILON)
        
        return loss, node_feature_finallist, Kp, time_result_finallist, Kp_final, np.array(all_list1_3d)
    


class Net(nn.Module):
    """
    Main network wrapper for the asymmetric matching framework.
    
    This class wraps the GMNet and provides a unified interface for training and inference.
    """
    def __init__(self):
        super(Net, self).__init__()
        # Note: encoder parameter is kept for compatibility but not used in GMNet
        self.onlineNet = GMNet(encoder=None) 
        assert cfg.PROBLEM.TYPE == '2GM'  # only support 2GM problem currently

    def forward(self, data_dict, training=False, iter_num=0, epoch=0):
        """
        Forward pass of the network.
        
        Returns:
            data_dict: Updated dictionary containing:
                - perm_mat: SEAG similarity matrix
                - ds_mat: TE encoder similarity matrix
                - mix_mat: SCB similarity matrix
                - loss: SE contrastive loss
        """
        loss, node_feature_list, x_list, time_result_list, Kp_final, oEmbed = self.onlineNet(data_dict)

        if training:
            if cfg.PROBLEM.TYPE == '2GM':
                data_dict.update({
                    'perm_mat': x_list,  # SEAG output
                    'loss': loss,  # SE loss
                    'ds_mat': node_feature_list,  # TE output
                    'mix_mat': time_result_list,  # SCB output
                    'perm_final': Kp_final
                })
        else:
            if cfg.PROBLEM.TYPE == '2GM':
                data_dict.update({
                    'loss': loss,
                    'perm_mat': x_list,
                    'ds_mat': node_feature_list,
                    'mix_mat': time_result_list,
                    'perm_final': Kp_final,
                    'oEmbed': oEmbed,
                })
        return data_dict


# ==================== ResNet Backbone ====================
class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=24, width=128, num_channels=1):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv3d(num_channels, width // 2, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bn1 = nn.BatchNorm3d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(width // 2, width // 2, kernel_size=(3,3,3), padding=(1,1,1), bias=False)
        self.bn2 = nn.BatchNorm3d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv3d(width // 2, width, kernel_size=(3,3,3), padding=(1,1,1), bias=False)
        self.bn3 = nn.BatchNorm3d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool3d(kernel_size=(2,2,2), stride=(2,2,2))

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool3d(input_resolution // 32, embed_dim, heads, output_dim)
    
    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv3d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool3d(
            kernel_size=(1, stride, stride), padding=(0, 1, 1)
        ) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv3d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool3d(stride)),
                ("0", nn.Conv3d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm3d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)
            if out.shape != identity.shape:
                identity = F.interpolate(
                    identity, size=out.shape[2:], mode='trilinear', align_corners=True
                )
        out += identity
        out = self.relu3(out)
        return out


class AttentionPool3d(nn.Module):
    """3D attention pooling for ResNet features."""
    
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim ** 3 + 1, embed_dim) / embed_dim ** 0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.view(B, D * H * W, C)
        
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        
        x, _ = F.multi_head_attention_forward(
            query, key, value,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)
    



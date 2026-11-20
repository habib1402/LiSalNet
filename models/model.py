import torch
import torch.nn as nn
import torch.nn.functional as F

from models.MobileNetV2 import mobilenet_v2


# ============================================================================
# BASIC BUILDING BLOCKS
# ============================================================================

class ConvBNReLU(nn.Module):
    """Basic convolution block with batch normalization and activation"""

    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
                 bias=True, use_relu=True, leaky_relu=False, use_bn=True, prelu=False):
        super(ConvBNReLU, self).__init__()

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=ksize, stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=bias)

        # Batch normalization
        if use_bn:
            self.bn = nn.BatchNorm2d(nOut)
        else:
            self.bn = None

        # Activation function
        if use_relu:
            if leaky_relu:
                self.act = nn.LeakyReLU(0.1, inplace=True)
            elif prelu:
                self.act = nn.PReLU(nOut)
            else:
                self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class EfficientResidualUnit(nn.Module):
    """Efficient inverted residual unit: expansion, depthwise filtering, projection."""

    def __init__(self, inp, oup, stride=1, expand_ratio=2, residual=True):
        super(EfficientResidualUnit, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        if self.stride == 1 and inp == oup:
            self.use_res_connect = residual
        else:
            self.use_res_connect = False

        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.append(ConvBNReLU(inp, hidden_dim, ksize=1, pad=0))

        layers.extend([
            # Depthwise convolution
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # Pointwise projection
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for efficient computation"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                 dilation=dilation, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# ============================================================================
# EFFICIENT ATTENTION AND REFINEMENT MODULES
# ============================================================================

class DualPoolAttentionGate(nn.Module):
    """DPAG: parallel average/max pooling paths with a shared channel MLP."""

    def __init__(self, channels, reduction):
        super(DualPoolAttentionGate, self).__init__()

        # Ensure minimum reduction channels
        reduction_channels = channels // reduction

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP with reduced parameters
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduction_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction_channels, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        return x * channel_att


class SpatialDetailRefiner(nn.Module):
    """Spatial Detail Refiner (SDR) - Compact spatial refinement using efficient multi-scale features"""

    def __init__(self, channels):
        super(SpatialDetailRefiner, self).__init__()

        # Adaptive channel reduction based on input size
        if channels <= 32:
            mid_channels = channels // 2
        else:
            mid_channels = channels // 4

        # Efficient multi-scale processing using factorized convolutions
        self.channel_reduce = nn.Conv2d(channels, mid_channels, 1, bias=False)

        # Factorized spatial filtering with a 3x3 sampling extent
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, (1, 3), padding=(0, 1), groups=mid_channels, bias=False),
            nn.Conv2d(mid_channels, mid_channels, (3, 1), padding=(1, 0), groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Channel restoration
        self.channel_restore = nn.Sequential(
            nn.Conv2d(mid_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        # Channel reduction
        feat = self.channel_reduce(x)

        # Spatial processing
        spatial_feat = self.spatial_conv(feat)

        # Restore channels
        refined = self.channel_restore(spatial_feat)
        return F.relu(refined + x)  # Residual connection


class ParallelFocusUnit(nn.Module):
    """PFU: DPAG followed by SDR; the two pooling paths within DPAG are parallel."""

    def __init__(self, channels, reduction):
        super(ParallelFocusUnit, self).__init__()

        # Channel attention for feature recalibration
        self.channel_attention = DualPoolAttentionGate(channels, reduction)

        # Spatial refinement for enhanced spatial details
        self.spatial_refinement = SpatialDetailRefiner(channels)

        # Pointwise final refinement
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

        # Adaptive dropout based on channel size
        dropout_rate = 0.05 if channels <= 32 else 0.1
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        # Store identity for residual connection
        identity = x

        # Step 1: Channel attention refinement
        x = self.channel_attention(x)

        # Step 2: Spatial feature refinement
        x = self.spatial_refinement(x)

        # Step 3: Final refinement
        x = self.final_conv(x)
        x = self.dropout(x)

        # Residual connection with ReLU activation
        return F.relu(x + identity)


# ============================================================================
# MULTI-SCALE AND DILATED CONVOLUTION MODULES
# ============================================================================

class ContextFieldEnhancer(nn.Module):
    """CFE: preactivation, channel reduction, dilated depthwise/pointwise filtering."""

    def __init__(self, in_channels, out_channels, dilation_rate, dropout=0.1):
        super(ContextFieldEnhancer, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3,
                                 dilation=dilation_rate, padding=dilation_rate)
        )
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None

    def forward(self, x):
        out = self.conv_block(x)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class DilatedPerceptionModule(nn.Module):
    """Dilated Perception Module (DPM) - Compact multi-scale feature extraction module"""

    def __init__(self, channel, reduction_ratio=4):
        super(DilatedPerceptionModule, self).__init__()
        out_channel = channel // reduction_ratio

        # Four branches: 32->8 each in the Section 3 configuration.
        self.dilated_conv_6 = ContextFieldEnhancer(channel, out_channel, 6)
        self.dilated_conv_12 = ContextFieldEnhancer(channel, out_channel, 12)
        self.dilated_conv_18 = ContextFieldEnhancer(channel, out_channel, 18)
        self.conv_1x1 = nn.Conv2d(channel, out_channel, 1)

        self.final_conv = nn.Conv2d(channel, channel, 1)

    def forward(self, x):
        feat6 = self.dilated_conv_6(x)
        feat12 = self.dilated_conv_12(x)
        feat18 = self.dilated_conv_18(x)
        feat1x1 = self.conv_1x1(x)

        out = torch.cat([feat1x1, feat6, feat12, feat18], dim=1)
        return self.final_conv(out)


# ============================================================================
# FUSION MODULES
# ============================================================================

class SymmetricFusionBlock(nn.Module):
    """SFB: one fused output with residual contributions from both modalities."""

    def __init__(self, inc, expand_ratio=2):
        super(SymmetricFusionBlock, self).__init__()
        expanded_ch = inc * expand_ratio

        # Symmetric factorization
        self.horizontal_conv = nn.Conv2d(expanded_ch, expanded_ch,
                                       (1, 3), padding=(0, 1),
                                       groups=expanded_ch, bias=False)
        self.vertical_conv = nn.Conv2d(expanded_ch, expanded_ch,
                                     (3, 1), padding=(1, 0),
                                     groups=expanded_ch, bias=False)

        # Channel reduction after spatial processing
        self.channel_reduce = nn.Conv2d(expanded_ch, inc, 1, bias=False)
        self.norm = nn.GroupNorm(8, inc)

    def forward(self, conv, depth):
        # Simple concatenation for efficiency
        fused = torch.cat([conv, depth], dim=1)
        out = self.horizontal_conv(fused)
        out = self.vertical_conv(out)
        out = self.channel_reduce(out)
        out = self.norm(out)

        # Residual with both inputs
        return F.gelu(out + conv + depth)


class ModalityAlignmentNetwork(nn.Module):
    """MAN: depth-conditioned RGB feature gating, without geometric warping."""

    def __init__(self, inchannels=320):
        super(ModalityAlignmentNetwork, self).__init__()

        # Expansion ratio 2 at both selected feature levels
        self.conv1 = EfficientResidualUnit(inchannels, inchannels,
                                       expand_ratio=2, residual=True)

        # Adaptive global context based on channel size
        reduction_ratio = max(4, inchannels // 32)
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inchannels, inchannels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannels // reduction_ratio, inchannels, 1),
            nn.Sigmoid()
        )

        # Pointwise fusion followed by BN and ReLU
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(inchannels * 2, inchannels, 1, bias=False),
            nn.BatchNorm2d(inchannels),
            nn.ReLU(inplace=True)
        )

        # Cross-modal attention
        self.cross_attention = nn.Sequential(
            nn.Conv2d(inchannels * 2, inchannels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannels // 4, inchannels, 1),
            nn.Sigmoid()
        )

    def forward(self, conv, depth):
        # Store original inputs for residual connection
        conv_orig = conv

        # Initial feature enhancement
        f_enhanced = self.conv1(conv * depth)

        # Global context attention
        global_att = self.global_context(conv)
        f_context = conv * global_att

        # Cross-modal attention
        cross_input = torch.cat([f_context, depth], dim=1)
        cross_att = self.cross_attention(cross_input)
        f_cross = f_context * cross_att

        # Final fusion
        fused_input = torch.cat([f_enhanced, f_cross], dim=1)
        output = self.fusion_conv(fused_input)

        # Residual connection with original RGB features
        return output + conv_orig



class Fusion(nn.Module):
    """Simple fusion module for combining two features"""

    def __init__(self, channels_per_input, out_channels):
        super(Fusion, self).__init__()

        # Multi-input fusion - calculate total input channels
        total_in_channels = channels_per_input * 2
        self.fusion = nn.Sequential(
            nn.Conv2d(total_in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, input1, input2):
        # Ensure both inputs have the same spatial size
        target_size = input1.shape[2:]
        if input2.shape[2:] != target_size:
            input2 = F.interpolate(input2, size=target_size, mode='bilinear', align_corners=False)

        fused = torch.cat([input1, input2], dim=1)
        return self.fusion(fused)

# ============================================================================
# DEPTH NETWORK
# ============================================================================

class DepthFeatureExtractor (nn.Module):
    """Depth feature extraction network"""

    def __init__(self):
        super(DepthFeatureExtractor, self).__init__()
        block = EfficientResidualUnit
        input_channel = 1

        # Network configuration: [expansion_ratio, channels, num_blocks, stride, dilation]
        efficient_residual_setting = [
            [1, 16, 2, 2, 1],
            [4, 24, 2, 2, 1],
            [4, 32, 2, 2, 1],
            [4, 96, 2, 2, 1],
            [4, 320, 2, 2, 1],
        ]

        features = []
        # Build efficient residual blocks
        for t, c, n, s, _ in efficient_residual_setting:
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, c, stride, expand_ratio=t))
                input_channel = c

        self.features = nn.Sequential(*features)

    def forward(self, x):
        feats = []
        for i, block in enumerate(self.features):
            x = block(x)
            # Extract features at specific layers
            if i in [1, 3, 5, 7, 9]:
                feats.append(x)
        return feats


# ============================================================================
# DECODER MODULES
# ============================================================================

class SequentialFeatureDecoder(nn.Module):
    """Sequential Feature Decoder (SFD) - Cross-Scale Progressive Refinement Decoder"""

    def __init__(self, in_channels, out_channels):
        super(SequentialFeatureDecoder, self).__init__()

        # Validate channel configurations
        assert len(in_channels) == len(out_channels), "Channel lists must have same length"

        self.num_stages = len(in_channels)

        # Lateral connections (process current level features)
        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_stages):
            self.lateral_convs.append(
                ConvBNReLU(in_channels[i], out_channels[i], ksize=1, pad=0)
            )

        # Top-down connections (process higher level features for fusion)
        self.top_down_convs = nn.ModuleList()
        for i in range(self.num_stages - 1):  # No top-down for the highest level
            # More gradual channel reduction to avoid bottlenecks
            reduction_channels = min(out_channels[i+1], out_channels[i])
            self.top_down_convs.append(
                ConvBNReLU(out_channels[i+1], reduction_channels, ksize=1, pad=0)
            )

        # Fusion modules
        self.fusion_modules = nn.ModuleList()
        for i in range(self.num_stages):
            if i == self.num_stages - 1:
                # Highest level - no fusion needed
                self.fusion_modules.append(nn.Identity())
            else:
                self.fusion_modules.append(
                    Fusion(out_channels[i], out_channels[i])
                )

    def forward(self, features):
        """
        Args:
            features: List of feature maps from encoder [high_res, mid_res, low_res]
                     e.g., [fused1, fused3, fused5] with shapes:
                     [[B,16,160,160], [B,32,40,40], [B,320,10,10]] at 320 input
        Returns:
            List of refined feature maps at multiple scales
        """
        assert len(features) == self.num_stages, f"Expected {self.num_stages} features, got {len(features)}"

        results = []

        # Start from the highest level (smallest spatial size, most semantic)
        # Process lateral connection for highest level
        current_feat = self.lateral_convs[-1](features[-1])
        results.append(current_feat)

        # Progressive refinement from high level to low level
        for idx in range(self.num_stages - 2, -1, -1):
            # Process lateral connection for current level
            lateral_feat = self.lateral_convs[idx](features[idx])

            # Process top-down connection
            top_down_feat = self.top_down_convs[idx](current_feat)

            # Upsample top-down features to match current level size
            if top_down_feat.shape[2:] != lateral_feat.shape[2:]:
                top_down_feat = F.interpolate(
                    top_down_feat,
                    size=lateral_feat.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

            current_feat = self.fusion_modules[idx](lateral_feat, top_down_feat)

            results.insert(0, current_feat)

        return results


# ============================================================================
# MAIN MODEL
# ============================================================================

class LiSalNet(nn.Module):
    """Efficient RGB-D SOD model implementing revised Section 3.

    Default configuration: 3,984,607 parameters after removing RGB block 18.
    forward() returns probabilities [P1, P3, P5]; inference selects [:, :1].
    Use pretrained=False when loading a trained LiSalNet checkpoint.
    """

    def __init__(self, pretrained=True,
                enc_channels=(16, 32, 320),
                dec_channels=(16, 32, 320)):
        super(LiSalNet, self).__init__()

        # Backbone networks
        self.Backbone = mobilenet_v2(pretrained)
        self.DFE = DepthFeatureExtractor()

        # One shared DPM: two forward calls, shared weights and BN running state.
        self.DPM_3 = DilatedPerceptionModule(channel=32)

        # Global feature matching
        self.SFB_3 = SymmetricFusionBlock(inc=32, expand_ratio=2)

        # Depth fusion modules
        self.MAN_1 = ModalityAlignmentNetwork(inchannels=16)
        self.MAN_5 = ModalityAlignmentNetwork(inchannels=320)

        # Enhanced spatial-channel refinement
        self.PFU1 = ParallelFocusUnit(16, reduction=2)
        self.PFU3  = ParallelFocusUnit(32, reduction=4)
        self.PFU5 = ParallelFocusUnit(320, reduction=16)

        # Feature sequential network decoder
        self.SFD = SequentialFeatureDecoder(enc_channels, dec_channels)

        # Classification heads
        self.cls1 = nn.Conv2d(dec_channels[0], 1, 1, stride=1, padding=0)
        self.cls2 = nn.Conv2d(dec_channels[1], 1, 1, stride=1, padding=0)
        self.cls3 = nn.Conv2d(dec_channels[2], 1, 1, stride=1, padding=0)

    def forward(self, input, depth):
        """
        Forward pass of LiSalNet, with logit resizing before sigmoid.

        Args:
            input: RGB input tensor [B, 3, H, W]
            depth: Depth input tensor [B, 1, H, W]

        Returns:
            saliency_maps: Tensor of saliency predictions [B, 3, H, W]
        """

        # Generate backbone features
        conv1, conv2, conv3, conv4, conv5 = self.Backbone(input)

        # Generate depth features
        depth1, depth2, depth3, depth4, depth5 = self.DFE(depth)

        # Multi-scale processing for middle-level features
        conv3 = self.DPM_3(conv3)
        depth3 = self.DPM_3(depth3)

        # Global feature matching for level 3
        fused3 = self.SFB_3(conv3, depth3)

        # Depth fusion for levels 1 and 5
        fused1 = self.MAN_1(conv1, depth1)
        fused5 = self.MAN_5(conv5, depth5)

        # Enhanced spatial-channel refinement
        fused1 = self.PFU1(fused1)
        fused3 = self.PFU3(fused3)
        fused5 = self.PFU5(fused5)

        # Feature pyramid network processing
        features = self.SFD([fused1, fused3, fused5])

        # Generate saliency maps at multiple scales
        saliency_maps = []
        for idx, feature in enumerate(features[:3]):
            saliency_maps.append(F.interpolate(
                    getattr(self, 'cls' + str(idx + 1))(feature),
                    input.shape[2:],
                    mode='bilinear',
                    align_corners=False)
            )

        # Combine and apply sigmoid activation
        saliency_maps = torch.sigmoid(torch.cat(saliency_maps, dim=1))

        return saliency_maps

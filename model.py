import torch
import torch.nn as nn
import torch.nn.functional as F

# Auxiliary classifier for intermediate features
class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super(AuxiliaryClassifier, self).__init__()
        self.aux_conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1)
        self.aux_bn1 = nn.BatchNorm2d(256)
        self.aux_relu1 = nn.ReLU(inplace=True)
        self.aux_conv2 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        # Permute to [B, C, H, W] for Conv2d
        x = x.permute(0, 3, 1, 2)
        x = self.aux_conv1(x)
        x = self.aux_bn1(x)
        x = self.aux_relu1(x)
        x = self.aux_conv2(x)
        # Upsample to target size
        x = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)
        return x

# Segmentation head with Multi-Level Aggregation (MLA)
class SegmentationHead(nn.Module):
    def __init__(self, in_channels, intermediate_channels, out_channels=1, align_corners=False):
        super(SegmentationHead, self).__init__()
        self.align_corners = align_corners

        # Define MLA branches for intermediate features
        self.mla_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ) for _ in range(4)
        ])

        # Image embedding branch
        self.mla_image_branch = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Classifier branch for aggregated features
        self.mla_classifier_branch = nn.Sequential(
            nn.Conv2d(256 * 5, intermediate_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_channels, out_channels, kernel_size=1, stride=1)
        )

    def forward(self, image_embedding, inter_features):
        if inter_features is None:
            raise ValueError("inter_features must be provided for MLA strategy")
        if len(inter_features) < 24:
            raise ValueError(f"Expected at least 24 inter_features for MLA strategy, but got {len(inter_features)}")

        # Select specific intermediate features
        selected_features = [inter_features[i] for i in [5, 11, 17, 23]]
        selected_features = [feat.permute(0, 3, 1, 2) for feat in selected_features]

        processed_features = []
        # Process each selected feature through its branch
        for i, feat in enumerate(selected_features):
            branch = self.mla_branches[i]
            x_feat = branch(feat)
            x_feat = F.interpolate(x_feat, scale_factor=4, mode='bilinear', align_corners=self.align_corners)
            processed_features.append(x_feat)

        # Process image embedding
        img_feat = self.mla_image_branch(image_embedding)
        img_feat = F.interpolate(img_feat, scale_factor=4, mode='bilinear', align_corners=self.align_corners)
        processed_features.append(img_feat)

        # Aggregate features and classify
        aggregated = torch.cat(processed_features, dim=1)
        x = self.mla_classifier_branch(aggregated)
        x = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=self.align_corners)

        return x
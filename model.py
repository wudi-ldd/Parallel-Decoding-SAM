import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_gn(c, prefer_groups=(32, 16, 8, 4, 2, 1)):
    """Select a GroupNorm divisor that evenly partitions the channel dimension."""
    for g in prefer_groups:
        if c % g == 0:
            return nn.GroupNorm(g, c)
    return nn.GroupNorm(1, c)


class DSConv(nn.Module):
    """Depthwise separable convolution block with GroupNorm and ReLU activations."""

    def __init__(self, ch, out_ch, dilation=1):
        super().__init__()
        pad = dilation
        self.dw = nn.Conv2d(ch, ch, 3, padding=pad, dilation=dilation, groups=ch, bias=False)
        self.gn = _make_gn(ch)
        self.act = nn.ReLU(inplace=True)
        self.pw = nn.Conv2d(ch, out_ch, 1, bias=False)

    def forward(self, x):
        """Apply depthwise followed by pointwise convolution with non-linearities."""
        x = self.dw(x)
        x = self.gn(x)
        x = self.act(x)
        x = self.pw(x)
        x = self.act(x)
        return x


class SegmentationHead(nn.Module):
    """Three-branch atrous decoder that upsamples SAM embeddings to mask logits.

    Args:
        in_channels (int): Number of SAM embedding channels (default 256).
        intermediate_channels (int): Channel width per atrous branch.
        out_channels (int): Number of output mask channels (default 1 for binary tasks).
        align_corners (bool): Whether to align corners during bilinear upsampling.

    Shapes:
        Input: ``(B, in_channels, 64, 64)``
        Output: ``(B, out_channels, 1024, 1024)``
    """

    def __init__(self, in_channels=256, intermediate_channels=64, out_channels=1, align_corners=False):
        super().__init__()
        self.align_corners = align_corners
        # Three dilated branches keep the head lightweight
        self.b1 = DSConv(in_channels, intermediate_channels, dilation=1)
        self.b2 = DSConv(in_channels, intermediate_channels, dilation=2)
        self.b3 = DSConv(in_channels, intermediate_channels, dilation=3)
        self.merge = nn.Conv2d(intermediate_channels * 3, out_channels, 1, bias=True)
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, image_embedding):
        """Aggregate multi-dilation features and upsample to 1024×1024 logits."""
        x = self.dropout(image_embedding)
        f1 = self.b1(x)
        f2 = self.b2(x)
        f3 = self.b3(x)
        x = torch.cat([f1, f2, f3], dim=1)
        x = self.merge(x)
        x = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=self.align_corners)
        return x

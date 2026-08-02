import torch
import torch.nn as nn


class BasicCNN(nn.Module):
    """
    A lightweight, basic CNN model for 2D coordinate regression [touch_x, touch_y].
    Inputs are expected to be batch of RGB images of shape (Batch, 3, Height, Width).
    """

    def __init__(self, num_outputs=2):
        super(BasicCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 3 x 240 x 320 -> 32 x 240 x 320 -> 32 x 120 x 160
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2: 32 x 120 x 160 -> 64 x 120 x 160 -> 64 x 60 x 80
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3: 64 x 60 x 80 -> 128 x 60 x 80 -> 128 x 30 x 40
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4: 128 x 30 x 40 -> 256 x 30 x 40 -> 256 x 15 x 20
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Replaces the massive Dense layer with a 1x1 conv to create a single-channel spatial heatmap
        self.heatmap_conv = nn.Conv2d(256, 1, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.out_channels == 1 and m.kernel_size == (1, 1):
                    # Heatmap conv: initialize with near-zero variance to start with a flat, uniform softmax distribution
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        heatmap = self.heatmap_conv(x)  # Shape: (Batch, 1, H, W)

        B, C, H, W = heatmap.shape
        heatmap_flat = heatmap.view(B, C, -1)

        # Apply spatial softmax across all pixels in the HxW grid
        attention = torch.nn.functional.softmax(heatmap_flat, dim=-1)
        attention = attention.view(B, C, H, W)

        # Create normalized coordinate grid [-1, 1] mapped to the device
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij",
        )

        # Compute Expected X and Expected Y by doing a weighted sum of the coordinate grid
        expected_x = torch.sum(attention * grid_x, dim=(2, 3))
        expected_y = torch.sum(attention * grid_y, dim=(2, 3))

        # Concatenate expected coordinates to shape (Batch, 2)
        out = torch.cat([expected_x, expected_y], dim=1)
        return out

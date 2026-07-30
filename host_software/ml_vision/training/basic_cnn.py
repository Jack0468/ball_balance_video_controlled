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
        
        # Adaptive average pooling maps any input spatial dimensions to a fixed (4, 4) grid
        self.pool = nn.AdaptiveAvgPool2d((4, 4)) # Output shape: Batch x 256 x 4 x 4 (flattened to 4096)
        
        # Regression head with dropout to prevent overfitting
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 4 * 4, 256),
            nn.GELU(),
            nn.Linear(256, num_outputs)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_p=0.2):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
        
    def forward(self, x):
        return x + self.block(x)

class CorrectorMLP(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=128, output_dim=2, num_blocks=2, dropout_p=0.4):
        super(CorrectorMLP, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_p) for _ in range(num_blocks)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        return self.output_layer(x)

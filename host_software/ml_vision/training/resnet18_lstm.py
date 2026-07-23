import torch
import torch.nn as nn
import torchvision.models as models

class TemporalExpertTracker(nn.Module):
    """
    Temporal Expert Tracker
    Combines a ResNet18 spatial feature extractor with an LSTM temporal sequence model.
    """
    def __init__(self, hidden_size=256, num_layers=1):
        super(TemporalExpertTracker, self).__init__()
        
        # 1. Spatial Feature Extractor (ResNet18)
        # Load a pretrained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Remove the final fully connected layer to output raw 512D feature vectors
        # resnet.fc is originally Linear(in_features=512, out_features=1000)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512 # ResNet18 outputs 512 channels before the FC layer
        
        # 2. Temporal Sequence Model (LSTM)
        # Takes the sequence of 512D features and learns velocity/momentum
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True # Input shape: (Batch, SeqLen, Features)
        )
        
        # 3. Regression Head
        # Maps the LSTM's final hidden state to the 2D coordinate [touch_x, touch_y]
        self.fc = nn.Linear(hidden_size, 2)
        
    def forward(self, x):
        # x shape: (Batch, SeqLen, C, H, W)
        batch_size, seq_len, C, H, W = x.size()
        
        # Reshape to (Batch * SeqLen, C, H, W) to process all frames through CNN efficiently
        x_flat = x.view(batch_size * seq_len, C, H, W)
        
        # Extract spatial features
        # Output shape: (Batch * SeqLen, 512, 1, 1)
        features = self.feature_extractor(x_flat)
        
        # Flatten the spatial dimensions (1, 1)
        # Output shape: (Batch * SeqLen, 512)
        features = features.view(batch_size * seq_len, -1)
        
        # Reshape back to sequence format for LSTM
        # Output shape: (Batch, SeqLen, 512)
        features_seq = features.view(batch_size, seq_len, self.feature_dim)
        
        # Pass through LSTM
        # lstm_out shape: (Batch, SeqLen, hidden_size)
        # hn shape: (num_layers, Batch, hidden_size)
        lstm_out, (hn, cn) = self.lstm(features_seq)
        
        # We only care about the final prediction after seeing the entire sequence
        # We take the output of the LSTM at the last timestep
        last_timestep_out = lstm_out[:, -1, :] # Shape: (Batch, hidden_size)
        
        # Predict final coordinate
        out = self.fc(last_timestep_out) # Shape: (Batch, 2)
        
        return out

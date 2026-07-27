import torch
import torch.nn as nn
import torchvision.models as models

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    Conditions vision features based on language embeddings.
    """
    def __init__(self, language_dim, vision_dim):
        super(FiLMLayer, self).__init__()
        self.gamma = nn.Linear(language_dim, vision_dim)
        self.beta = nn.Linear(language_dim, vision_dim)
        
    def forward(self, vision_features, language_embed):
        # vision_features: [B, C, H, W]
        # language_embed: [B, language_dim]
        gamma = self.gamma(language_embed).unsqueeze(-1).unsqueeze(-1) # [B, C, 1, 1]
        beta = self.beta(language_embed).unsqueeze(-1).unsqueeze(-1)   # [B, C, 1, 1]
        
        return gamma * vision_features + beta

class RT1LiteVLA(nn.Module):
    """
    RT-1-Lite Architecture for Vision-Language-Action (VLA) modelling.
    Replaces the naive MLP concatenation with FiLM conditioning and a Transformer backbone,
    serving as a robust research baseline against models like RT-1 and Octo-small.
    """
    def __init__(self, num_commands=5, state_dim=2, action_dim=3, language_dim=64):
        super(RT1LiteVLA, self).__init__()
        
        # 1. Language/Audio Tokenizer
        self.audio_embed = nn.Embedding(num_commands, language_dim)
        
        # 2. Vision Backbone (ResNet18 without the final pooling/fc layers)
        resnet = models.resnet18(pretrained=True)
        self.vision_stem = nn.Sequential(*list(resnet.children())[:-2]) # Outputs [B, 512, 7, 7]
        
        # Freeze early vision layers for stability on small datasets
        for param in self.vision_stem.parameters():
            param.requires_grad = False
            
        # 3. Vision-Language Fusion (FiLM)
        self.film = FiLMLayer(language_dim=language_dim, vision_dim=512)
        
        # 4. State Embedder
        self.state_embed = nn.Linear(state_dim, 512)
        
        # 5. Transformer Action Decoder
        # Flattens the 7x7 spatial patches into a sequence of 49 tokens
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 6. Action Head
        self.action_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, img, cmd_idx, state):
        # Embed Language
        lang_emb = self.audio_embed(cmd_idx) # [B, language_dim]
        
        # Extract Spatial Vision Features
        v_feat = self.vision_stem(img) # [B, 512, 7, 7]
        
        # FiLM Conditioning (Fuse Language into Vision at spatial level)
        fused_v_feat = self.film(v_feat, lang_emb) # [B, 512, 7, 7]
        
        # Flatten spatial dimensions to sequence: [B, 512, 49] -> [B, 49, 512]
        B, C, H, W = fused_v_feat.shape
        v_seq = fused_v_feat.view(B, C, H * W).permute(0, 2, 1)
        
        # Append State Token to the sequence: [B, 50, 512]
        s_emb = self.state_embed(state).unsqueeze(1) # [B, 1, 512]
        seq = torch.cat([s_emb, v_seq], dim=1)
        
        # Transformer Processing
        trans_out = self.transformer(seq) # [B, 50, 512]
        
        # Pool the sequence (use the state token's output representation for action)
        pooled = trans_out[:, 0, :] # [B, 512]
        
        # Decode to motor actions
        action_pred = self.action_head(pooled) # [B, action_dim]
        
        return action_pred

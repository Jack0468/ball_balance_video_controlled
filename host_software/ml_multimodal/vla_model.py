import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class VLADataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.transform = transform
        
        # Simple vocabulary for audio commands
        self.vocab = {"hold": 0, "go red": 1, "go blue": 2, "go green": 3, "go yellow": 4}
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image (Vision Token)
        img_path = item['image_path']
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            # Fallback for missing images in a mock dataset
            img = Image.new('RGB', (224, 224))
            
        if self.transform:
            img = self.transform(img)
            
        # Audio Token (Word index)
        cmd_idx = self.vocab.get(item['audio_command'], 0)
        
        # State Token
        state = torch.tensor([item['state_x'], item['state_y']], dtype=torch.float32)
        
        # Action Targets (Expert PID output to imitate)
        action = torch.tensor([item['action_theta_a'], item['action_theta_b'], item['action_theta_c']], dtype=torch.float32)
        
        return img, cmd_idx, state, action


class LightweightVLA(nn.Module):
    def __init__(self, num_commands=5, state_dim=2, action_dim=3):
        super(LightweightVLA, self).__init__()
        
        # Vision Backbone (Frozen ResNet18 for fast lightweight extraction)
        self.vision_backbone = models.resnet18(pretrained=True)
        for param in self.vision_backbone.parameters():
            param.requires_grad = False
            
        # Replace fully connected layer to output a 128-dim vision embedding
        num_ftrs = self.vision_backbone.fc.in_features
        self.vision_backbone.fc = nn.Linear(num_ftrs, 128)
        
        # Audio/Language embedding
        self.audio_embed = nn.Embedding(num_commands, 32)
        
        # State embedding
        self.state_embed = nn.Linear(state_dim, 32)
        
        # Fusion MLP (Action Decoding)
        # Vision (128) + Audio (32) + State (32) = 192
        self.fusion = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, img, cmd_idx, state):
        v_emb = self.vision_backbone(img)      # [B, 128]
        a_emb = self.audio_embed(cmd_idx)      # [B, 32]
        s_emb = self.state_embed(state)        # [B, 32]
        
        fused = torch.cat((v_emb, a_emb, s_emb), dim=1) # [B, 192]
        action_pred = self.fusion(fused)       # [B, action_dim]
        
        return action_pred

def train_vla():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "data/03_gold/vla_dataset.json")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Loading VLA Dataset...")
    dataset = VLADataset(dataset_path, transform=transform)
    
    if len(dataset) == 0:
        print("Empty dataset.")
        return
        
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightVLA().to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Starting Behavioral Cloning on {device}...")
    
    for epoch in range(1, 6): # Short training loop for testing
        model.train()
        train_loss = 0
        for imgs, cmds, states, actions in train_loader:
            imgs, cmds, states, actions = imgs.to(device), cmds.to(device), states.to(device), actions.to(device)
            
            optimizer.zero_grad()
            preds = model(imgs, cmds, states)
            loss = criterion(preds, actions)
            loss.backward()
            optimizer.save_step = optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Eval
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, cmds, states, actions in test_loader:
                imgs, cmds, states, actions = imgs.to(device), cmds.to(device), states.to(device), actions.to(device)
                preds = model(imgs, cmds, states)
                loss = criterion(preds, actions)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(test_loader.dataset)
        
        print(f"Epoch {epoch}/5 | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

if __name__ == "__main__":
    train_vla()

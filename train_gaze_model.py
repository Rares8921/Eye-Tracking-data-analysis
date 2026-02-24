import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json


class GazeDataset(Dataset):
    def __init__(self, recordings_dir='recordings', features_dir='features',
                 attention_dir='attention_maps', saliency_dir='saliency_metrics',
                 image_size=(224, 224)):
        
        self.image_size = image_size
        self.samples = []
        
        recordings = list(Path(recordings_dir).glob('*'))
        
        for rec_path in recordings:
            rec_id = rec_path.name
            
            gaze_file = rec_path / 'gaze_positions_on_surface_Surface 1.csv'
            if not gaze_file.exists():
                continue
            
            features_file = Path(features_dir) / f'features_{rec_id}.csv'
            if not features_file.exists():
                continue
            
            attention_kde = Path(attention_dir) / f'{rec_id}_map_kde.npy'
            if not attention_kde.exists():
                continue
            
            saliency_map = Path(saliency_dir) / f'{rec_id}_saliency_map.npy'
            if not saliency_map.exists():
                continue
            
            self.samples.append({
                'recording_id': rec_id,
                'gaze_file': str(gaze_file),
                'features_file': str(features_file),
                'attention_map': str(attention_kde),
                'saliency_map': str(saliency_map)
            })
        
        print(f"Loaded {len(self.samples)} training samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        attention_map = np.load(sample['attention_map']).astype(np.float32)
        if attention_map.shape != self.image_size:
            attention_map = cv2.resize(attention_map, self.image_size)
        if attention_map.max() > 0:
            attention_map = attention_map / attention_map.max()
        
        saliency_map = np.load(sample['saliency_map']).astype(np.float32)
        if saliency_map.shape != self.image_size:
            saliency_map = cv2.resize(saliency_map, self.image_size)
        if saliency_map.max() > 0:
            saliency_map = saliency_map / saliency_map.max()
        
        features_df = pd.read_csv(sample['features_file'])
        features = np.array([
            features_df['avg_fixation_duration_s'].values[0],
            features_df['saccade_length_avg'].values[0],
            features_df['gaze_entropy'].values[0],
            features_df['fixation_dispersion'].values[0],
            features_df['waldo_fixation_ratio'].values[0]
        ], dtype=np.float32)
        
        gaze_df = pd.read_csv(sample['gaze_file'])
        if 'gaze detected on surface' in gaze_df.columns:
            gaze_df = gaze_df[gaze_df['gaze detected on surface'] == True]
        
        gaze_df = gaze_df.sort_values('timestamp [ns]')
        xs = gaze_df['gaze position on surface x [normalized]'].values[:500]
        ys = gaze_df['gaze position on surface y [normalized]'].values[:500]
        
        if len(xs) < 500:
            xs = np.pad(xs, (0, 500 - len(xs)), 'edge')
            ys = np.pad(ys, (0, 500 - len(ys)), 'edge')
        
        gaze_sequence = np.stack([xs, ys], axis=1).astype(np.float32)
        
        return {
            'attention_map': torch.from_numpy(attention_map).unsqueeze(0),
            'saliency_map': torch.from_numpy(saliency_map).unsqueeze(0),
            'features': torch.from_numpy(features),
            'gaze_sequence': torch.from_numpy(gaze_sequence)
        }


class GazeGenerationModel(nn.Module):
    def __init__(self, feature_dim=5, hidden_dim=256, num_points=500):
        super().__init__()
        
        self.num_points = num_points
        
        self.attention_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.saliency_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.feature_fc = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128 + 128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            self._create_positional_encoding(num_points, 64), 
            requires_grad=True
        )
        
        self.lstm = nn.LSTM(hidden_dim + 64, hidden_dim // 2, num_layers=2, 
                           batch_first=True, bidirectional=True, dropout=0.3)
        
        self.output_layer = nn.Linear(hidden_dim, 2)
    
    def _create_positional_encoding(self, seq_len, d_model):
        position = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
        
    def forward(self, attention_map, saliency_map, features):
        batch_size = attention_map.size(0)
        
        att_feat = self.attention_encoder(attention_map).squeeze(-1).squeeze(-1)
        sal_feat = self.saliency_encoder(saliency_map).squeeze(-1).squeeze(-1)
        feat_emb = self.feature_fc(features)
        
        fused = torch.cat([att_feat, sal_feat, feat_emb], dim=1)
        context = self.fusion(fused)
        
        context_expanded = context.unsqueeze(1).expand(-1, self.num_points, -1)
        
        # Add positional encoding
        pos_enc = self.positional_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        lstm_input = torch.cat([context_expanded, pos_enc], dim=2)
        
        lstm_out, _ = self.lstm(lstm_input)
        
        gaze_points = self.output_layer(lstm_out)
        gaze_points = torch.sigmoid(gaze_points)
        
        return gaze_points


def train_model(num_epochs=50, batch_size=4, lr=0.001, device='cuda'):
    dataset = GazeDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    model = GazeGenerationModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            attention = batch['attention_map'].to(device)
            saliency = batch['saliency_map'].to(device)
            features = batch['features'].to(device)
            target_gaze = batch['gaze_sequence'].to(device)
            
            pred_gaze = model(attention, saliency, features)
            
            loss = criterion(pred_gaze, target_gaze)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'gaze_model_best.pth')
            print(f"  Saved best model (loss: {best_loss:.4f})")
    
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, 'gaze_model_final.pth')
    
    print("Training complete!")
    return model


def generate_gaze_for_image(model, image_path, output_path='predicted_gaze.npy', device='cuda'):
    model.eval()
    
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None
    
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
    
    default_features = torch.tensor([[0.3, 0.15, 7.0, 0.2, 0.05]], device=device)
    
    with torch.no_grad():
        pred_gaze = model(img_tensor, img_tensor, default_features)
    
    gaze_points = pred_gaze.cpu().numpy()[0]
    
    np.save(output_path, gaze_points)
    print(f"Generated gaze sequence saved to {output_path}")
    
    return gaze_points


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train gaze generation model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    model = train_model(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )

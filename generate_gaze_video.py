import numpy as np
import cv2
from pathlib import Path
import pandas as pd
import argparse
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import torch
import torch.nn as nn


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


class GazeVideoGenerator:
    def __init__(self, fps=30, model_path=None, device='cuda'):
        self.fps = fps
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
            print(f"GazeVideoGenerator initialized (FPS: {fps}, Model: loaded, Device: {self.device})")
        else:
            print(f"GazeVideoGenerator initialized (FPS: {fps}, Model: None)")
    
    def load_model(self, model_path):
        self.model = GazeGenerationModel().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"  Model loaded from {model_path}")
    
    # Savitzky-Golay filter
    #https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
    def smooth_gaze_path(self, xs, ys, window_length=15, polyorder=3):
        if len(xs) < window_length:
            return xs, ys
        
        if window_length % 2 == 0:
            window_length += 1
        
        xs_smooth = savgol_filter(xs, window_length, polyorder)
        ys_smooth = savgol_filter(ys, window_length, polyorder)
        
        return xs_smooth, ys_smooth
    
    def compute_saliency(self, img):
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        try:
            sal = cv2.saliency.StaticSaliencySpectralResidual_create()
            _, salmap = sal.computeSaliency(img)
            salmap = salmap.astype(np.float32)
        except:
            grayf = gray.astype(np.float32)
            dft = cv2.dft(grayf, flags=cv2.DFT_COMPLEX_OUTPUT)
            mag, ang = cv2.cartToPolar(dft[:, :, 0], dft[:, :, 1])
            logmag = np.log(mag + 1e-9)
            avg = cv2.GaussianBlur(logmag, (3, 3), 0)
            spectral = logmag - avg
            exp_spec = np.exp(spectral)
            salmap = (exp_spec - exp_spec.min()) / (exp_spec.max() - exp_spec.min() + 1e-9)
        
        salmap = cv2.resize(salmap, (224, 224))
        from scipy.ndimage import gaussian_filter
        salmap = gaussian_filter(salmap, sigma=2)
        
        if salmap.max() > 0:
            salmap = salmap / salmap.max()
        
        return salmap.astype(np.float32)
    
    def create_center_bias_map(self, size=(224, 224)):
        y, x = np.ogrid[:size[0], :size[1]]
        cy, cx = size[0] / 2, size[1] / 2
        
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        
        center_bias = 1 - (dist / max_dist)
        center_bias = center_bias ** 2
        
        return center_bias.astype(np.float32)
    
    def predict_gaze_for_image(self, image_path):
        if self.model is None:
            print("Error: No model loaded")
            return None
        
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error loading image: {image_path}")
            return None
        
        img_resized = cv2.resize(img, (224, 224))
        
        saliency_map = self.compute_saliency(img_resized)
        
        center_bias = self.create_center_bias_map((224, 224))
        
        sal_tensor = torch.from_numpy(saliency_map).unsqueeze(0).unsqueeze(0).to(self.device)
        att_tensor = torch.from_numpy(center_bias).unsqueeze(0).unsqueeze(0).to(self.device)
        
        default_features = torch.tensor([[0.35, 0.17, 7.5, 0.22, 0.05]], device=self.device)
        
        with torch.no_grad():
            pred_gaze = self.model(att_tensor, sal_tensor, default_features)
        
        gaze_points = pred_gaze.cpu().numpy()[0]
        
        print(f"  Model output stats:")
        print(f"    X: mean={gaze_points[:, 0].mean():.3f}, std={gaze_points[:, 0].std():.3f}, "
              f"min={gaze_points[:, 0].min():.3f}, max={gaze_points[:, 0].max():.3f}")
        print(f"    Y: mean={gaze_points[:, 1].mean():.3f}, std={gaze_points[:, 1].std():.3f}, "
              f"min={gaze_points[:, 1].min():.3f}, max={gaze_points[:, 1].max():.3f}")
        
        return gaze_points
    
    def draw_gaze_point(self, frame, x, y, trail):
        trail_length = min(15, int(self.fps * 0.25))
        trail.append((x, y))
        if len(trail) > trail_length:
            trail.pop(0)
        
        for j, (tx, ty) in enumerate(trail):
            alpha = j / len(trail)
            color = (int(50 * alpha), int(50 * alpha), int(255 * alpha))
            radius = max(1, int(3 * alpha))
            cv2.circle(frame, (int(tx), int(ty)), radius, color, -1)
        
        cv2.circle(frame, (int(x), int(y)), 15, (0, 0, 255), 2)
        cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 255), -1)
        cv2.line(frame, (int(x)-10, int(y)), (int(x)+10, int(y)), (255, 255, 255), 1)
        cv2.line(frame, (int(x), int(y)-10), (int(x), int(y)+10), (255, 255, 255), 1)
        
        return trail
    
    def draw_info_panel(self, frame, current_time, total_time, mode):
        h, w = frame.shape[:2]
        panel_h = 100
        panel_w = 300
        
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        panel[:] = (20, 20, 20)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1
        y_offset = 25
        
        cv2.putText(panel, f"Time: {current_time:.2f}s / {total_time:.1f}s", 
                   (10, y_offset), font, font_scale, color, thickness)
        y_offset += 25
        
        cv2.putText(panel, f"Mode: {mode}", 
                   (10, y_offset), font, font_scale, (100, 255, 255), thickness)
        y_offset += 25
        
        cv2.putText(panel, f"FPS: {self.fps}", 
                   (10, y_offset), font, font_scale, (150, 200, 255), thickness)
        
        frame[10:10+panel_h, w-panel_w-10:w-10] = cv2.addWeighted(
            frame[10:10+panel_h, w-panel_w-10:w-10], 0.3, panel, 0.7, 0
        )
        
        return frame
    
    def generate_video_predicted(self, image_path, output_path, duration=30):
        if self.model is None:
            print("Error: Model not loaded. Use --model path/to/model.pth")
            return False
        
        print(f"Generating predicted gaze video for: {image_path}")
        
        gaze_points = self.predict_gaze_for_image(image_path)
        if gaze_points is None:
            return False
        
        img = cv2.imread(str(image_path))
        if img is None:
            return False
        
        h, w = img.shape[:2]
        
        xs = gaze_points[:, 0]
        ys = gaze_points[:, 1]
        
        xs_smooth, ys_smooth = self.smooth_gaze_path(xs, ys, window_length=21, polyorder=3)
        
        num_frames = int(duration * self.fps)
        frame_indices = np.linspace(0, len(xs_smooth) - 1, num_frames).astype(int)
        
        frame_xs = (xs_smooth[frame_indices] * w).astype(int)
        frame_ys = (ys_smooth[frame_indices] * h).astype(int)
        
        frame_xs = np.clip(frame_xs, 0, w - 1)
        frame_ys = np.clip(frame_ys, 0, h - 1)
        
        print(f"  Generating {num_frames} frames ({duration}s) with smooth gaze path")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (w, h))
        
        trail = []
        
        for i in tqdm(range(num_frames), desc="Rendering predicted video"):
            frame = img.copy()
            
            x, y = frame_xs[i], frame_ys[i]
            trail = self.draw_gaze_point(frame, x, y, trail)
            
            current_time = i / self.fps
            frame = self.draw_info_panel(frame, current_time, duration, "AI Predicted")
            
            out.write(frame)
        
        out.release()
        
        print(f"Video saved: {output_path}")
        print(f"  Duration: {duration}s | Frames: {num_frames} | FPS: {self.fps}")
        
        return True
    
    def generate_video_from_recording(self, recording_id, output_dir='gaze_videos'):
        print(f"Generating video from recording: {recording_id}")
        
        gaze_file = Path('recordings') / recording_id / 'gaze_positions_on_surface_Surface 1.csv'
        if not gaze_file.exists():
            print(f"Error: gaze positions not found for {recording_id}")
            return False
        
        gaze_df = pd.read_csv(gaze_file)
        if 'gaze detected on surface' in gaze_df.columns:
            gaze_df = gaze_df[gaze_df['gaze detected on surface'] == True]
        gaze_df = gaze_df.sort_values('timestamp [ns]')
        
        xs = gaze_df['gaze position on surface x [normalized]'].values
        ys = gaze_df['gaze position on surface y [normalized]'].values
        timestamps = gaze_df['timestamp [ns]'].values
        timestamps_rel = (timestamps - timestamps[0]) / 1e9
        
        print(f"  Loaded {len(xs)} samples (~{len(xs)/timestamps_rel[-1]:.0f}Hz)")
        
        image_files = list(Path('dataset/test/images').glob("*.png"))
        if not image_files:
            print("Error: No test images found")
            return False
        
        img = cv2.imread(str(image_files[0]))
        if img is None:
            return False
        
        h, w = img.shape[:2]
        
        xs_px = (xs * w).astype(int)
        ys_px = (ys * h).astype(int)
        
        xs_smooth, ys_smooth = self.smooth_gaze_path(xs_px, ys_px, window_length=21, polyorder=3)
        
        total_duration = timestamps_rel[-1]
        num_frames = int(total_duration * self.fps)
        frame_timestamps = np.linspace(0, total_duration, num_frames)
        
        fx = interp1d(timestamps_rel, xs_smooth, kind='cubic', bounds_error=False, fill_value='extrapolate')
        fy = interp1d(timestamps_rel, ys_smooth, kind='cubic', bounds_error=False, fill_value='extrapolate')
        
        frame_xs = np.clip(fx(frame_timestamps).astype(int), 0, w - 1)
        frame_ys = np.clip(fy(frame_timestamps).astype(int), 0, h - 1)
        
        print(f"  Generating {num_frames} frames ({total_duration:.1f}s) with smooth gaze path")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        video_path = output_path / f'{recording_id}_smooth.mp4'
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, self.fps, (w, h))
        
        trail = []
        
        for i in tqdm(range(num_frames), desc="Rendering recording video"):
            frame = img.copy()
            
            x, y = frame_xs[i], frame_ys[i]
            trail = self.draw_gaze_point(frame, x, y, trail)
            
            current_time = frame_timestamps[i]
            frame = self.draw_info_panel(frame, current_time, total_duration, "Human Recording")
            
            out.write(frame)
        
        out.release()
        
        print(f"Video saved: {video_path}")
        print(f"  Duration: {total_duration:.1f}s | Frames: {num_frames} | FPS: {self.fps}")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate eye-tracking video (predicted or from recording)'
    )
    parser.add_argument('--mode', choices=['predict', 'recording'], required=True,
                       help='Mode: predict (AI model) or recording (human data)')
    parser.add_argument('--image', help='Path to test image (for predict mode)')
    parser.add_argument('--recording', help='Recording ID (for recording mode)')
    parser.add_argument('--model', default='gaze_model_best.pth',
                       help='Path to trained model (for predict mode)')
    parser.add_argument('--output', help='Output video path (optional)')
    parser.add_argument('--duration', type=int, default=30,
                       help='Video duration in seconds (for predict mode)')
    parser.add_argument('--fps', type=int, default=30, help='Video FPS')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    if args.mode == 'predict':
        if not args.image:
            print("Error: --image required for predict mode")
            return
        
        generator = GazeVideoGenerator(fps=args.fps, model_path=args.model, device=args.device)
        
        if args.output:
            output_path = args.output
        else:
            image_name = Path(args.image).stem
            output_path = f'gaze_videos/{image_name}_predicted.mp4'
        
        Path('gaze_videos').mkdir(exist_ok=True)
        success = generator.generate_video_predicted(args.image, output_path, args.duration)
        
    elif args.mode == 'recording':
        if not args.recording:
            print("Error: --recording required for recording mode")
            return
        
        generator = GazeVideoGenerator(fps=args.fps)
        success = generator.generate_video_from_recording(args.recording)
    
    if success:
        print("\nDone!")
    else:
        print("\nError: Video generation failed")


if __name__ == "__main__":
    main()

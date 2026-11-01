"""Single-stage four-class comparison model using the original Figure 3 scale."""
import os
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import timm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve()
TRAIN = ROOT / 'single stage (four-class classification) training codes'

def widths(scale):
    return [max(8, round(c * scale / 8) * 8) for c in (384, 192, 96, 48, 24)]

def decoder_parameters(ws, skip=True, classes=4):
    total, prev = 0, 768
    for i, w in enumerate(ws):
        extra = (384, 192, 96)[i] if skip and i < 3 else 0
        total += prev * w * 4 + w + 2 * w
        total += (w + extra) * w * 9 + w + 2 * w
        prev = w
    return total + prev * classes + classes

class Model(nn.Module):
    def __init__(self, variant, pretrained=False):
        super().__init__()
        self.encoder = timm.create_model('convnext_tiny.fb_in1k', pretrained=pretrained,
                                        features_only=True, out_indices=(0, 1, 2, 3))
        encoder_n = sum(p.numel() for p in self.encoder.parameters())
        self.target = 2 * encoder_n + decoder_parameters(widths(1), classes=1) + decoder_parameters(widths(1))
        if variant != 'A':
            raise ValueError('Only the original-scale single-stage comparison is supported')
        self.widths = widths(1)
        self.skip = True
        self.up, self.fuse = nn.ModuleList(), nn.ModuleList()
        prev = 768
        for i, w in enumerate(self.widths):
            extra = (384, 192, 96)[i] if self.skip and i < 3 else 0
            self.up.append(nn.Sequential(nn.ConvTranspose2d(prev, w, 2, 2), nn.BatchNorm2d(w), nn.ReLU(True)))
            self.fuse.append(nn.Sequential(nn.Conv2d(w + extra, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(True)))
            prev = w
        self.head = nn.Conv2d(prev, 4, 1)

    def forward(self, x):
        fs = self.encoder(x)
        x = fs[-1]
        for i, (up, fuse) in enumerate(zip(self.up, self.fuse)):
            x = up(x)
            if self.skip and i < 3:
                x = torch.cat((x, fs[2-i]), dim=1)
            x = fuse(x)
        return self.head(x)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=TRAIN/'convnext_tiny_single_stage_results/best_convnext_tiny_single_stage.pth')
    parser.add_argument('--image_dir', type=Path, default=ROOT/'image datasets/validation dataset')
    parser.add_argument('--output_dir', type=Path, default=Path(__file__).resolve().parent/'output')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--limit', type=int, default=None, help='Optional maximum number of images')
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint missing: {args.checkpoint}')
    files = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg'})
    if args.limit is not None:
        if args.limit < 1:
            parser.error('--limit must be positive')
        files = files[:args.limit]
    if not files:
        raise ValueError('No input images')
    if len({p.stem.lower() for p in files}) != len(files):
        raise ValueError('Duplicate image stems would overwrite output files')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model('A').to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt.get('state_dict', ckpt), strict=True)
    transform = A.Compose([A.Resize(512,512), A.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225]), ToTensorV2()])
    palette = np.array([[0,0,0],[0,0,255],[255,0,255],[0,255,0]], dtype=np.uint8)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for i, path in enumerate(files,1):
            rgb = np.asarray(Image.open(path).convert('RGB'))
            tensor = transform(image=rgb)['image'][None].to(device)
            labels = model(tensor).argmax(1)[0].cpu().numpy().astype(np.uint8)
            rgb = cv2.resize(rgb, (512, 512))  # Match other comparison visualization outputs.
            colored = palette[labels]
            overlay = rgb.copy()
            mask = labels > 0
            overlay[mask] = (rgb[mask].astype(np.float32)*(1-args.alpha) + colored[mask]*args.alpha).astype(np.uint8)
            Image.fromarray(overlay).save(args.output_dir/f'{path.stem}_overlay.png')
            Image.fromarray(((labels==1)*255).astype(np.uint8)).save(args.output_dir/f'{path.stem}_mask.png')
            print(f'{i}/{len(files)} {path.name}',flush=True)


if __name__ == '__main__':
    main()

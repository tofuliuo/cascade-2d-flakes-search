"""
MiT 
,

:
    # MiT B0 
    python visualize_mit.py --model mit_b0_binary_results/best_mit_b0_binary.pth --num_classes 1

    # MiT B2 
    python visualize_mit.py --model mit_b2_binary_results/best_mit_b2_binary.pth --num_classes 1 --model_name mit_b2

    # 
    python visualize_mit.py --checkpoint best.pth --image_dir ./test --output_dir ./output
"""

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

#  HF_HOME 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
os.environ['HF_HOME'] = BASE_DIR
os.environ['TRANSFORMERS_CACHE'] = BASE_DIR


# =====================  =====================
class MiTUNet(nn.Module):
    """ MiT + U-Net """

    def __init__(self, encoder_name='mit_b0', encoder_weights=None, in_channels=3, classes=1, activation='sigmoid'):
        super().__init__()

        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, 512, 512)
            features = self.encoder(dummy_input)[2:]
            encoder_channels = [f.shape[1] for f in features]

        self.encoder_channels = encoder_channels

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[3], encoder_channels[2], kernel_size=2, stride=2),
            nn.BatchNorm2d(encoder_channels[2]),
            nn.ReLU(inplace=True)
        )
        self.decoder1_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[2] * 2, encoder_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_channels[2]),
            nn.ReLU(inplace=True)
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[2], encoder_channels[1], kernel_size=2, stride=2),
            nn.BatchNorm2d(encoder_channels[1]),
            nn.ReLU(inplace=True)
        )
        self.decoder2_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[1] * 2, encoder_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_channels[1]),
            nn.ReLU(inplace=True)
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[1], encoder_channels[0], kernel_size=2, stride=2),
            nn.BatchNorm2d(encoder_channels[0]),
            nn.ReLU(inplace=True)
        )
        self.decoder3_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[0] * 2, encoder_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_channels[0]),
            nn.ReLU(inplace=True)
        )

        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[0], encoder_channels[0] // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(encoder_channels[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(encoder_channels[0] // 2, encoder_channels[0] // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_channels[0] // 2),
            nn.ReLU(inplace=True)
        )

        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[0] // 2, encoder_channels[0] // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(encoder_channels[0] // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(encoder_channels[0] // 4, encoder_channels[0] // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_channels[0] // 4),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(encoder_channels[0] // 4, classes, kernel_size=1)

        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        else:
            self.activation = None

    def forward(self, x):
        features = self.encoder(x)[2:]
        x = features[3]
        x = self.decoder1(x)
        x = torch.cat([x, features[2]], dim=1)
        x = self.decoder1_conv(x)
        x = self.decoder2(x)
        x = torch.cat([x, features[1]], dim=1)
        x = self.decoder2_conv(x)
        x = self.decoder3(x)
        x = torch.cat([x, features[0]], dim=1)
        x = self.decoder3_conv(x)
        x = self.decoder4(x)
        x = self.decoder5(x)
        x = self.final_conv(x)
        if self.activation:
            x = self.activation(x)
        return x


def get_model(encoder_name='mit_b0', num_classes=1, activation='sigmoid', device='cpu'):
    """ MiT UNet """
    model = MiTUNet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=num_classes,
        activation=activation
    )
    return model.to(device)


def load_checkpoint(model, checkpoint_path, device='cpu'):
    """"""
    print(f": {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        print(f"  : {len(missing)} ")
    if unexpected:
        print(f"  : {len(unexpected)} ")

    model.eval()
    print("!")
    return model


# =====================  =====================
def preprocess_image(image_path, target_size=(512, 512)):
    """"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f": {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, target_size)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).float()
    return image, resized, tensor


# =====================  () =====================
def inference_binary(model, tensor, device, threshold=0.5):
    """"""
    with torch.no_grad():
        tensor = tensor.unsqueeze(0).to(device)
        output = model(tensor)
        output = output.squeeze().cpu().numpy()
    pred_mask = (output > threshold).astype(np.uint8)
    return pred_mask, output


def create_overlay_binary(original_img, prediction, threshold=0.5, alpha=0.4, color='green'):
    """"""
    binary_mask = (prediction > threshold).astype(np.uint8)

    color_map = {
        'green': [0, 255, 0],
        'red': [255, 0, 0],
        'blue': [0, 0, 255],
        'yellow': [255, 255, 0],
    }
    overlay_color = np.array(color_map.get(color, [0, 255, 0]))

    colored_mask = np.zeros_like(original_img)
    colored_mask[binary_mask == 1] = overlay_color

    overlaid = original_img.copy().astype(np.float32)
    for c in range(3):
        overlaid[:, :, c] = np.where(
            binary_mask == 1,
            overlaid[:, :, c] * (1 - alpha) + colored_mask[:, :, c] * alpha,
            overlaid[:, :, c]
        )
    overlaid = np.clip(overlaid, 0, 255).astype(np.uint8)

    return overlaid, binary_mask


def visualize_binary(model, image_path, output_path, device, threshold=0.5, alpha=0.4, color='green'):
    """"""
    original_img, resized_img, tensor = preprocess_image(image_path)
    pred_mask, prediction = inference_binary(model, tensor, device, threshold)

    overlay, binary_mask = create_overlay_binary(resized_img, prediction, threshold, alpha, color)

    # 
    heatmap = cv2.applyColorMap((prediction * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(binary_mask * 255, cmap='gray')
    axes[0, 1].set_title('Binary Mask', fontsize=14)
    axes[0, 1].axis('off')

    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title(f'Overlay (alpha={alpha})', fontsize=14)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(heatmap)
    axes[1, 1].set_title('Probability Heatmap', fontsize=14)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 
    overlay_path = output_path.replace('_visualization.png', '_overlay.png')
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return pred_mask.sum(), prediction.mean()


# =====================  =====================
def scan_image_dir(image_dir):
    """"""
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = []
    for f in os.listdir(image_dir):
        if f.lower().endswith(extensions):
            image_files.append(f)
    image_files.sort()
    return image_files


def main():
    parser = argparse.ArgumentParser(description='MiT ')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                       help='')
    parser.add_argument('--encoder', '-e', type=str, default='mit_b2',
                       choices=['mit_b0', 'mit_b2', 'mit_b4'],
                       help='')
    parser.add_argument('--num_classes', type=int, default=1,
                       help=' (1=, 3=)')
    parser.add_argument('--image_dir', '-i', type=str,
                       default=os.path.join(PROJECT_ROOT, 'image datasets', 'validation dataset'),
                       help='')
    parser.add_argument('--output_dir', '-o', type=str, default=None,
                       help=' (: mit_b2_stage1_results/visualization)')
    parser.add_argument('--num_images', '-n', type=int, default=None,
                       help=' (: )')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                       help='')
    parser.add_argument('--alpha', type=float, default=0.4,
                       help='')
    parser.add_argument('--color', type=str, default='green',
                       choices=['green', 'red', 'blue', 'yellow'],
                       help='')
    parser.add_argument('--image_size', type=int, default=512,
                       help='')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f": {device}")

    #  checkpoint 
    if args.checkpoint is None:
        if args.num_classes == 1:
            args.checkpoint = os.path.join(PROJECT_ROOT, 'stage1 mit-b2 (binary classification) training codes', f'{args.encoder}_stage1_results', f'best_{args.encoder}_stage1.pth')
        else:
            args.checkpoint = os.path.join(BASE_DIR, f'{args.encoder}_stage2_results', f'best_{args.encoder}_stage2.pth')

    # 
    if args.output_dir is None:
        args.output_dir = os.path.join(BASE_DIR, f'{args.encoder}_stage1_results', 'visualization')
    os.makedirs(args.output_dir, exist_ok=True)

    # 
    if args.num_classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    model = get_model(args.encoder, args.num_classes, activation, device)
    model = load_checkpoint(model, args.checkpoint, device)

    # 
    if not os.path.exists(args.image_dir):
        print(f": {args.image_dir}")
        return

    image_files = scan_image_dir(args.image_dir)
    if not image_files:
        print(f"")
        return

    if args.num_images is not None:
        image_files = image_files[:args.num_images]

    print(f" {len(image_files)} ,")

    # 
    target_size = (args.image_size, args.image_size)
    for i, filename in enumerate(tqdm(image_files, desc="")):
        image_path = os.path.join(args.image_dir, filename)
        name = os.path.splitext(filename)[0]
        output_path = os.path.join(args.output_dir, f'{name}_visualization.png')

        try:
            if args.num_classes == 1:
                n_fg, avg_prob = visualize_binary(
                    model, image_path, output_path, device,
                    args.threshold, args.alpha, args.color
                )
                print(f"[{i+1}/{len(image_files)}] {filename}: ={n_fg}, ={avg_prob:.4f}")
            else:
                print(f" visualize_mit_stage2.py ")

        except Exception as e:
            print(f" {filename}: {e}")

    print(f"\n! : {args.output_dir}")


if __name__ == '__main__':
    main()

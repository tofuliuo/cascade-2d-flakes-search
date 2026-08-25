"""
MiT + U-Net Stage 2 
:label 0=(), label 1=(), label 2=(), label 3=()
 stage 1 

:
    python visualize_inference_mit_stage2.py --model b0
    python visualize_inference_mit_stage2.py --model b2
    python visualize_inference_mit_stage2.py --model b0 --compare
"""

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# HF_HOME,

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
os.environ['HF_HOME'] = BASE_DIR
os.environ['TRANSFORMERS_CACHE'] = BASE_DIR

import segmentation_models_pytorch as smp


# =====================  =====================
# label 0 =  → 
# label 1 =  → 
# label 2 =  → 
# label 3 =  → 
CLASS_COLORS = {
    0: [128, 128, 128],    #  - 
    1: [0, 255, 0],        #  - 
    2: [255, 0, 255],      #  - 
    3: [255, 255, 0],      #  - 
}
CLASS_NAMES = {
    0: 'Background',
    1: 'Single Layer',
    2: 'Contaminant',
    3: 'Multilayer',
}


# =====================  =====================
class MiTUNet(nn.Module):
    """MiT + U-Net"""

    def __init__(self, encoder_name='mit_b0', encoder_weights=None, in_channels=3, classes=4, activation=None):
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


def load_model(model_path, encoder_name='mit_b0', device='cpu', classes=4, activation=None):
    """ MiT + U-Net ."""
    print(f": {encoder_name}")
    print(f": {model_path}")

    model = MiTUNet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=classes,
        activation=activation
    )

    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    final_w = state_dict.get('final_conv.weight', None)
    if final_w is not None and final_w.shape[0] != classes:
        raise ValueError(
            f": final_conv.weight[0]={final_w.shape[0]},"
            f" classes={classes}. checkpoint ."
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    print(f"missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    if len(missing) > 0:
        print(f"missing sample: {missing[:5]}")
    if len(unexpected) > 0:
        print(f"unexpected sample: {unexpected[:5]}")

    model = model.to(device)
    model.eval()
    print("!")
    return model


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


def inference_stage2(model, tensor, device):
    """Stage 2 inference: logits -> softmax probabilities + argmax class."""
    with torch.no_grad():
        tensor = tensor.unsqueeze(0).to(device)

        logits = model(tensor)

        probs = torch.softmax(logits, dim=1)

        pred_mask = torch.argmax(
            logits,
            dim=1
        ).squeeze().cpu().numpy().astype(np.uint8)

    return pred_mask, probs[0].cpu().numpy()


def inference_two_stage(
    stage1_model,
    stage2_model,
    tensor,
    device,
    stage1_threshold=0.5
):
    """
    Final cascade:
    Single Layer = Stage1 Foreground - Stage2 Contaminant - Stage2 Multilayer
    """
    with torch.no_grad():
        stage1_in = tensor.unsqueeze(0).to(device)

        # Stage 1 model already contains sigmoid
        stage1_prob = stage1_model(stage1_in).squeeze().cpu().numpy()

    stage1_fg = (
        stage1_prob > stage1_threshold
    ).astype(np.uint8)

    # Stage 2: pure argmax
    stage2_pred, stage2_probs = inference_stage2(
        stage2_model,
        tensor,
        device
    )

    stage2_multilayer = (
        (stage2_pred == 3) & (stage1_fg > 0)
    ).astype(np.uint8)

    stage2_contaminant = (
        (stage2_pred == 2) & (stage1_fg > 0)
    ).astype(np.uint8)

    single_layer = (
        stage1_fg.astype(bool)
        & ~stage2_multilayer.astype(bool)
        & ~stage2_contaminant.astype(bool)
    ).astype(np.uint8)

    pred_4class = np.zeros_like(stage2_pred, dtype=np.uint8)

    pred_4class[single_layer > 0] = 1
    pred_4class[stage2_contaminant > 0] = 2
    pred_4class[stage2_multilayer > 0] = 3

    return pred_4class, stage2_probs


def create_colored_mask(pred_mask, num_classes=4):
    """mask"""
    h, w = pred_mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id in range(num_classes):
        mask_pixels = (pred_mask == class_id)
        colored[mask_pixels] = CLASS_COLORS[class_id]

    return colored


def create_overlay_visualization(original_img, pred_mask, alpha=0.5):
    """
    
    :=, =, =
    """
    overlay = original_img.copy().astype(np.float32)

    for class_id in [1, 2, 3]:
        color = CLASS_COLORS[class_id]
        mask = (pred_mask == class_id)
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask,
                overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                overlay[:, :, c]
            )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def create_multi_view_visualization(original_img, pred_mask, alpha=0.5):
    """
    ( stage 1 )
    1. 
    2. mask()
    3. 
    4-6. mask
    """
    h, w = pred_mask.shape

    # mask
    mask_background = (pred_mask == 0).astype(np.uint8) * 255
    mask_single_layer = (pred_mask == 1).astype(np.uint8) * 255
    mask_contaminant = (pred_mask == 2).astype(np.uint8) * 255
    mask_multilayer = (pred_mask == 3).astype(np.uint8) * 255

    # mask
    colored_mask = create_colored_mask(pred_mask)

    # 
    overlay = create_overlay_visualization(original_img, pred_mask, alpha)

    # :24
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # 1: | mask | 
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(colored_mask)
    axes[0, 1].set_title('Segmentation Mask\nGray=BG, Green=SL, Magenta=CT, Yellow=ML', fontsize=10)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    axes[0, 2].set_title(f'Overlay (alpha={alpha})', fontsize=12)
    axes[0, 2].axis('off')

    # 4:
    axes[0, 3].axis('off')
    legend_text = "Legend:\n" + \
                  "■ Gray: Background (0)\n" + \
                  "■ Green: Single Layer (1)\n" + \
                  "■ Magenta: Contaminant (2)\n" + \
                  "■ Yellow: Multilayer (3)"
    axes[0, 3].text(0.1, 0.5, legend_text, fontsize=11, verticalalignment='center',
                    fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2:mask
    axes[1, 0].imshow(mask_background, cmap='gray')
    axes[1, 0].set_title(f'Class 0: Background', fontsize=11)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(mask_single_layer, cmap='gray')
    axes[1, 1].set_title(f'Class 1: Single Layer', fontsize=11)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(mask_contaminant, cmap='gray')
    axes[1, 2].set_title(f'Class 2: Contaminant', fontsize=11)
    axes[1, 2].axis('off')

    axes[1, 3].imshow(mask_multilayer, cmap='gray')
    axes[1, 3].set_title(f'Class 3: Multilayer', fontsize=11)
    axes[1, 3].axis('off')

    plt.tight_layout()

    # 
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    combined = np.array(buf)[:, :, :3]
    combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    plt.close(fig)

    return combined


def visualize_single_image(stage1_model, stage2_model, image_path, device, alpha=0.5,
                          stage1_threshold=0.5):
    """"""
    original_img, resized_img, tensor = preprocess_image(image_path)
    pred_mask, probs = inference_two_stage(
        stage1_model, stage2_model, tensor, device,
        stage1_threshold=stage1_threshold
    )

    overlay = create_overlay_visualization(resized_img, pred_mask, alpha)
    single_layer_mask = (pred_mask == 1).astype(np.uint8) * 255

    return overlay, single_layer_mask, pred_mask, probs


def main():
    parser = argparse.ArgumentParser(description='MiT Stage2 ')
    parser.add_argument('--model', type=str, default='b2', choices=['b0', 'b2'],
                       help=': b0  b2 (: b0)')
    parser.add_argument('--compare', action='store_true',
                       help='B0B2')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help=' Stage2 checkpoint (: )')
    parser.add_argument('--stage1_checkpoint', type=str, default=None,
                       help=' Stage1 checkpoint (: )')
    parser.add_argument('--image_dir', type=str,
                       default=os.path.join(PROJECT_ROOT, 'image datasets', 'validation dataset'),
                       help='')
    parser.add_argument('--output_dir', type=str, default=None,
                       help=' (: mit_b2_stage2_results/visualization)')
    parser.add_argument('--alpha', type=float, default=0.5, help='')
    parser.add_argument('--stage1_threshold', type=float, default=0.5,
                       help='Stage1 ')
    parser.add_argument('--debug_counts', action='store_true',
                       help=',/')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f": {device}")

    # 
    if args.output_dir is None:
        if args.compare:
            args.output_dir = os.path.join(BASE_DIR, 'visualization_compare')
        else:
            args.output_dir = os.path.join(BASE_DIR, f'mit_{args.model}_stage2_results', 'visualization')
    os.makedirs(args.output_dir, exist_ok=True)

    # 
    image_dir = args.image_dir
    if not os.path.exists(image_dir):
        print(f": {image_dir}")
        return

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f": {image_dir}")
        return

    print(f" {len(image_files)} ,")

    def resolve_path(p):
        if p is None:
            return p
        return p if os.path.isabs(p) else os.path.join(BASE_DIR, p)

    def get_stage2_model_path(model_key):
        """ Stage2 ."""
        if args.checkpoint and not args.compare:
            return args.checkpoint
        candidates = [
            os.path.join(BASE_DIR, f'mit_{model_key}_stage2_mit_4class_results', f'best_mit_{model_key}_stage2_mit_4class.pth'),
            os.path.join(BASE_DIR, f'mit_{model_key}_stage2_mit_4class_results', f'final_mit_{model_key}_stage2_mit_4class.pth'),
            os.path.join(PROJECT_ROOT, 'stage2 mit-b2 (ternary classification) training codes', f'mit_{model_key}_stage2_results', f'best_mit_{model_key}_stage2.pth'),
            os.path.join(PROJECT_ROOT, 'stage2 mit-b2 (ternary classification) training codes', f'mit_{model_key}_stage2_results', f'final_mit_{model_key}_stage2.pth'),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return candidates[0]

    def get_stage1_model_path(model_key):
        """ Stage1 ."""
        if args.stage1_checkpoint and not args.compare:
            return args.stage1_checkpoint
        candidates = [
            os.path.join(PROJECT_ROOT, 'stage1 mit-b2 (binary classification) training codes', f'mit_{model_key}_stage1_results', f'best_mit_{model_key}_stage1.pth'),
            os.path.join(PROJECT_ROOT, 'stage1 mit-b2 (binary classification) training codes', f'mit_{model_key}_stage1_results', f'final_mit_{model_key}_stage1.pth'),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return candidates[0]

    if args.compare:
        #  B0  B2
        model_b0_stage1_path = get_stage1_model_path('b0')
        model_b2_stage1_path = get_stage1_model_path('b2')
        model_b0_stage2_path = get_stage2_model_path('b0')
        model_b2_stage2_path = get_stage2_model_path('b2')

        if args.stage1_checkpoint:
            model_b0_stage1_path = args.stage1_checkpoint
            model_b2_stage1_path = args.stage1_checkpoint
        if args.checkpoint:
            model_b0_stage2_path = args.checkpoint
            model_b2_stage2_path = args.checkpoint
        model_b0_stage1_path = resolve_path(model_b0_stage1_path)
        model_b2_stage1_path = resolve_path(model_b2_stage1_path)
        model_b0_stage2_path = resolve_path(model_b0_stage2_path)
        model_b2_stage2_path = resolve_path(model_b2_stage2_path)

        print("\n" + "=" * 50)
        print(" MiT B0 Stage1/Stage2 ...")
        stage1_b0 = load_model(model_b0_stage1_path, 'mit_b0', device, classes=1, activation='sigmoid')
        stage2_b0 = load_model(model_b0_stage2_path, 'mit_b0', device, classes=4, activation=None)

        print("\n" + "=" * 50)
        print(" MiT B2 Stage1/Stage2 ...")
        stage1_b2 = load_model(model_b2_stage1_path, 'mit_b2', device, classes=1, activation='sigmoid')
        stage2_b2 = load_model(model_b2_stage2_path, 'mit_b2', device, classes=4, activation=None)

        for img_file in tqdm(image_files, desc=""):
            img_path = os.path.join(image_dir, img_file)
            img_name = os.path.splitext(img_file)[0]

            _, resized_b0, tensor = preprocess_image(img_path)
            pred_b0, _ = inference_two_stage(
                stage1_b0, stage2_b0, tensor, device,
                stage1_threshold=args.stage1_threshold
            )
            pred_b2, _ = inference_two_stage(
                stage1_b2, stage2_b2, tensor, device,
                stage1_threshold=args.stage1_threshold
            )

            overlay_b0 = create_overlay_visualization(resized_b0, pred_b0, args.alpha)
            overlay_b2 = create_overlay_visualization(resized_b0, pred_b2, args.alpha)

            # :22
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))

            axes[0, 0].imshow(cv2.cvtColor(resized_b0, cv2.COLOR_RGB2BGR))
            axes[0, 0].set_title('Original Image', fontsize=14)
            axes[0, 0].axis('off')

            axes[0, 1].imshow(cv2.cvtColor(overlay_b0, cv2.COLOR_RGB2BGR))
            axes[0, 1].set_title('MiT B0 Stage2', fontsize=14)
            axes[0, 1].axis('off')

            axes[1, 0].imshow(cv2.cvtColor(overlay_b2, cv2.COLOR_RGB2BGR))
            axes[1, 0].set_title('MiT B2 Stage2', fontsize=14)
            axes[1, 0].axis('off')

            # 
            diff = (pred_b0.astype(int) != pred_b2.astype(int)).astype(np.uint8) * 255
            axes[1, 1].imshow(diff, cmap='gray')
            axes[1, 1].set_title('B0 vs B2 Difference', fontsize=14)
            axes[1, 1].axis('off')

            plt.suptitle(f'Image: {img_name}\nGray=BG, Green=SL, Magenta=CT, Yellow=ML', fontsize=13)
            plt.tight_layout()

            output_path = os.path.join(args.output_dir, f"{img_name}_compare.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f": {output_path}")

            # 
            cv2.imwrite(os.path.join(args.output_dir, f"{img_name}_b0_overlay.png"), overlay_b0)
            cv2.imwrite(os.path.join(args.output_dir, f"{img_name}_b2_overlay.png"), overlay_b2)

    else:
        # 
        stage1_path = get_stage1_model_path(args.model)
        stage2_path = get_stage2_model_path(args.model)
        if args.stage1_checkpoint:
            stage1_path = args.stage1_checkpoint
        if args.checkpoint:
            stage2_path = args.checkpoint
        stage1_path = resolve_path(stage1_path)
        stage2_path = resolve_path(stage2_path)
        print(f"Stage1 checkpoint: {stage1_path}")
        print(f"Stage2 checkpoint: {stage2_path}")

        print("\n" + "=" * 50)
        print(f": 0=(), 1=(), 2=(), 3=()")
        print(f"Stage1: >{args.stage1_threshold}")
        print("=" * 50)
        stage1_model = load_model(stage1_path, f'mit_{args.model}', device, classes=1, activation='sigmoid')
        stage2_model = load_model(stage2_path, f'mit_{args.model}', device, classes=4, activation=None)

        for img_file in tqdm(image_files, desc=""):
            img_path = os.path.join(image_dir, img_file)
            img_name = os.path.splitext(img_file)[0]

            try:
                overlay, single_layer_mask, pred_mask, probs = visualize_single_image(
                    stage1_model, stage2_model, img_path,
                    device, args.alpha, args.stage1_threshold
                )
                if args.debug_counts:
                    raw_stage2 = np.argmax(probs, axis=0).astype(np.uint8)
                    print(
                        f"[DEBUG] {img_name}: "
                        f"raw_s2 cls1={int((raw_stage2==1).sum())}, cls2={int((raw_stage2==2).sum())}, cls3={int((raw_stage2==3).sum())}; "
                        f"final cls1={int((pred_mask==1).sum())}, cls2={int((pred_mask==2).sum())}, cls3={int((pred_mask==3).sum())}"
                    )

                # 
                overlay_path = os.path.join(args.output_dir, f"{img_name}_overlay.png")
                cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                # mask
                mask_path = os.path.join(args.output_dir, f"{img_name}_mask.png")
                cv2.imwrite(mask_path, single_layer_mask)

            except Exception as e:
                print(f" {img_file}: {e}")

    print("\n" + "=" * 50)
    print(f"! : {args.output_dir}")
    print("=" * 50)


if __name__ == '__main__':
    main()

# python visualize_inference_mit_stage2.py --model b0

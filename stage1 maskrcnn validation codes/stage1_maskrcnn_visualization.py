"""
Mask R-CNN 
,

:
    python visualize_maskrcnn.py  #  resnet50_maskrcnn_stage1_results/best_resnet50_maskrcnn_stage1.pth
    python visualize_maskrcnn.py -c resnet50_maskrcnn_binary_results/best_resnet50_maskrcnn_binary.pth
"""

import os
import argparse
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from torchvision.models import ResNet50_Weights

import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)


# =====================  =====================
def get_model(num_classes=2, device='cpu'):
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
        trainable_backbone_layers=5,
        min_size=512,
        max_size=512
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes
    )

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 256, num_classes
    )

    return model.to(device)


def load_checkpoint(model, checkpoint_path, device='cpu'):
    """"""
    print(f": {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print("!")
    return model


# =====================  =====================
def preprocess_image(image_path, target_size=(512, 512)):
    """"""
    image = Image.open(image_path).convert('RGB')
    orig_w, orig_h = image.size

    # 
    if (orig_w, orig_h) != target_size:
        image_resized = image.resize(target_size, Image.BILINEAR)
    else:
        image_resized = image

    #  numpy
    image_np = np.array(image)
    image_resized_np = np.array(image_resized)

    #  tensor 
    image_tensor = torch.from_numpy(image_resized_np).permute(2, 0, 1).float() / 255.0


    return image_np, image_resized_np, image_tensor, (orig_h, orig_w)


# =====================  =====================
def visualize_single(model, image_path, output_path, device, score_thresh=0.3, mask_thresh=0.3, target_size=(512, 512)):
    """"""
    # 
    orig_image, resized_image, image_tensor, (orig_h, orig_w) = preprocess_image(image_path, target_size)

    #  (Mask R-CNN  list of [C,H,W] tensors)
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model([image_tensor])[0]

    # 
    keep = outputs['scores'] > score_thresh
    boxes = outputs['boxes'][keep].cpu().numpy()
    masks = outputs['masks'][keep].cpu().numpy()

    # 
    overlay = orig_image.copy()
    mask_overlay = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

    # 
    color = [0, 1, 0]  # RGB  (0, 255, 0)

    for mask in masks:
        mask_resized = cv2.resize(mask[0], (orig_w, orig_h))
        binary_mask = (mask_resized > mask_thresh).astype(np.uint8)

        for c in range(3):
            overlay[:, :, c] = np.where(
                binary_mask > 0,
                (overlay[:, :, c] * 0.5 + color[c] * 255 * 0.5).astype(np.uint8),
                overlay[:, :, c]
            )

        mask_overlay[binary_mask > 0] = [
            int(color[c] * 255) for c in range(3)
        ]

    # 
    name = os.path.splitext(os.path.basename(output_path))[0]
    output_dir = os.path.dirname(output_path)

    # 1. 
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.imshow(orig_image)
    ax1.set_title(f'Original', fontsize=14)
    ax1.axis('off')
    fig1.savefig(os.path.join(output_dir, f'{name}_original.png'), dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # 2. 
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.imshow(overlay)
    ax2.set_title(f'Prediction Overlay (n={len(boxes)})', fontsize=14)
    ax2.axis('off')
    fig2.savefig(os.path.join(output_dir, f'{name}_overlay.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # 3. Mask 
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    ax3.imshow(mask_overlay)
    ax3.set_title('Instance Masks', fontsize=14)
    ax3.axis('off')
    fig3.savefig(os.path.join(output_dir, f'{name}_mask.png'), dpi=150, bbox_inches='tight')
    plt.close(fig3)

    return len(boxes)


def main():
    parser = argparse.ArgumentParser(description='Mask R-CNN ')
    parser.add_argument('--checkpoint', '-c', type=str,
                       default=os.path.join(PROJECT_ROOT, 'stage1 maskrcnn (binary classification) training codes', 'resnet50_maskrcnn_stage1_results', 'best_resnet50_maskrcnn_stage1.pth'),
                       help=' (: resnet50_maskrcnn_stage1_results/best_resnet50_maskrcnn_stage1.pth)')
    parser.add_argument('--image_dir', '-i', type=str,
                       default=os.path.join(PROJECT_ROOT, 'image datasets', 'validation dataset'),
                       help='')
    parser.add_argument('--output_dir', '-o', type=str, default=None,
                       help=' (: resnet50_maskrcnn_stage1_results/visualization)')
    parser.add_argument('--score_thresh', '-s', type=float, default=0.5,
                       help='')
    parser.add_argument('--mask_thresh', type=float, default=0.5,
                       help='Mask ')
    parser.add_argument('--num_classes', type=int, default=2,
                       help=' (+=2)')
    parser.add_argument('--image_size', type=int, default=512,
                       help='')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f": {device}")

    # 
    if args.output_dir is None:
        args.output_dir = os.path.join(BASE_DIR, 'resnet50_maskrcnn_stage1_results', 'visualization')
    os.makedirs(args.output_dir, exist_ok=True)

    # 
    if not os.path.exists(args.image_dir):
        print(f": {args.image_dir}")
        return

    # 
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
    image_paths.sort()

    if len(image_paths) == 0:
        print(f"")
        return

    print(f" {len(image_paths)} ,")

    # 
    model = get_model(num_classes=args.num_classes, device=device)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, device=device)
    else:
        print("checkpoint,COCO")

    # 
    target_size = (args.image_size, args.image_size)
    for i in tqdm(range(len(image_paths))):
        image_path = image_paths[i]
        filename = os.path.basename(image_path)
        name = os.path.splitext(filename)[0]
        output_path = os.path.join(args.output_dir, f'{name}_visualization.png')

        try:
            n_instances = visualize_single(
                model, image_path, output_path, device,
                args.score_thresh, args.mask_thresh, target_size
            )
            print(f"[{i+1}/{len(image_paths)}] {filename}: {n_instances} instances")
        except Exception as e:
            print(f" {filename}: {e}")

    print(f"\n! : {args.output_dir}")


if __name__ == '__main__':
    main()
    # stage 1

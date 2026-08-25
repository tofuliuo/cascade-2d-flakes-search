"""
Mask R-CNN Stage 2 visualization script.
Generates single-layer overlays and masks from:
- Stage 1 foreground prediction
- Stage 2 contaminant / multilayer prediction

Usage:
    python visualize_maskrcnn_stage2.py
    python visualize_maskrcnn_stage2.py --checkpoint resnet50_maskrcnn_stage2_results/best_resnet50_maskrcnn_stage2.pth
    python visualize_maskrcnn_stage2.py --checkpoint best.pth --image_dir ./test_images
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


# =====================  =====================
CLASS_NAMES = {
    0: 'Background',
    1: 'Single Layer',
    2: 'Contaminant',
    3: 'Multilayer',
}
CLASS_COLORS = {
    0: [128, 128, 128],    #  - 
    1: [0, 255, 0],        #  - 
    2: [255, 0, 255],      #  - 
    3: [255, 255, 0],      #  - 
}


# =====================  =====================
def get_stage1_model(device='cpu'):
    """ Stage1 Mask R-CNN  (0=, 1=)."""
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
        trainable_backbone_layers=5,
        min_size=512,
        max_size=512
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 2)
    return model.to(device)


def get_stage2_model(num_classes=4, device='cpu'):
    """ Stage2 Mask R-CNN  (0=, 1=, 2=, 3=)."""
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
        trainable_backbone_layers=5,
        min_size=512,
        max_size=512
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model.to(device)


def load_checkpoint(model, checkpoint_path, device='cpu'):
    """"""
    print(f": {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print("!")
    return model


def maskrcnn_to_semantic_mask(
    outputs,
    image_size=(512, 512),
    score_threshold=0.5,
    mask_threshold=0.5
):
    """Convert Mask R-CNN instance outputs to a semantic mask."""

    semantic_mask = np.zeros(image_size, dtype=np.uint8)

    if len(outputs['scores']) == 0:
        return semantic_mask

    # Instance confidence threshold
    keep = outputs['scores'] > score_threshold

    masks = outputs['masks'][keep]
    labels = outputs['labels'][keep]

    for mask, label in zip(masks, labels):
        # Torchvision already returns [1, H, W] full-image masks
        binary_mask = (
            mask[0] > mask_threshold
        ).cpu().numpy()

        label = int(label.item())

        # Keep class label at predicted pixels
        semantic_mask[binary_mask] = np.maximum(
            semantic_mask[binary_mask],
            label
        )

    return semantic_mask



def stage1_minus_stage2_to_single_layer(stage1_pred, stage2_pred):
    """ = Stage1 - Stage2(+)."""
    stage1_binary = (stage1_pred > 0).astype(np.uint8)
    stage2_multilayer = ((stage2_pred == 3) & (stage1_binary > 0)).astype(np.uint8)
    stage2_contaminant = ((stage2_pred == 2) & (stage1_binary > 0)).astype(np.uint8)
    single_layer = stage1_binary.copy().astype(np.float32)
    single_layer = single_layer - stage2_multilayer.astype(np.float32)
    single_layer = single_layer - stage2_contaminant.astype(np.float32)
    single_layer = (single_layer > 0).astype(np.uint8)

    # : / > ,
    pred_4class = np.zeros_like(stage2_pred, dtype=np.uint8)
    pred_4class[single_layer > 0] = 1
    pred_4class[stage2_contaminant > 0] = 2
    pred_4class[stage2_multilayer > 0] = 3
    return pred_4class


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
def visualize_single(
    stage1_model,
    stage2_model,
    image_path,
    device,
    score_thresh=0.5,
    stage1_score_thresh=0.5,
    target_size=(512, 512),
):
    """"""
    # 
    orig_image, resized_image, image_tensor, (orig_h, orig_w) = preprocess_image(image_path, target_size)

    #  (Mask R-CNN  list of [C,H,W] tensors)
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        stage1_outputs = stage1_model([image_tensor])[0]
        stage2_outputs = stage2_model([image_tensor])[0]

    stage1_pred = maskrcnn_to_semantic_mask(
        stage1_outputs,
        image_size=target_size,
        score_threshold=stage1_score_thresh,
        mask_threshold=0.5
    )

    stage2_pred = maskrcnn_to_semantic_mask(
        stage2_outputs,
        image_size=target_size,
        score_threshold=score_thresh,
        mask_threshold=0.5
    )

    pred_4class = stage1_minus_stage2_to_single_layer(stage1_pred, stage2_pred)

    # ( resize ,)
    overlay = resized_image.copy().astype(np.float32)
    mask_overlay = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    class_counts = {label: int((pred_4class == label).sum()) for label in [0, 1, 2, 3]}

    # ,
    for label in [1, 2, 3]:
        color = CLASS_COLORS[label]
        binary_mask = (pred_4class == label)
        for c in range(3):
            overlay[:, :, c] = np.where(
                binary_mask,
                (overlay[:, :, c] * 0.5 + color[c] * 0.5).astype(np.float32),
                overlay[:, :, c],
            )
        mask_overlay[binary_mask] = color
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    single_layer_mask = (pred_4class == 1).astype(np.uint8) * 255
    return overlay, single_layer_mask, class_counts


def main():
    parser = argparse.ArgumentParser(description='Mask R-CNN Stage2 ')
    parser.add_argument('--checkpoint', '-c', type=str,
                       default=os.path.join(PROJECT_ROOT, 'stage2 maskrcnn (ternary classification) training codes', 'resnet50_maskrcnn_stage2_results', 'best_resnet50_maskrcnn_stage2.pth'),
                       help='Stage2 ')
    parser.add_argument('--stage1_checkpoint', type=str,
                       default=os.path.join(PROJECT_ROOT, 'stage1 maskrcnn (binary classification) training codes', 'resnet50_maskrcnn_stage1_results', 'best_resnet50_maskrcnn_stage1.pth'),
                       help='Stage1 ')
    parser.add_argument('--image_dir', '-i', type=str,
                       default=os.path.join(PROJECT_ROOT, 'image datasets', 'validation dataset'),
                       help='')
    parser.add_argument('--output_dir', '-o', type=str, default=None,
                       help=' (: resnet50_maskrcnn_stage2_results/visualization)')
    parser.add_argument('--score_thresh', '-s', type=float, default=0.5,
                       help='Stage2 ')
    parser.add_argument('--stage1_score_thresh', type=float, default=0.5,
                       help='Stage1 ')
    parser.add_argument('--num_classes', type=int, default=4,
                       help='Mask R-CNN (,=+1)')
    parser.add_argument('--image_size', type=int, default=512,
                       help='')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f": {device}")
    print(f": 0=(), 1=(), 2=(), 3=()")

    # 
    if args.output_dir is None:
        args.output_dir = os.path.join(BASE_DIR, 'resnet50_maskrcnn_stage2_results', 'visualization')
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
    stage1_path = args.stage1_checkpoint
    stage2_path = args.checkpoint
    if not os.path.isabs(stage1_path):
        stage1_path = os.path.join(BASE_DIR, stage1_path)
    if not os.path.isabs(stage2_path):
        stage2_path = os.path.join(BASE_DIR, stage2_path)
    print(f"Stage1: {stage1_path}")
    print(f"Stage2: {stage2_path}")

    stage1_model = get_stage1_model(device=device)
    load_checkpoint(stage1_model, stage1_path, device=device)
    stage2_model = get_stage2_model(num_classes=args.num_classes, device=device)
    load_checkpoint(stage2_model, stage2_path, device=device)

    # 
    target_size = (args.image_size, args.image_size)
    for i in tqdm(range(len(image_paths))):
        image_path = image_paths[i]
        filename = os.path.basename(image_path)
        name = os.path.splitext(filename)[0]
        overlay_path = os.path.join(args.output_dir, f'{name}_overlay.png')
        mask_path = os.path.join(args.output_dir, f'{name}_mask.png')

        try:
            overlay, single_layer_mask, class_counts = visualize_single(
                stage1_model,
                stage2_model,
                image_path,
                device,
                args.score_thresh,
                args.stage1_score_thresh,
                target_size,
            )
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            cv2.imwrite(mask_path, single_layer_mask)
            print(
                f"[{i+1}/{len(image_paths)}] {filename}: "
                f"SL={class_counts[1]}, CT={class_counts[2]}, ML={class_counts[3]}"
            )
        except Exception as e:
            print(f" {filename}: {e}")

    print(f"\n! : {args.output_dir}")


if __name__ == '__main__':
    main()

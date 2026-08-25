"""
Mask R-CNN Stage 2 pure single-layer extraction evaluation script

Pipeline:
    1. Stage 1 Foreground
    2. Stage 2 ForegroundContaminant(class 2)Multilayer(class 3)
    3. : Stage1Foreground - Stage2Multilayer - Stage2Contaminant
    4. GT: COCO Foreground - COCO Multilayer - COCO Contaminant = Single Layer

GT: COCO  (instances_val2019.json)
Metrics: IoU, Dice, Precision, Recall, F1

Usage:
    python evaluate_maskrcnn_stage2.py
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from torchvision.models import ResNet50_Weights

from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# ===================== Dataset =====================
class MaskRCNNEvalDataset(Dataset):
    """Mask R-CNN Dataset"""

    def __init__(self, image_dir, annotation_file, image_size=(512, 512)):
        self.image_dir = image_dir
        self.image_size = image_size

        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.coco_data = json.load(f)

        self.image_id_to_info = {img['id']: img for img in self.coco_data['images']}
        self.image_id_to_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append(ann)

        self.category_id_mapping = {}
        for cat in self.coco_data['categories']:
            name = cat['name']
            if name == 'single layer':
                self.category_id_mapping[cat['id']] = 1
            elif name == 'contaminants':
                self.category_id_mapping[cat['id']] = 2
            elif name == 'multilayer':
                self.category_id_mapping[cat['id']] = 3

        self.valid_image_ids = list(self.image_id_to_annotations.keys())
        self.valid_image_ids.sort()
        print(f"Loaded {len(self.valid_image_ids)} annotated images")

    def __len__(self):
        return len(self.valid_image_ids)

    def __getitem__(self, idx):
        image_id = self.valid_image_ids[idx]
        img_info = self.image_id_to_info[image_id]

        img_path = os.path.join(self.image_dir, img_info['file_name'])
        try:
            image = np.array(Image.open(img_path).convert('RGB'))
        except:
            image = np.zeros((512, 512, 3), dtype=np.uint8)

        orig_h, orig_w = image.shape[:2]

        # 
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size)

        # tensor (C, H, W), [0,1]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0


        # 4-class: 0=Background, 1=Single Layer, 2=Contaminant, 3=Multilayer
        multilayer_mask = np.zeros((512, 512), dtype=np.uint8)
        contaminants_mask = np.zeros((512, 512), dtype=np.uint8)
        single_layer_mask = np.zeros((512, 512), dtype=np.uint8)

        if image_id in self.image_id_to_annotations:
            for ann in self.image_id_to_annotations[image_id]:
                category_id = ann['category_id']
                category_id = self.category_id_mapping.get(category_id, category_id)

                if 'segmentation' in ann and ann['segmentation']:
                    for seg in ann['segmentation']:
                        if len(seg) >= 6:
                            poly_coords = []
                            for i in range(0, len(seg), 2):
                                x = int(seg[i] * 512 / img_info['width'])
                                y = int(seg[i+1] * 512 / img_info['height'])
                                poly_coords.append((x, y))

                            from PIL import ImageDraw
                            temp_mask = Image.new('L', (512, 512), 0)
                            draw = ImageDraw.Draw(temp_mask)
                            draw.polygon(poly_coords, outline=1, fill=1)
                            temp_mask_array = np.array(temp_mask)

                            if category_id == 1:
                                single_layer_mask = np.maximum(single_layer_mask, temp_mask_array)
                            elif category_id == 2:
                                contaminants_mask = np.maximum(contaminants_mask, temp_mask_array)
                            elif category_id == 3:
                                multilayer_mask = np.maximum(multilayer_mask, temp_mask_array)

        # Single Layermask(MultilayerContaminant)
        single_layer_clean = single_layer_mask.copy()
        single_layer_clean[(multilayer_mask > 0) | (contaminants_mask > 0)] = 0

        # 4-class: 0=Background, 1=Single Layer, 2=Contaminant, 3=Multilayer
        gt_mask_4class = np.zeros((512, 512), dtype=np.uint8)
        gt_mask_4class = np.where(multilayer_mask > 0, 3, gt_mask_4class)
        gt_mask_4class = np.where(contaminants_mask > 0, 2, gt_mask_4class)
        gt_mask_4class = np.where(single_layer_clean > 0, 1, gt_mask_4class)

        return image_tensor, gt_mask_4class, image_id


def collate_fn(batch):
    """Custom batch collate"""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


# ===================== Model =====================
def get_maskrcnn_binary_model(num_classes=2, device='cuda'):
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


def get_maskrcnn_4class_model(num_classes=3, device='cuda'):
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
        trainable_backbone_layers=5,
        min_size=512,
        max_size=512
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes + 1
    )

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 256, num_classes + 1
    )

    return model.to(device)



def maskrcnn_to_semantic_mask(
    outputs,
    image_size=(512, 512),
    score_threshold=0.5,
    mask_threshold=0.5
):
    semantic_mask = np.zeros(image_size, dtype=np.uint8)

    if len(outputs['scores']) == 0:
        return semantic_mask

    keep = outputs['scores'] > score_threshold

    masks = outputs['masks'][keep]
    labels = outputs['labels'][keep]

    for mask, label in zip(masks, labels):
        binary_mask = (
            mask[0] > mask_threshold
        ).cpu().numpy()

        label = int(label.item())

        semantic_mask[binary_mask] = np.maximum(
            semantic_mask[binary_mask],
            label
        )

    return semantic_mask



# ===================== Evaluation metrics =====================
def compute_metrics(pred_mask, gt_mask, num_classes):
    """Class IoU, Dice, Precision, Recall, F1"""
    ious, dices, precisions, recalls, f1s = {}, {}, {}, {}, {}

    for cls_id in range(num_classes):
        pred_cls = (pred_mask == cls_id).astype(np.float32)
        gt_cls = (gt_mask == cls_id).astype(np.float32)

        tp = (pred_cls * gt_cls).sum()
        fp = (pred_cls * (1 - gt_cls)).sum()
        fn = ((1 - pred_cls) * gt_cls).sum()

        union = pred_cls.sum() + gt_cls.sum()
        iou = (tp + 1e-6) / (union - tp + 1e-6)
        dice = (2 * tp + 1e-6) / (pred_cls.sum() + gt_cls.sum() + 1e-6)
        precision = (tp + 1e-6) / (tp + fp + 1e-6)
        recall = (tp + 1e-6) / (tp + fn + 1e-6)
        f1 = (2 * precision * recall + 1e-6) / (precision + recall + 1e-6)

        ious[cls_id] = float(iou)
        dices[cls_id] = float(dice)
        precisions[cls_id] = float(precision)
        recalls[cls_id] = float(recall)
        f1s[cls_id] = float(f1)

    return ious, dices, precisions, recalls, f1s


def stage1_minus_stage2_to_single_layer(stage1_pred, stage2_pred, debug=False):
    """
    Stage1Foreground - Stage2MultilayerContaminant = Single Layer

    Stage1: binary classification (0=Background, 1=Foreground)
    Stage2: 4-class (0=Background, 1=Single Layer, 2=Contaminant, 3=Multilayer)

    4-class:
    - Multilayer: stage2Multilayer
    - Contaminant: stage2Contaminant
    - Single Layer: stage1Foreground - stage2Multilayer - stage2Contaminant
    - Background: 
    """
    # Stage1Foreground
    stage1_binary = (stage1_pred > 0).astype(np.uint8)

    # Stage2
    # stage2_pred: 0=Background, 1=Single Layer, 2=Contaminant, 3=Multilayer

    # stage2MultilayerContaminantmask
    stage2_multilayer = (stage2_pred == 3).astype(np.uint8)
    stage2_contaminant = (stage2_pred == 2).astype(np.uint8)

    # Single Layer = stage1Foreground - Multilayer - Contaminant
    single_layer = stage1_binary.copy().astype(np.float32)
    # Multilayer
    single_layer = single_layer - stage2_multilayer.astype(np.float32)
    # Contaminant
    single_layer = single_layer - stage2_contaminant.astype(np.float32)
    # 
    single_layer = (single_layer > 0).astype(np.uint8)

    # 4-class
    pred_4class = np.zeros_like(stage2_pred)

    # Multilayer: stage2
    pred_4class[stage2_pred == 3] = 3

    # Contaminant: stage2
    pred_4class[stage2_pred == 2] = 2

    # Single Layer: stage1Foreground - Multilayer - Contaminant
    pred_4class[single_layer > 0] = 1

    # Background:  (stage2Background + stage1)
    # default0

    return pred_4class


def evaluate_two_stage_method(stage1_model, stage2_model, dataset, device, num_classes=4):
    """stage: Stage1Foreground - Stage2MultilayerContaminant = Single Layer"""
    stage1_model.eval()
    stage2_model.eval()

    single_layer_values = {
        'iou': [],
        'dice': [],
        'precision': [],
        'recall': [],
        'f1': [],
    }

    print("\nStart evaluation "
          "(Stage 1 Foreground - Stage 2 Multilayer/Contaminant = Single Layer)...")

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Evaluating"):
            image_tensor, gt_mask_4class, image_id = dataset[i]

            # Mask R-CNN  3D  [C, H, W]
            image_batch = image_tensor.to(device)

            # Stage1 (Foreground - binary classification)
            stage1_output = stage1_model([image_batch])[0]
            stage1_pred = maskrcnn_to_semantic_mask(stage1_output, score_threshold=0.5)

            # Stage2 (4-class)
            stage2_output = stage2_model([image_batch])[0]
            stage2_pred = maskrcnn_to_semantic_mask(stage2_output, score_threshold=0.5)

            pred_mask_4class = stage1_minus_stage2_to_single_layer(
                stage1_pred, stage2_pred
            )

            # 4-classMetrics
            ious, dices, precisions, recalls, f1s = compute_metrics(
                pred_mask_4class, gt_mask_4class, num_classes
            )

            single_layer_values['iou'].append(ious[1])
            single_layer_values['dice'].append(dices[1])
            single_layer_values['precision'].append(precisions[1])
            single_layer_values['recall'].append(recalls[1])
            single_layer_values['f1'].append(f1s[1])

    return {
        name: np.mean(values) if values else 0.0
        for name, values in single_layer_values.items()
    }


# ===================== Main function =====================
def main():
    parser = argparse.ArgumentParser(description='Mask R-CNN Stage1Foreground - Stage2Multilayer/Contaminant = Single Layer ')
    parser.add_argument('--stage1_checkpoint', type=str,
                       default=os.path.join(PROJECT_ROOT, 'stage1 maskrcnn (binary classification) training codes', 'resnet50_maskrcnn_stage1_results', 'best_resnet50_maskrcnn_stage1.pth'),
                       help='Stage1 ForegroundModel')
    parser.add_argument('--stage2_checkpoint', type=str,
                       default=os.path.join(BASE_DIR, 'resnet50_maskrcnn_stage2_results', 'best_resnet50_maskrcnn_stage2.pth'),
                       help='Stage2 4-classModel')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Result directory (default: maskrcnn_stage1_minus_stage2_results)')
    args = parser.parse_args()

    stage1_path = args.stage1_checkpoint
    stage2_path = args.stage2_checkpoint

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = 4  # 4-class: 0=Background, 1=Single Layer, 2=Contaminant, 3=Multilayer
    evaluation_title = (
        "Mask R-CNN: Stage 1 Foreground - "
        "Stage 2 Multilayer/Contaminant = Single Layer"
    )

    print("=" * 60)
    print(evaluation_title)
    print("=" * 60)
    print(f"Stage1 Model: {stage1_path}")
    print(f"Stage2 Model: {stage2_path}")
    print(f"Device: {device}")

    # Stage1Model (Foregroundbinary classification)
    print("\nLoading Stage 1 model...")
    stage1_model = get_maskrcnn_binary_model(num_classes=2, device=device)
    stage1_checkpoint = torch.load(stage1_path, map_location=device)
    stage1_model.load_state_dict(stage1_checkpoint)
    stage1_model = stage1_model.to(device)
    stage1_model.eval()
    print("Stage 1 model loaded.")

    # Stage2Model (4-class)
    print("\nLoading Stage 2 model...")
    stage2_model = get_maskrcnn_4class_model(num_classes=3, device=device)
    stage2_checkpoint = torch.load(stage2_path, map_location=device)
    stage2_model.load_state_dict(stage2_checkpoint)
    stage2_model = stage2_model.to(device)
    stage2_model.eval()
    print("Stage 2 model loaded.")

    # Dataset
    print("\nLoading validation set...")
    dataset_root = os.path.join(PROJECT_ROOT, "image datasets")
    val_dataset = MaskRCNNEvalDataset(
        image_dir=os.path.join(dataset_root, 'validation dataset'),
        annotation_file=os.path.join(dataset_root, 'annotation COCO dataset', 'instances_val2019.json')
    )
    print(f"Validation set: {len(val_dataset)} images")

    # 
    results = evaluate_two_stage_method(
        stage1_model, stage2_model, val_dataset, device, num_classes=4
    )

    # 
    print("\n" + "=" * 60)
    print(f"({evaluation_title})")
    print("=" * 60)
    print()
    print(f"Single Layer: IoU={results['iou']:.4f}, Dice={results['dice']:.4f}, "
          f"P={results['precision']:.4f}, R={results['recall']:.4f}, F1={results['f1']:.4f}")

    # 
    results_dir = args.output_dir or os.path.join(BASE_DIR, 'maskrcnn_stage1_minus_stage2_results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'evaluation_results.txt')

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"({evaluation_title})\n")
        f.write("=" * 60 + "\n")
        f.write(f"Stage1 Model: {stage1_path}\n")
        f.write(f"Stage2 Model: {stage2_path}\n")
        f.write("\n")
        f.write(f"Single Layer: IoU={results['iou']:.4f}, Dice={results['dice']:.4f}, "
                f"P={results['precision']:.4f}, R={results['recall']:.4f}, F1={results['f1']:.4f}\n")

    print(f"\nResults saved to: {results_file}")
    return results


if __name__ == '__main__':
    main()

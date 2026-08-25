"""
Mask R-CNN Stage 1 foreground segmentation evaluation script

Usage:
    python evaluate_maskrcnn.py
"""

import os
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
import numpy as np
import cv2
from PIL import Image
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.models import ResNet50_Weights

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# ===================== Configuration =====================
class Config:
    def __init__(self, checkpoint=None, score_threshold=0.5, mask_threshold=0.5):
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATASET_ROOT = os.path.join(PROJECT_ROOT, "image datasets")
        self.CHECKPOINT = checkpoint
        self.IMAGE_SIZE = (512, 512)
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.SCORE_THRESHOLD = score_threshold
        self.MASK_THRESHOLD = mask_threshold


# ===================== Dataset =====================
class MaskRCNNEvalDataset(Dataset):
    """Dataset - Stage 1 Foreground (binary classification: Background + Foreground)"""

    def __init__(self, image_dir, annotation_file, config):
        self.image_dir = image_dir
        self.config = config

        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.coco_data = json.load(f)

        self.image_id_to_info = {img['id']: img for img in self.coco_data['images']}
        self.image_id_to_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append(ann)

        self.valid_image_ids = []
        for img_id in self.image_id_to_annotations.keys():
            img_info = self.image_id_to_info[img_id]
            img_path = os.path.join(self.image_dir, img_info['file_name'])
            if os.path.exists(img_path):
                self.valid_image_ids.append(img_id)
        self.valid_image_ids.sort()
        print(f"Loaded {len(self.valid_image_ids)} images")

    def __len__(self):
        return len(self.valid_image_ids)

    def _polys_to_mask(self, polygons, height, width):
        """mask"""
        mask = np.zeros((height, width), dtype=np.uint8)
        from PIL import ImageDraw
        for poly in polygons:
            if len(poly) < 6:
                continue
            coords = []
            for i in range(0, len(poly), 2):
                coords.append((int(poly[i]), int(poly[i+1])))
            temp = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(temp)
            draw.polygon(coords, outline=1, fill=1)
            mask = np.maximum(mask, np.array(temp))
        return mask

    def _get_gt_instances(self, image_id, img_height, img_width):
        """ - Stage 1: ClassForeground(1)"""
        boxes = []
        masks = []
        labels = []

        if image_id not in self.image_id_to_annotations:
            return boxes, masks, labels

        class_masks = {}

        for ann in self.image_id_to_annotations[image_id]:
            category_id = ann['category_id']
            cat_name = None
            for cat in self.coco_data['categories']:
                if cat['id'] == category_id:
                    cat_name = cat['name']
                    break

            # Stage 1: ClassForeground(1)
            if cat_name == 'single layer':
                label = 1
            else:
                continue

            if 'segmentation' not in ann or not ann['segmentation']:
                continue

            class_mask = self._polys_to_mask(ann['segmentation'], img_height, img_width)
            if label not in class_masks:
                class_masks[label] = class_mask.copy()
            else:
                class_masks[label] = np.maximum(class_masks[label], class_mask)

        for label, class_mask in class_masks.items():
            num_labels, labeled, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
            for comp_id in range(1, num_labels):
                component_mask = (labeled == comp_id).astype(np.uint8)

                rows = np.any(component_mask, axis=1)
                cols = np.any(component_mask, axis=0)
                if not rows.any() or not cols.any():
                    continue

                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]

                box_w = cmax - cmin
                box_h = rmax - rmin
                if box_w <= 1 or box_h <= 1:
                    continue

                boxes.append([cmin, rmin, cmax, rmax])
                masks.append(component_mask)
                labels.append(label)

        return boxes, masks, labels

    def __getitem__(self, idx):
        image_id = self.valid_image_ids[idx]
        img_info = self.image_id_to_info[image_id]

        img_path = os.path.join(self.image_dir, img_info['file_name'])
        try:
            image = np.array(Image.open(img_path).convert('RGB'))
        except Exception:
            image = np.zeros((512, 512, 3), dtype=np.uint8)

        orig_h, orig_w = image.shape[:2]

        if image.shape[:2] != self.config.IMAGE_SIZE:
            image = cv2.resize(image, self.config.IMAGE_SIZE)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        scale_x = self.config.IMAGE_SIZE[1] / orig_w
        scale_y = self.config.IMAGE_SIZE[0] / orig_h

        #  GT 
        boxes, masks, labels = self._get_gt_instances(image_id, orig_h, orig_w)

        if len(boxes) == 0:
            gt_target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'masks': torch.zeros((0, *self.config.IMAGE_SIZE), dtype=torch.uint8),
            }
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            labels = torch.tensor(labels, dtype=torch.int64)

            box_w = boxes[:, 2] - boxes[:, 0]
            box_h = boxes[:, 3] - boxes[:, 1]
            valid_mask = (box_w > 1) & (box_h > 1)
            boxes = boxes[valid_mask]
            labels = labels[valid_mask]

            masks_np = np.stack(masks, axis=0)
            masks_np = masks_np[valid_mask.numpy()]
            if len(masks_np) == 0:
                masks_np = np.zeros((0, *self.config.IMAGE_SIZE), dtype=np.uint8)
            elif masks_np.shape[1:] != self.config.IMAGE_SIZE:
                resized_masks = [cv2.resize(m, self.config.IMAGE_SIZE,interpolation=cv2.INTER_NEAREST) for m in masks_np]
                masks_np = np.stack(resized_masks, axis=0)
            masks_tensor = torch.from_numpy(masks_np).float()

            gt_target = {
                'boxes': boxes,
                'labels': labels,
                'masks': masks_tensor,
            }

        return image_tensor, gt_target, image_id

    def get_raw_image(self, idx):
        image_id = self.valid_image_ids[idx]
        img_info = self.image_id_to_info[image_id]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        return Image.open(img_path).convert('RGB'), img_info['file_name']


# ===================== Model =====================
def get_model(num_classes, device):
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


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    image_ids = [item[2] for item in batch]
    return images, targets, image_ids


# ===================== Evaluation metrics =====================
def compute_pixel_metrics(pred_masks, pred_labels, gt_masks, gt_labels, num_classes, image_size,
                          mask_threshold=0.5):
    """
    Evaluation metrics
    Class IoU  Dice
    """
    h, w = image_size if isinstance(image_size, tuple) else (image_size, image_size)

    #  masks  masks
    pred_semantic = np.zeros((h, w), dtype=np.int32)
    gt_semantic = np.zeros((h, w), dtype=np.int32)

    # GT: gt_masks  3D (N, H, W)  2D (H, W)
    if gt_masks.ndim == 3:
        for i in range(len(gt_masks)):
            mask = gt_masks[i]
            label = int(gt_labels[i]) if i < len(gt_labels) else 0
            if mask.ndim > 2:
                mask = mask[0]
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h))
            gt_semantic[mask > 0.5] = label
    else:
        for mask, label in zip(gt_masks, gt_labels):
            if mask.ndim > 2:
                mask = mask[0]
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h))
            gt_semantic[mask > 0.5] = int(label)

    # : pred_masks  3D (N, H, W)  2D (H, W)
    if pred_masks.ndim == 3:
        for i in range(len(pred_masks)):
            mask = pred_masks[i]
            label = int(pred_labels[i]) if i < len(pred_labels) else 0
            if mask.ndim > 2:
                mask = mask[0]
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h))
            pred_semantic[mask > mask_threshold] = label
    else:
        for mask, label in zip(pred_masks, pred_labels):
            if mask.ndim > 2:
                mask = mask[0]
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h))
            pred_semantic[mask > mask_threshold] = int(label)

    # Class IoU, Dice, Precision, Recall, F1
    ious = {}
    dices = {}
    precisions = {}
    recalls = {}
    f1s = {}

    for cls_id in range(num_classes):
        pred_cls = (pred_semantic == cls_id).astype(np.float32)
        gt_cls = (gt_semantic == cls_id).astype(np.float32)

        tp = (pred_cls * gt_cls).sum()
        fp = (pred_cls * (1 - gt_cls)).sum()
        fn = ((1 - pred_cls) * gt_cls).sum()

        union = pred_cls.sum() + gt_cls.sum()

        iou = (tp + 1e-6) / (union - tp + 1e-6)
        dice = (2 * tp + 1e-6) / (pred_cls.sum() + gt_cls.sum() + 1e-6)
        precision = (tp + 1e-6) / (tp + fp + 1e-6)
        recall = (tp + 1e-6) / (tp + fn + 1e-6)
        f1_score = (2 * precision * recall + 1e-6) / (precision + recall + 1e-6)

        ious[cls_id] = iou
        dices[cls_id] = dice
        precisions[cls_id] = precision
        recalls[cls_id] = recall
        f1s[cls_id] = f1_score

    return ious, dices, precisions, recalls, f1s


# ===================== Evaluation function =====================
def evaluate_model(model, val_loader, config):
    """Model - Stage 1 Foreground"""
    model.eval()

    class_ids = range(2)
    all_ious = {i: [] for i in class_ids}
    all_dices = {i: [] for i in class_ids}
    all_precisions = {i: [] for i in class_ids}
    all_recalls = {i: [] for i in class_ids}
    all_f1s = {i: [] for i in class_ids}

    print("\nStart evaluation...")

    with torch.no_grad():
        for images, gt_targets, image_ids in tqdm(val_loader, desc="Evaluating"):
            images = [img.to(config.DEVICE) for img in images]

            # 
            outputs = model(images)

            for i, (output, gt_target) in enumerate(zip(outputs, gt_targets)):
                # 
                keep = output['scores'] > config.SCORE_THRESHOLD
                pred_boxes = output['boxes'][keep].cpu().numpy()
                pred_masks = output['masks'][keep].cpu().numpy()
                pred_labels = output['labels'][keep].cpu().numpy()
                pred_scores = output['scores'][keep].cpu().numpy()

                # GT
                gt_boxes = gt_target['boxes'].cpu().numpy()
                gt_masks = gt_target['masks'].cpu().numpy()
                gt_labels = gt_target['labels'].cpu().numpy()

                # Metrics
                ious, dices, precisions, recalls, f1s = compute_pixel_metrics(
                    pred_masks, pred_labels,
                    gt_masks, gt_labels,
                    2, config.IMAGE_SIZE, config.MASK_THRESHOLD
                )

                for cls_id in class_ids:
                    all_ious[cls_id].append(ious[cls_id])
                    all_dices[cls_id].append(dices[cls_id])
                    all_precisions[cls_id].append(precisions[cls_id])
                    all_recalls[cls_id].append(recalls[cls_id])
                    all_f1s[cls_id].append(f1s[cls_id])

    class_metrics = {}
    for cls_id in class_ids:
        class_metrics[cls_id] = {
            'iou': np.mean(all_ious[cls_id]) if all_ious[cls_id] else 0,
            'dice': np.mean(all_dices[cls_id]) if all_dices[cls_id] else 0,
            'precision': np.mean(all_precisions[cls_id]) if all_precisions[cls_id] else 0,
            'recall': np.mean(all_recalls[cls_id]) if all_recalls[cls_id] else 0,
            'f1': np.mean(all_f1s[cls_id]) if all_f1s[cls_id] else 0
        }

    return {
        'mean_iou': np.mean([class_metrics[i]['iou'] for i in class_ids]),
        'mean_dice': np.mean([class_metrics[i]['dice'] for i in class_ids]),
        'mean_precision': np.mean([class_metrics[i]['precision'] for i in class_ids]),
        'mean_recall': np.mean([class_metrics[i]['recall'] for i in class_ids]),
        'mean_f1': np.mean([class_metrics[i]['f1'] for i in class_ids]),
        'class_metrics': class_metrics,
    }


# ===================== Main function =====================
def main():
    parser = argparse.ArgumentParser(description='Mask R-CNN Stage 1 ')
    parser.add_argument('--checkpoint', '-c', type=str,
                       default=os.path.join(BASE_DIR, 'resnet50_maskrcnn_stage1_results', 'best_resnet50_maskrcnn_stage1.pth'),
                       help='Model')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='')
    parser.add_argument('--score_threshold', type=float, default=0.5,
                        help='detection confidence threshold')
    parser.add_argument('--mask_threshold', type=float, default=0.5,
                        help='predicted mask binarization threshold')
    args = parser.parse_args()

    config = Config(
        checkpoint=args.checkpoint,
        score_threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
    )

    print("=" * 60)
    print("Mask R-CNN Stage 1 Foreground")
    print("=" * 60)
    print(f"Model: {args.checkpoint}")
    print(f"Device: {config.DEVICE}")

    # Model (binary classification: Background + Foreground)
    model = get_model(num_classes=2, device=config.DEVICE)
    checkpoint = torch.load(args.checkpoint, map_location=config.DEVICE)
    model.load_state_dict(checkpoint)
    model = model.to(config.DEVICE)
    model.eval()
    print("Model!")

    # Loading validation set
    print("\nLoading validation set...")
    val_dataset = MaskRCNNEvalDataset(
        image_dir=os.path.join(config.DATASET_ROOT, 'validation dataset'),
        annotation_file=os.path.join(config.DATASET_ROOT, 'annotation COCO dataset', 'instances_val2019.json'),
        config=config,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    print(f"Validation set: {len(val_dataset)} images")

    # 
    results = evaluate_model(model, val_loader, config)

    # 
    print("\n" + "=" * 60)
    print(" (Stage 1 Foreground)")
    print("=" * 60)
    print("Mean metrics (Background + Foreground):")
    print(f"Mean IoU: {results['mean_iou']:.4f}")
    print(f"Mean Dice: {results['mean_dice']:.4f}")
    print(f"Mean Precision: {results['mean_precision']:.4f}")
    print(f"Mean Recall: {results['mean_recall']:.4f}")
    print(f"Mean F1: {results['mean_f1']:.4f}")

    class_names = {0: 'Background', 1: 'Foreground'}
    print("\nClassMetrics:")
    for cls_id, metrics in results['class_metrics'].items():
        name = class_names[cls_id]
        print(f"  {name}: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}, "
              f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

    # 
    checkpoint_dir = os.path.dirname(args.checkpoint)
    results_file = os.path.join(checkpoint_dir, 'evaluation_results.txt')

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Mask R-CNN Stage 1 Foreground\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {args.checkpoint}\n")
        f.write(f"Score threshold: {config.SCORE_THRESHOLD}\n")
        f.write(f"Mask threshold: {config.MASK_THRESHOLD}\n")
        f.write("Mean metrics (Background + Foreground):\n")
        f.write(f"Mean IoU: {results['mean_iou']:.4f}\n")
        f.write(f"Mean Dice: {results['mean_dice']:.4f}\n")
        f.write(f"Mean Precision: {results['mean_precision']:.4f}\n")
        f.write(f"Mean Recall: {results['mean_recall']:.4f}\n")
        f.write(f"Mean F1: {results['mean_f1']:.4f}\n")
        f.write("\nClassMetrics:\n")
        for cls_id, metrics in results['class_metrics'].items():
            name = class_names[cls_id]
            f.write(f"  {name}: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}, "
                    f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}\n")

    print(f"\nResults saved to: {results_file}")

    return results


if __name__ == '__main__':
    main()

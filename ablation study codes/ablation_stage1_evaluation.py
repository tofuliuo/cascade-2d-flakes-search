"""
ConvNeXt threshold sensitivity analysis script
Evaluatethreshold Stage 1 (binary classification)  Stage 2 (4-class) 
GT: COCO  raw single-layer footprint
Metrics: IoU, Precision, Recall, F1

Usage:
    python test_thresholds.py
    python test_thresholds.py --stage1_only
    python test_thresholds.py --stage2_only
"""

import os
import sys
import json
import glob
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image, ImageDraw
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    print("Error: timm ")
    TIMM_AVAILABLE = False



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
PARENT_DIR = os.path.dirname(BASE_DIR)

os.environ['HF_HOME'] = BASE_DIR


# ===================== Model definition =====================
class ConvNeXtUNet(nn.Module):
    """wo_cont Model: skip connection"""
    def __init__(self, encoder_name='convnext_tiny', in_channels=3, classes=1, activation=None):
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm")
        
        self.encoder = timm.create_model(encoder_name, pretrained=False, features_only=True, out_indices=(0, 1, 2, 3))
        
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, 512, 512)
            features = self.encoder(dummy_input)
            encoder_channels = [f.shape[1] for f in features]
        
        self.encoder_channels = encoder_channels
        
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[3], encoder_channels[2], 2, 2),
            nn.BatchNorm2d(encoder_channels[2]), nn.ReLU(inplace=True)
        )
        self.decoder1_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[2], encoder_channels[2], 3, padding=1),
            nn.BatchNorm2d(encoder_channels[2]), nn.ReLU(inplace=True)
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[2], encoder_channels[1], 2, 2),
            nn.BatchNorm2d(encoder_channels[1]), nn.ReLU(inplace=True)
        )
        self.decoder2_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[1], encoder_channels[1], 3, padding=1),
            nn.BatchNorm2d(encoder_channels[1]), nn.ReLU(inplace=True)
        )
        
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[1], encoder_channels[0], 2, 2),
            nn.BatchNorm2d(encoder_channels[0]), nn.ReLU(inplace=True)
        )
        self.decoder3_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[0], encoder_channels[0], 3, padding=1),
            nn.BatchNorm2d(encoder_channels[0]), nn.ReLU(inplace=True)
        )
        
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[0], encoder_channels[0] // 2, 2, 2),
            nn.BatchNorm2d(encoder_channels[0] // 2), nn.ReLU(inplace=True),
            nn.Conv2d(encoder_channels[0] // 2, encoder_channels[0] // 2, 3, padding=1),
            nn.BatchNorm2d(encoder_channels[0] // 2), nn.ReLU(inplace=True)
        )
        
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[0] // 2, encoder_channels[0] // 4, 2, 2),
            nn.BatchNorm2d(encoder_channels[0] // 4), nn.ReLU(inplace=True),
            nn.Conv2d(encoder_channels[0] // 4, encoder_channels[0] // 4, 3, padding=1),
            nn.BatchNorm2d(encoder_channels[0] // 4), nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Conv2d(encoder_channels[0] // 4, classes, 1)
        
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        else:
            self.activation = None
    
    def forward(self, x):
        features = self.encoder(x)
        x = features[3]
        x = self.decoder1(x)
        x = self.decoder1_conv(x)
        x = self.decoder2(x)
        x = self.decoder2_conv(x)
        x = self.decoder3(x)
        x = self.decoder3_conv(x)
        x = self.decoder4(x)
        x = self.decoder5(x)
        x = self.final_conv(x)
        if self.activation:
            x = self.activation(x)
        return x


# ===================== COCO  =====================
class COCODataset:
    """COCO Dataset
    
    : COCO  'single layer' ClassSingle Layer+Multilayer+Contaminant
    (,MultilayerContaminant)
    Foreground = COCO 'single layer' Class = 
    """
    
    def __init__(self, image_dir, annotation_file):
        self.image_dir = image_dir
        
        #  COCO 
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.coco_data = json.load(f)
        
        #  ID 
        self.image_id_to_annotations = {}
        self.image_id_to_info = {}
        
        for img in self.coco_data['images']:
            self.image_id_to_info[img['id']] = img
        
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append(ann)
        
        #  ID
        self.valid_image_ids = []
        for img_id in self.image_id_to_annotations.keys():
            img_info = self.image_id_to_info[img_id]
            img_path = os.path.join(self.image_dir, img_info['file_name'])
            if os.path.exists(img_path):
                self.valid_image_ids.append(img_id)
        
        self.valid_image_ids.sort()
        print(f"Loaded {len(self.valid_image_ids)} annotated images")
    
    def get_foreground_mask(self, image_id, target_size=(512, 512)):
        """Foreground
        
        Foreground = COCO 'single layer' Class (Single Layer+Multilayer+Contaminant)
        """
        img_info = self.image_id_to_info[image_id]
        h, w = img_info['height'], img_info['width']
        
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if image_id in self.image_id_to_annotations:
            for ann in self.image_id_to_annotations[image_id]:
                category_name = self._get_category_name(ann['category_id'])
                
                #  single layer Class (MultilayerContaminant)
                if category_name != 'single layer':
                    continue
                
                #  segmentation  mask
                if 'segmentation' in ann and ann['segmentation']:
                    for seg in ann['segmentation']:
                        if len(seg) >= 6:
                            poly_coords = [(int(seg[i]), int(seg[i+1])) for i in range(0, len(seg), 2)]
                            temp_mask = Image.new('L', (w, h), 0)
                            draw = ImageDraw.Draw(temp_mask)
                            draw.polygon(poly_coords, outline=1, fill=1)
                            temp_mask_array = np.array(temp_mask)
                            mask = np.maximum(mask, temp_mask_array)
        
        # 
        if mask.shape[:2] != target_size:
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        
        return mask
    
    def get_pure_single_layer_mask(self, image_id, target_size=(512, 512)):
        """Single Layer (Foreground - Multilayer - Contaminant)
        
        MultilayerContaminantSingle Layer
        """
        img_info = self.image_id_to_info[image_id]
        h, w = img_info['height'], img_info['width']
        
        fg_mask = np.zeros((h, w), dtype=np.uint8)
        ml_mask = np.zeros((h, w), dtype=np.uint8)
        imp_mask = np.zeros((h, w), dtype=np.uint8)
        
        if image_id in self.image_id_to_annotations:
            for ann in self.image_id_to_annotations[image_id]:
                category_name = self._get_category_name(ann['category_id'])
                
                if 'segmentation' not in ann or not ann['segmentation']:
                    continue
                
                #  segmentation  mask
                for seg in ann['segmentation']:
                    if len(seg) >= 6:
                        poly_coords = [(int(seg[i]), int(seg[i+1])) for i in range(0, len(seg), 2)]
                        temp_mask = Image.new('L', (w, h), 0)
                        draw = ImageDraw.Draw(temp_mask)
                        draw.polygon(poly_coords, outline=1, fill=1)
                        temp_mask_array = np.array(temp_mask)
                        
                        if category_name == 'single layer':
                            fg_mask = np.maximum(fg_mask, temp_mask_array)
                        elif category_name == 'multilayer':
                            ml_mask = np.maximum(ml_mask, temp_mask_array)
                        elif category_name == 'contaminants':
                            imp_mask = np.maximum(imp_mask, temp_mask_array)
        
        # Single Layer = Foreground - Multilayer - Contaminant
        pure_single = fg_mask.astype(bool) & ~ml_mask.astype(bool) & ~imp_mask.astype(bool)
        pure_single = pure_single.astype(np.uint8)
        
        # 
        if pure_single.shape[:2] != target_size:
            pure_single = cv2.resize(pure_single, target_size, interpolation=cv2.INTER_NEAREST)
        
        return pure_single
    
    def _get_category_name(self, category_id):
        for cat in self.coco_data['categories']:
            if cat['id'] == category_id:
                return cat['name']
        return ''
    
    def __len__(self):
        return len(self.valid_image_ids)
    
    def __getitem__(self, idx):
        image_id = self.valid_image_ids[idx]
        img_info = self.image_id_to_info[image_id]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        
        image = np.array(Image.open(img_path).convert('RGB'))
        if image.shape[:2] != (512, 512):
            image = cv2.resize(image, (512, 512))
        
        return image, img_path


# ===================== Configuration =====================
def get_paths():
    """Model"""
    paths = {
        # Model ( inference_package)
        'stage1_model': os.path.join(BASE_DIR, 'convnext_tiny_stage1_wo_cont_results', 'best_convnext_tiny_stage1.pth'),
        'stage2_model': os.path.join(BASE_DIR, 'convnext_tiny_stage2_wo_cont_results', 'best_convnext_tiny_stage2.pth'),
        
        # COCO  ( training_and_contrast_package)
        'coco_annotation': os.path.join(PROJECT_ROOT, 'image datasets', 'annotation COCO dataset', 'instances_val2019.json'),
        'coco_image_dir': os.path.join(PROJECT_ROOT, 'image datasets', 'validation dataset'),
    }
    return paths


def get_transform():
    """Get image preprocessing transform"""
    return A.Compose([
        A.Resize(height=512, width=512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def load_model(model_path, classes=1, activation=None, device='cpu'):
    """Load ablation model checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Model: {model_path}")

    model = ConvNeXtUNet(
        encoder_name='convnext_tiny.fb_in1k',
        in_channels=3,
        classes=classes,
        activation=activation
    )


    checkpoint = torch.load(model_path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    print("Model loaded.")
    return model



def calculate_metrics(gt_mask, pred_mask):
    """Metrics: IoU, Precision, Recall, F1"""
    gt_flat = gt_mask.flatten()
    pred_flat = pred_mask.flatten()
    
    tp = np.sum((gt_flat == 1) & (pred_flat == 1))
    fp = np.sum((gt_flat == 0) & (pred_flat == 1))
    fn = np.sum((gt_flat == 1) & (pred_flat == 0))
    
    intersection = tp
    union = tp + fp + fn
    
    iou = intersection / union if union > 0 else 0
    dice = 2 * intersection / (2 * intersection + fp + fn) if (2 * intersection + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn)
    }


def evaluate_stage1(model, dataset, device, thresholds):
    """
    Evaluate Stage 1 foreground segmentation.

    TTA:
        original
        horizontal flip
        vertical flip
        horizontal + vertical flip
    """
    print("\n" + "=" * 60)
    print("Stage 1 Ablation Evaluation")
    print("=" * 60)

    transform = get_transform()
    results = []

    model.eval()

    for threshold in thresholds:
        all_metrics = []

        with torch.no_grad():
            for idx in tqdm(range(len(dataset)), desc=f"threshold={threshold:.2f}"):
                image, _ = dataset[idx]

                image_id = dataset.valid_image_ids[idx]
                gt_mask = dataset.get_foreground_mask(
                    image_id,
                    target_size=(512, 512)
                )

                transformed = transform(image=image)
                image_tensor = transformed['image'].unsqueeze(0).to(device)

                # Original
                output = model(image_tensor)

                pred_prob = torch.sigmoid(
                    output
                ).squeeze().cpu().numpy()

                pred_mask = (
                    pred_prob > threshold
                ).astype(np.uint8)


                metrics = calculate_metrics(gt_mask, pred_mask)
                all_metrics.append(metrics)

        if all_metrics:
            avg_metrics = {
                'threshold': threshold,
                'iou': np.mean([m['iou'] for m in all_metrics]),
                'dice': np.mean([m['dice'] for m in all_metrics]),
                'precision': np.mean([m['precision'] for m in all_metrics]),
                'recall': np.mean([m['recall'] for m in all_metrics]),
                'f1': np.mean([m['f1'] for m in all_metrics]),
                'num_samples': len(all_metrics),
            }

            results.append(avg_metrics)

            print(
                f"threshold={threshold:.2f}: "
                f"IoU={avg_metrics['iou']:.4f}, "
                f"Dice={avg_metrics['dice']:.4f}, "
                f"P={avg_metrics['precision']:.4f}, "
                f"R={avg_metrics['recall']:.4f}, "
                f"F1={avg_metrics['f1']:.4f}"
            )

    return results


def evaluate_stage2(model, dataset, device, thresholds, imp_thresholds):
    """
    Evaluate Stage 2 threshold (4-classSingle Layer)
    GT: COCO 'single layer' Class 
    : Stage 2 Single Layer (class 1),Foreground GT 
    """
    print("\n" + "="*60)
    print("Stage 2: 4-classSingle Layer")
    print("="*60)
    
    transform = get_transform()
    results = []
    
    for threshold in thresholds:
        for imp_t in imp_thresholds:
            all_metrics = []
            
            for idx in tqdm(range(len(dataset)), desc=f"th={threshold:.2f}, imp={imp_t:.2f}"):
                image, img_path = dataset[idx]
                
                #  GT Foreground
                image_id = dataset.valid_image_ids[idx]
                gt_mask = dataset.get_foreground_mask(image_id, target_size=(512, 512))
                
                # 
                transformed = transform(image=image)
                image_tensor = transformed['image'].unsqueeze(0).to(device)
                
                # 
                with torch.no_grad():
                    output = model(image_tensor)
                    probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
                    pred_class = np.argmax(probs, axis=0).astype(np.uint8)
                    
                    # Contaminant
                    p_impurity = probs[2]
                    abs_pass = p_impurity > imp_t
                    rel_pass = (p_impurity - probs[0]) > 0.15
                    keep_impurity = abs_pass & rel_pass
                    pred_class[(pred_class == 2) & ~keep_impurity] = 0
                    
                    # Single Layer (class 1)
                    pred_mask = (pred_class == 1).astype(np.uint8)
                
                # Metrics
                metrics = calculate_metrics(gt_mask, pred_mask)
                all_metrics.append(metrics)
            
            if all_metrics:
                avg_metrics = {
                    'threshold': threshold,
                    'imp_threshold': imp_t,
                    'iou': np.mean([m['iou'] for m in all_metrics]),
                    'dice': np.mean([m['dice'] for m in all_metrics]),
                    'precision': np.mean([m['precision'] for m in all_metrics]),
                    'recall': np.mean([m['recall'] for m in all_metrics]),
                    'f1': np.mean([m['f1'] for m in all_metrics]),
                    'num_samples': len(all_metrics)
                }
                results.append(avg_metrics)
                print(f"threshold={threshold:.2f}, Contaminantthreshold={imp_t:.2f}: "
                      f"IoU={avg_metrics['iou']:.4f}, P={avg_metrics['precision']:.4f}, "
                      f"R={avg_metrics['recall']:.4f}, F1={avg_metrics['f1']:.4f}")
    
    return results


def evaluate_stage2_pure_single(stage1_model, stage2_model, dataset, device, thresholds, multi_thresholds, imp_thresholds):
    """
    Evaluate Stage 2 Single Layer
    
    Pipeline:
    1. Stage 1 Foreground
    2. Stage 2 ForegroundContaminant(class 2)Multilayer(class 3)
    3. : Stage1Foreground - Stage2Multilayer - Stage2Contaminant
    4. GT: COCO Foreground - COCO Multilayer - COCO Contaminant = Single Layer
    
    Evaluate Stage 2 Multilayer/Contaminant
    """
    print("\n" + "="*60)
    print("Stage 2: Single LayerEvaluate")
    print("Pipeline: Stage1Foreground - Stage2Multilayer - Stage2Contaminant")
    print("GT: COCOForeground - COCOMultilayer - COCOContaminant (Single Layer)")
    print("="*60)
    
    transform = get_transform()
    stage1_thresh = 0.5
    results = []
    
    for threshold in thresholds:
        for multi_t in multi_thresholds:
            for imp_t in imp_thresholds:
                all_metrics = []
                
                for idx in tqdm(range(len(dataset)), desc=f": "):
                    image, img_path = dataset[idx]
                    
                    #  COCO Single Layer GT
                    image_id = dataset.valid_image_ids[idx]
                    gt_pure_single = dataset.get_pure_single_layer_mask(image_id, target_size=(512, 512))
                    
                    # 
                    transformed = transform(image=image)
                    image_tensor = transformed['image'].unsqueeze(0).to(device)
                    
                    # Stage 1 
                    with torch.no_grad():
                        stage1_output = stage1_model(image_tensor)
                        stage1_prob = torch.sigmoid(stage1_output).squeeze().cpu().numpy()
                        stage1_fg = (stage1_prob > stage1_thresh).astype(np.uint8)
                    
                    # Stage 2 
                    with torch.no_grad():
                        stage2_output = stage2_model(image_tensor)
                        probs = torch.softmax(stage2_output, dim=1).squeeze().cpu().numpy()
                    
                    # Class
                    p_bg = probs[0]
                    p_single = probs[1]
                    p_cont = probs[2]
                    p_multi = probs[3]
                    
                    # segmentation threshold:  p_single > threshold Single Layer
                    #  argmax,Single LayerBackground
                    pred_class = np.zeros_like(p_single, dtype=np.uint8)
                    
                    # Single Layer > thresholdSingle Layer
                    single_mask = p_single > threshold
                    
                    # Multilayerthreshold:  p_multi > multi_t Multilayer
                    multi_mask = (p_multi > multi_t)
                    
                    # Contaminant (threshold + threshold)
                    abs_pass = p_cont > imp_t
                    rel_pass = (p_cont - p_single) > 0.15
                    cont_mask = multi_mask & abs_pass & rel_pass
                    
                    # : Contaminant > Multilayer > Single Layer > Background
                    pred_class[cont_mask] = 2  # Contaminant
                    pred_class[multi_mask & ~cont_mask] = 3  # Multilayer
                    pred_class[single_mask] = 1  # Single Layer (threshold)
                    # Background (class 0)
                    
                    # Contaminant(class 2)Multilayer(class 3)
                    pred_multi = (pred_class == 3).astype(np.uint8)
                    pred_cont = (pred_class == 2).astype(np.uint8)
                    
                    #  Stage 1 Foreground
                    pred_multi_in_fg = stage1_fg & pred_multi
                    pred_cont_in_fg = stage1_fg & pred_cont
                    
                    # Stage 2 Single Layer = Stage1Foreground - Multilayer - Contaminant
                    pred_pure_single = stage1_fg.astype(bool) & ~pred_multi_in_fg.astype(bool) & ~pred_cont_in_fg.astype(bool)
                    pred_pure_single = pred_pure_single.astype(np.uint8)
                
                    # Metrics
                    metrics = calculate_metrics(gt_pure_single, pred_pure_single)
                    all_metrics.append(metrics)
                
                if all_metrics:
                    avg_metrics = {
                        'threshold': threshold,
                        'multi_threshold': multi_t,
                        'imp_threshold': imp_t,
                        'iou': np.mean([m['iou'] for m in all_metrics]),
                        'dice': np.mean([m['dice'] for m in all_metrics]),
                        'precision': np.mean([m['precision'] for m in all_metrics]),
                        'recall': np.mean([m['recall'] for m in all_metrics]),
                        'f1': np.mean([m['f1'] for m in all_metrics]),
                        'num_samples': len(all_metrics)
                    }
                    results.append(avg_metrics)
                    print(f"IoU={avg_metrics['iou']:.4f}, P={avg_metrics['precision']:.4f}, "
                          f"R={avg_metrics['recall']:.4f}, F1={avg_metrics['f1']:.4f}")
    
    return results



def save_results(stage1_results, stage2_results, output_dir):
    """ evaluation_results.txt()"""
    out_path = os.path.join(output_dir, 'evaluation_results.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ConvNeXt threshold (wo_cont)\n")
        f.write("=" * 80 + "\n\n")
        f.write("GT: COCO 'single layer' Class (Single Layer+Multilayer+Contaminant,)\n\n")

        if stage1_results:
            f.write("[Stage 1]\n")
            f.write(f"{'threshold':<10} {'IoU':<10} {'Dice':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}\n")
            f.write("-" * 70 + "\n")
            for r in stage1_results:
                f.write(f"{r['threshold']:<10.2f} {r['iou']:<10.4f} {r['dice']:<10.4f} "
                        f"{r['precision']:<10.4f} {r['recall']:<10.4f} {r['f1']:<10.4f}\n")
            best_s1 = max(stage1_results, key=lambda x: x['iou'])
            f.write(f"\nStage1: threshold={best_s1['threshold']:.2f}, IoU={best_s1['iou']:.4f}, F1={best_s1['f1']:.4f}\n\n")

        if stage2_results:
            f.write("[Stage 2]\n")
            f.write(f"{'segmentation threshold':<10} {'Contaminantthreshold':<10} {'IoU':<10} {'Dice':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}\n")
            f.write("-" * 80 + "\n")
            for r in stage2_results:
                f.write(f"{r['threshold']:<10.2f} {r['imp_threshold']:<10.2f} {r['iou']:<10.4f} "
                        f"{r['dice']:<10.4f} {r['precision']:<10.4f} {r['recall']:<10.4f} {r['f1']:<10.4f}\n")
            best_s2 = max(stage2_results, key=lambda x: x['iou'])
            f.write(
                f"\nStage2: segmentation threshold={best_s2['threshold']:.2f}, Contaminantthreshold={best_s2['imp_threshold']:.2f}, "
                f"IoU={best_s2['iou']:.4f}, F1={best_s2['f1']:.4f}\n\n"
            )

    print(f"Results saved: {out_path}")


def save_results_pure_single(stage2_pure_results, output_dir):
    """ Stage 2 Single Layer()"""
    if not stage2_pure_results:
        return

    s2_path = os.path.join(output_dir, f'evaluation_results.txt')

    with open(s2_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Stage 2: Single LayerEvaluate\n")
        f.write("Pipeline: Stage1Foreground - Stage2Multilayer - Stage2Contaminant\n")
        f.write("GT: COCOForeground - COCOMultilayer - COCOContaminant (Single Layer)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Single Layer':<8} {'Multilayer':<8} {'Contaminant':<8} {'IoU':<10} {'Dice':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}\n")
        f.write("-" * 80 + "\n")
        for r in stage2_pure_results:
            f.write(
                f"{r['threshold']:<8.2f} {r['multi_threshold']:<8.2f} {r['imp_threshold']:<8.2f} "
                f"{r['iou']:<10.4f} {r['dice']:<10.4f} {r['precision']:<10.4f} {r['recall']:<10.4f} {r['f1']:<10.4f}\n"
            )

    print(f"Stage 2 Single LayerResults saved: {s2_path}")

def main():
    parser = argparse.ArgumentParser(description='ConvNeXt threshold sensitivity analysis (wo_cont)')
    parser.add_argument('--stage1_only', action='store_true', help='Only evaluate Stage 1')
    parser.add_argument('--stage2_only', action='store_true', help='Only evaluate Stage 2 (Foreground)')
    parser.add_argument('--stage2_pure', action='store_true', help='Evaluate Stage 2 Single Layer (Stage1Foreground - Stage2MultilayerContaminant)')
    parser.add_argument('--thresholds', type=str, default='0.5',
                       help='Single Layersegmentation threshold (default: 0.5)')
    parser.add_argument('--multi_thresholds', type=str, default='0.5',
                       help='Multilayersegmentation threshold (default: 0.5)')
    parser.add_argument('--imp_thresholds', type=str, default='0.5',
                       help='Contaminantsegmentation threshold (default: 0.5)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: output/threshold_analysis)')
    args = parser.parse_args()
    # This split entry point evaluates Stage 1 by default.
    if not args.stage1_only and not args.stage2_only and not args.stage2_pure:
        args.stage1_only = True
    
    # threshold
    thresholds = [float(t) for t in args.thresholds.split(',')]
    multi_thresholds = [float(t) for t in args.multi_thresholds.split(',')]
    imp_thresholds = [float(t) for t in args.imp_thresholds.split(',')]
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Get paths
    paths = get_paths()
    
    # Checking COCO data
    if not os.path.exists(paths['coco_annotation']):
        print(f"Error: COCO Annotation file not found: {paths['coco_annotation']}")
        return
    
    if not os.path.exists(paths['coco_image_dir']):
        print(f"Error: COCO Image directory not found: {paths['coco_image_dir']}")
        return
    
    #  COCO Dataset
    print("\n COCO Dataset...")
    print("GT: COCO 'single layer' Class (,Single Layer+Multilayer+Contaminant)")
    coco_dataset = COCODataset(
        image_dir=paths['coco_image_dir'],
        annotation_file=paths['coco_annotation']
    )
    
    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        if args.stage1_only and not args.stage2_only and not args.stage2_pure:
            output_dir = os.path.join(BASE_DIR, 'convnext_tiny_stage1_wo_cont_results')
        else:
            # stage2_only  stage2_pure(default) stage2 
            output_dir = os.path.join(BASE_DIR, 'convnext_tiny_stage2_wo_cont_results')
    os.makedirs(output_dir, exist_ok=True)
    
    stage1_results = []
    stage2_results = []
    stage2_pure_results = []
    
    # Evaluate Stage 2 Single Layer
    if args.stage2_pure:
        try:
            stage1_model = load_model(paths['stage1_model'], classes=1, activation=None, device=device)
            stage2_model = load_model(paths['stage2_model'], classes=4, activation=None, device=device)
            stage2_pure_results = evaluate_stage2_pure_single(
                stage1_model, stage2_model, coco_dataset, device, thresholds, multi_thresholds, imp_thresholds
            )
        except Exception as e:
            print(f"Stage 2 Single LayerEvaluate: {e}")
            import traceback
            traceback.print_exc()
    
    # Evaluate Stage 1 (Foreground)
    elif not args.stage2_only:
        try:
            stage1_model = load_model(paths['stage1_model'], classes=1, activation=None, device=device)
            stage1_results = evaluate_stage1(stage1_model, coco_dataset, device, thresholds)
        except Exception as e:
            print(f"Stage 1 Evaluate: {e}")
            import traceback
            traceback.print_exc()
    
    # Evaluate Stage 2 (Foreground)
    if args.stage2_only:
        try:
            stage2_model = load_model(paths['stage2_model'], classes=4, activation=None, device=device)
            stage2_results = evaluate_stage2(stage2_model, coco_dataset, device, thresholds, imp_thresholds)
        except Exception as e:
            print(f"Stage 2 Evaluate: {e}")
            import traceback
            traceback.print_exc()
    
    # 
    if stage1_results or stage2_results or stage2_pure_results:
        if stage2_pure_results:
            save_results_pure_single(stage2_pure_results, output_dir)
        else:
            save_results(stage1_results, stage2_results, output_dir)
    else:
        print("No available results")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()






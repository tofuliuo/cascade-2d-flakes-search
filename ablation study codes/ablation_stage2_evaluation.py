"""
Stage 2 no-skip ablation evaluated in the complete Stage 1-Stage 2 cascade.

Usage:
    python ablation_stage2_evaluation.py
"""

import os
import json
import argparse

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


class ConvNeXtUNetWithSkips(ConvNeXtUNet):
    """Normal Stage 1 architecture used while evaluating the Stage 2 ablation."""

    def __init__(self, encoder_name='convnext_tiny', in_channels=3, classes=1, activation=None):
        super().__init__(encoder_name, in_channels, classes, activation)
        channels = self.encoder_channels
        self.decoder1_conv[0] = nn.Conv2d(channels[2] * 2, channels[2], 3, padding=1)
        self.decoder2_conv[0] = nn.Conv2d(channels[1] * 2, channels[1], 3, padding=1)
        self.decoder3_conv[0] = nn.Conv2d(channels[0] * 2, channels[0], 3, padding=1)

    def forward(self, x):
        features = self.encoder(x)
        x = self.decoder1(features[3])
        x = self.decoder1_conv(torch.cat([x, features[2]], dim=1))
        x = self.decoder2(x)
        x = self.decoder2_conv(torch.cat([x, features[1]], dim=1))
        x = self.decoder3(x)
        x = self.decoder3_conv(torch.cat([x, features[0]], dim=1))
        x = self.decoder4(x)
        x = self.decoder5(x)
        x = self.final_conv(x)
        return self.activation(x) if self.activation else x


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
        target_h, target_w = target_size
        fg_mask = np.zeros((target_h, target_w), dtype=np.uint8)
        ml_mask = np.zeros((target_h, target_w), dtype=np.uint8)
        imp_mask = np.zeros((target_h, target_w), dtype=np.uint8)
        
        if image_id in self.image_id_to_annotations:
            for ann in self.image_id_to_annotations[image_id]:
                category_name = self._get_category_name(ann['category_id'])
                
                if 'segmentation' not in ann or not ann['segmentation']:
                    continue
                
                #  segmentation  mask
                for seg in ann['segmentation']:
                    if len(seg) >= 6:
                        poly_coords = [
                            (
                                int(seg[i] * target_w / img_info['width']),
                                int(seg[i + 1] * target_h / img_info['height'])
                            )
                            for i in range(0, len(seg), 2)
                        ]
                        temp_mask = Image.new('L', (target_w, target_h), 0)
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
        'stage1_model': os.path.join(PROJECT_ROOT, 'stage1 (binary classification) training codes', 'convnext_tiny_stage1_results', 'best_convnext_tiny_stage1.pth'),
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


def load_model(model_path, classes=1, activation=None, device='cpu', with_skips=False):
    """Model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model: {model_path}")
    
    print(f"Model: {model_path}")
    
    model_class = ConvNeXtUNetWithSkips if with_skips else ConvNeXtUNet
    model = model_class(
        encoder_name='convnext_tiny.fb_in1k',
        in_channels=3,
        classes=classes,
        activation=activation
    )

    
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    
    print("Model")
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


def evaluate_stage2_pure_single(stage1_model, stage2_model, dataset, device, stage1_threshold=0.5):
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
    all_metrics = []
    with torch.inference_mode():
        for idx in tqdm(range(len(dataset)), desc="Evaluating cascade"):
            image, _ = dataset[idx]
            image_id = dataset.valid_image_ids[idx]
            gt_pure_single = dataset.get_pure_single_layer_mask(
                image_id, target_size=(512, 512)
            )
            image_tensor = transform(image=image)['image'].unsqueeze(0).to(device)

            stage1_logits = stage1_model(image_tensor)
            stage1_fg = (torch.sigmoid(stage1_logits) > stage1_threshold).squeeze().cpu().numpy()
            stage2_logits = stage2_model(image_tensor)
            stage2_class = torch.argmax(stage2_logits, dim=1).squeeze().cpu().numpy()
            pred_pure_single = (
                stage1_fg & (stage2_class != 2) & (stage2_class != 3)
            ).astype(np.uint8)
            all_metrics.append(calculate_metrics(gt_pure_single, pred_pure_single))

    return [{
        'stage1_threshold': stage1_threshold,
        'iou': np.mean([m['iou'] for m in all_metrics]),
        'dice': np.mean([m['dice'] for m in all_metrics]),
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics]),
        'f1': np.mean([m['f1'] for m in all_metrics]),
        'num_samples': len(all_metrics),
    }]



def save_results_pure_single(stage2_pure_results, output_dir):
    """ Stage 2 Single Layer()"""
    if not stage2_pure_results:
        return

    s2_path = os.path.join(output_dir, f'evaluation_results.txt')

    with open(s2_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Stage 1-Stage 2 cascade: Single Layer evaluation\n")
        f.write("Pipeline: Stage 1 Foreground - Stage 2 Multilayer - Stage 2 Contaminant\n")
        f.write("=" * 80 + "\n\n")
        r = stage2_pure_results[0]
        f.write(
            f"Single Layer: IoU={r['iou']:.4f}, Dice={r['dice']:.4f}, "
            f"P={r['precision']:.4f}, R={r['recall']:.4f}, F1={r['f1']:.4f}\n"
        )

    print(f"Stage 2 Single LayerResults saved: {s2_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate the no-skip Stage 1-Stage 2 cascade')
    parser.add_argument('--stage1_threshold', type=float, default=0.5,
                        help='Stage 1 foreground threshold (default: 0.5)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: output/threshold_analysis)')
    args = parser.parse_args()
    
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
        output_dir = os.path.join(BASE_DIR, 'convnext_tiny_stage2_wo_cont_results')
    os.makedirs(output_dir, exist_ok=True)
    
    stage1_model = load_model(
        paths['stage1_model'], classes=1, activation=None,
        device=device, with_skips=True
    )
    stage2_model = load_model(paths['stage2_model'], classes=4, activation=None, device=device)
    results = evaluate_stage2_pure_single(
        stage1_model, stage2_model, coco_dataset, device, args.stage1_threshold
    )
    save_results_pure_single(results, output_dir)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()







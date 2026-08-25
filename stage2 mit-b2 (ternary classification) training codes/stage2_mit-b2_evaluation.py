"""
MiT Stage 2 pure single-layer extraction evaluation script
Supports MiT-B0 and MiT-B2
Pipeline:
    1. Stage 1 Foreground
    2. Stage 2 ForegroundMultilayer(class 2)Contaminant(class 3)
    3. : Stage1Foreground - Stage2Multilayer - Stage2Contaminant
    4. GT: COCO Foreground - COCO Multilayer - COCO Contaminant = Single Layer

GT: COCO  (instances_val2019.json)
Metrics: IoU, Dice, Precision, Recall, F1

Usage:
    python evaluate_mit_stage2.py --mit_size b0
    python evaluate_mit_stage2.py --mit_size b2
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
os.environ['HF_HOME'] = BASE_DIR

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

import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    import timm
    import segmentation_models_pytorch as smp
    TIMM_AVAILABLE = True
except ImportError:
    print("Warning: timmsmp")
    TIMM_AVAILABLE = False


# ===================== MiT + U-Net Model =====================
class MiTUNet(nn.Module):
    """MiT + U-Net"""

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

        # 1: 16x16 -> 32x32
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

        # 2: 32x32 -> 64x64
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

        # 3: 64x64 -> 128x128
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

        # 4: 128x128 -> 256x256
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[0], encoder_channels[0] // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(encoder_channels[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(encoder_channels[0] // 2, encoder_channels[0] // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_channels[0] // 2),
            nn.ReLU(inplace=True)
        )

        # 5: 256x256 -> 512x512
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

        x = features[3]  # 16x16
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


# ===================== Dataset =====================
class Stage2MulticlassDataset:
    """Stage 2 4-classDataset"""

    def __init__(self, image_dir, annotation_file):
        self.image_dir = image_dir

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

        # Single Layermask
        single_layer_clean = single_layer_mask.copy()
        single_layer_clean[(multilayer_mask > 0) | (contaminants_mask > 0)] = 0

        # 4-class: 0=Background, 1=Single Layer, 2=Contaminant, 3=Multilayer
        gt_mask_4class = np.zeros((512, 512), dtype=np.uint8)
        gt_mask_4class = np.where(multilayer_mask > 0, 3, gt_mask_4class)
        gt_mask_4class = np.where(contaminants_mask > 0, 2, gt_mask_4class)
        gt_mask_4class = np.where(single_layer_clean > 0, 1, gt_mask_4class)

        if image.shape[:2] != (512, 512):
            image = cv2.resize(image, (512, 512))

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        gt_mask_4class = torch.from_numpy(gt_mask_4class).unsqueeze(0).long()

        return image, gt_mask_4class, image_id


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


def stage1_minus_stage2_to_single_layer(stage1_pred, stage2_pred, stage1_threshold=0.5):
    """
    Stage1Foreground - Stage2MultilayerContaminant = Single Layer
    Stage1: binary classification (Foreground)
    Stage2: 4-class (0=Background, 1=Single Layer, 2=Contaminant, 3=Multilayer)
    """
    # Stage1Foreground
    if stage1_pred.max() <= 1.0:
        stage1_binary = (stage1_pred > stage1_threshold).astype(np.uint8)
    else:
        stage1_binary = (stage1_pred > 127).astype(np.uint8)

    # stage2MultilayerContaminantmask
    stage2_multilayer = (stage2_pred == 3).astype(np.uint8)
    stage2_contaminant = (stage2_pred == 2).astype(np.uint8)

    # Single Layer = stage1Foreground - Multilayer - Contaminant
    single_layer = stage1_binary.copy().astype(np.float32)
    single_layer = single_layer - stage2_multilayer.astype(np.float32)
    single_layer = single_layer - stage2_contaminant.astype(np.float32)
    single_layer = (single_layer > 0).astype(np.uint8)

    # 4-class
    pred_4class = np.zeros_like(stage2_pred)
    pred_4class[stage2_pred == 3] = 3
    pred_4class[stage2_pred == 2] = 2
    pred_4class[single_layer > 0] = 1

    return pred_4class


def evaluate_two_stage_method(stage1_model, stage2_model, dataset, device, num_classes=4, stage1_threshold=0.5):
    """stage"""
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
            image, gt_mask_4class, image_id = dataset[i]

            # Stage1 (Foreground)
            image_tensor = image.unsqueeze(0).to(device)
            stage1_output = stage1_model(image_tensor)
            stage1_pred = torch.sigmoid(stage1_output).squeeze().cpu().numpy()

            # Stage2 (4-class)
            stage2_output = stage2_model(image_tensor)
            stage2_pred = torch.argmax(stage2_output, dim=1).squeeze().cpu().numpy()

            # stage
            pred_mask_4class = stage1_minus_stage2_to_single_layer(
                stage1_pred, stage2_pred, stage1_threshold
            )

            gt_mask_4class_np = gt_mask_4class.squeeze().numpy()

            # 4-classMetrics
            ious, dices, precisions, recalls, f1s = compute_metrics(
                pred_mask_4class, gt_mask_4class_np, num_classes
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
    parser = argparse.ArgumentParser(description='MiT Stage1Foreground - Stage2Multilayer/Contaminant = Single Layer ')
    parser.add_argument('--stage1_checkpoint', type=str, default=None,
                       help='Stage1 ForegroundModel (defaultmit_size)')
    parser.add_argument('--stage2_checkpoint', type=str, default=None,
                       help='Stage2 4-classModel (defaultmit_size)')
    parser.add_argument('--mit_size', type=str, default='b2', choices=['b0', 'b2'],
                       help='MiT (default: b0)')
    parser.add_argument('--stage1_threshold', type=float, default=0.5, help='Stage1 Foregroundthreshold')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Result directory (default: model-specific cascade results directory)')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    #  mit_size default
    if args.mit_size == 'b0':
        default_stage1 = 'mit_b0_stage1_results/best_mit_b0_stage1.pth'
        default_stage2 = 'mit_b0_stage2_results/best_mit_b0_stage2.pth'
    else:  # b2
        default_stage1 = 'mit_b2_stage1_results/best_mit_b2_stage1.pth'
        default_stage2 = 'mit_b2_stage2_results/best_mit_b2_stage2.pth'

    stage1_path = os.path.join(base_dir, args.stage1_checkpoint) if args.stage1_checkpoint else os.path.join(PROJECT_ROOT, 'stage1 mit-b2 (binary classification) training codes', default_stage1)
    stage2_path = os.path.join(base_dir, args.stage2_checkpoint) if args.stage2_checkpoint else os.path.join(base_dir, default_stage2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # MiT
    if args.mit_size == 'b0':
        encoder_name = 'mit_b0'
    else:
        encoder_name = 'mit_b2'

    num_classes = 4  # 4-class: 0=Background, 1=Single Layer, 2=Contaminant, 3=Multilayer
    evaluation_title = (
        f"MiT-{args.mit_size.upper()}: Stage 1 Foreground - "
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
    stage1_model = MiTUNet(
        encoder_name=encoder_name,
        in_channels=3,
        classes=1,
        activation=None
    )
    stage1_checkpoint = torch.load(stage1_path, map_location=device)
    if 'state_dict' in stage1_checkpoint:
        stage1_model.load_state_dict(stage1_checkpoint['state_dict'], strict=True)
    else:
        stage1_model.load_state_dict(stage1_checkpoint, strict=True)
    stage1_model = stage1_model.to(device)
    stage1_model.eval()
    print("Stage 1 model loaded.")

    # Stage2Model (4-class)
    print("\nLoading Stage 2 model...")
    stage2_model = MiTUNet(
        encoder_name=encoder_name,
        in_channels=3,
        classes=4,
        activation=None
    )
    stage2_checkpoint = torch.load(stage2_path, map_location=device)
    if 'state_dict' in stage2_checkpoint:
        stage2_model.load_state_dict(stage2_checkpoint['state_dict'], strict=True)
    else:
        stage2_model.load_state_dict(stage2_checkpoint, strict=True)
    stage2_model = stage2_model.to(device)
    stage2_model.eval()
    print("Stage 2 model loaded.")

    # Dataset
    print("\nLoading validation set...")
    dataset_root = os.path.join(PROJECT_ROOT, "image datasets")
    val_dataset = Stage2MulticlassDataset(
        image_dir=os.path.join(dataset_root, 'validation dataset'),
        annotation_file=os.path.join(dataset_root, 'annotation COCO dataset', 'instances_val2019.json')
    )
    print(f"Validation set: {len(val_dataset)} images")

    # 
    results = evaluate_two_stage_method(
        stage1_model, stage2_model, val_dataset, device,
        num_classes=4, stage1_threshold=args.stage1_threshold
    )

    # 
    print("\n" + "=" * 60)
    print(f"({evaluation_title})")
    print("=" * 60)
    print()
    print(f"Single Layer: IoU={results['iou']:.4f}, Dice={results['dice']:.4f}, "
          f"P={results['precision']:.4f}, R={results['recall']:.4f}, F1={results['f1']:.4f}")

    # 
    results_dir = args.output_dir or os.path.join(
        base_dir, f'mit_{args.mit_size}_stage1_minus_stage2_results'
    )
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
# # MiT-B0
# python evaluate_mit_stage1_minus_stage2.py --mit_size b0

# # MiT-B2
# python evaluate_mit_stage1_minus_stage2.py --mit_size b2

"""
ConvNeXt + U-Net evaluation script (Stage 1 binary / Stage 2 4-class)
Evaluation metrics: IoU, Dice, Precision, Recall, F1

Usage:
    python evaluate_convnext.py
    python evaluate_convnext.py --checkpoint convnext_tiny_stage1_results/best_convnext_tiny_stage1.pth --num_classes 1
    python evaluate_convnext.py --checkpoint convnext_tiny_stage2_results/best_convnext_tiny_stage2.pth --num_classes 4
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
import segmentation_models_pytorch as smp
import argparse

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    print("Warning: timm,ConvNeXtModel")
    TIMM_AVAILABLE = False


# ===================== Model =====================
class ConvNeXtUNet(nn.Module):
    """ConvNeXt + U-Net"""

    def __init__(self, encoder_name='convnext_tiny', in_channels=3, classes=4, activation='softmax'):
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError("timm,ConvNeXtModel")

        self.encoder = timm.create_model(encoder_name, pretrained=False, features_only=True, out_indices=(0, 1, 2, 3))

        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, 512, 512)
            features = self.encoder(dummy_input)
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
        features = self.encoder(x)

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


# ===================== Dataset =====================
class Stage1Dataset:
    """Stage 1 Dataset: raw single-layer footprint = foreground"""

    def __init__(self, image_dir, annotation_file):
        self.image_dir = image_dir

        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.coco_data = json.load(f)

        self.image_id_to_info = {
            img['id']: img for img in self.coco_data['images']
        }

        self.image_id_to_annotations = {}
        for ann in self.coco_data['annotations']:
            self.image_id_to_annotations.setdefault(
                ann['image_id'], []
            ).append(ann)

        self.single_layer_id = next(
            cat['id'] for cat in self.coco_data['categories']
            if cat['name'] == 'single layer'
        )

        self.valid_image_ids = sorted(
            self.image_id_to_annotations.keys()
        )

    def __len__(self):
        return len(self.valid_image_ids)

    def __getitem__(self, idx):
        image_id = self.valid_image_ids[idx]
        img_info = self.image_id_to_info[image_id]

        img_path = os.path.join(
            self.image_dir, img_info['file_name']
        )
        image = np.array(Image.open(img_path).convert('RGB'))

        mask = np.zeros((512, 512), dtype=np.uint8)

        for ann in self.image_id_to_annotations.get(image_id, []):
            if ann['category_id'] != self.single_layer_id:
                continue

            if 'segmentation' not in ann or not ann['segmentation']:
                continue

            for seg in ann['segmentation']:
                if len(seg) < 6:
                    continue

                poly_coords = []
                for i in range(0, len(seg), 2):
                    x = int(seg[i] * 512 / img_info['width'])
                    y = int(seg[i + 1] * 512 / img_info['height'])
                    poly_coords.append((x, y))

                from PIL import ImageDraw
                temp_mask = Image.new('L', (512, 512), 0)
                draw = ImageDraw.Draw(temp_mask)
                draw.polygon(poly_coords, outline=1, fill=1)
                mask = np.maximum(mask, np.array(temp_mask))

        if image.shape[:2] != (512, 512):
            image = cv2.resize(image, (512, 512))

        image = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .float() / 255.0
        )

        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(3, 1, 1)
        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)
        image = (image - mean) / std

        mask = torch.from_numpy(mask).unsqueeze(0).long()

        return image, mask, image_id

class Stage2Dataset:
    """Stage 2 Dataset:4-class"""

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

        mask = np.zeros((512, 512), dtype=np.uint8)
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

        single_layer_clean = single_layer_mask.copy()
        single_layer_clean[(multilayer_mask > 0) | (contaminants_mask > 0)] = 0

        mask = np.where(multilayer_mask > 0, 3, mask)
        mask = np.where(contaminants_mask > 0, 2, mask)
        mask = np.where(single_layer_clean > 0, 1, mask)

        if image.shape[:2] != (512, 512):
            image = cv2.resize(image, (512, 512))


        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std

        mask = torch.from_numpy(mask).unsqueeze(0).long()
        return image, mask, image_id


# ===================== Evaluation metrics =====================
def compute_metrics(pred_mask, gt_mask, num_classes):
    """Class IoU, Dice, Precision, Recall, F1"""
    ious, dices, precisions, recalls, f1s = {}, {}, {}, {}, {}

    metric_class_count = 2 if num_classes == 1 else num_classes
    for cls_id in range(metric_class_count):
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
        f1_score = (2 * precision * recall + 1e-6) / (precision + recall + 1e-6)

        ious[cls_id] = float(iou)
        dices[cls_id] = float(dice)
        precisions[cls_id] = float(precision)
        recalls[cls_id] = float(recall)
        f1s[cls_id] = float(f1_score)

    return ious, dices, precisions, recalls, f1s


def evaluate_model(model, dataset, device, num_classes=4, threshold=0.5, use_tta=False):
    """Model"""
    model.eval()

    class_keys = list(range(2 if num_classes == 1 else num_classes))

    all_ious = {i: [] for i in class_keys}
    all_dices = {i: [] for i in class_keys}
    all_precisions = {i: [] for i in class_keys}
    all_recalls = {i: [] for i in class_keys}
    all_f1s = {i: [] for i in class_keys}

    print("\nStart evaluation...")

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Evaluating"):
            image, gt_mask, _ = dataset[i]
            image = image.unsqueeze(0).to(device)

            if num_classes == 1:
                output = model(image)
                if use_tta:
                    output_h = torch.flip(model(torch.flip(image, dims=(3,))), dims=(3,))
                    output_v = torch.flip(model(torch.flip(image, dims=(2,))), dims=(2,))
                    output_hv = torch.flip(model(torch.flip(image, dims=(2, 3))), dims=(2, 3))
                    output = (output + output_h + output_v + output_hv) / 4
                pred_mask = (output.squeeze() > threshold).float().cpu().numpy()
                gt_binary = (gt_mask.squeeze().numpy() > 0).astype(np.float32)
            elif num_classes == 2:
                output = model(image)
                pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
                gt_binary = (gt_mask.squeeze().numpy() > 0).astype(np.float32)
            else:
                output = model(image)
                pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
                gt_binary = gt_mask.squeeze().numpy().astype(np.float32)

            ious, dices, precisions, recalls, f1s = compute_metrics(pred_mask, gt_binary, num_classes)

            for cls_id in ious.keys():
                all_ious[cls_id].append(ious[cls_id])
                all_dices[cls_id].append(dices[cls_id])
                all_precisions[cls_id].append(precisions[cls_id])
                all_recalls[cls_id].append(recalls[cls_id])
                all_f1s[cls_id].append(f1s[cls_id])

    if num_classes in (1, 2):
        reported_class_ids = [0, 1]
    else:
        reported_class_ids = list(range(num_classes))

    class_metrics = {}
    for cls_id in reported_class_ids:
        class_metrics[cls_id] = {
            'iou': np.mean(all_ious[cls_id]) if all_ious[cls_id] else 0,
            'dice': np.mean(all_dices[cls_id]) if all_dices[cls_id] else 0,
            'precision': np.mean(all_precisions[cls_id]) if all_precisions[cls_id] else 0,
            'recall': np.mean(all_recalls[cls_id]) if all_recalls[cls_id] else 0,
            'f1': np.mean(all_f1s[cls_id]) if all_f1s[cls_id] else 0
        }

    valid_classes = [0, 1] if num_classes in (1, 2) else range(1, num_classes)
    mean_iou = np.mean([np.mean(all_ious[i]) for i in valid_classes if all_ious[i]])
    mean_dice = np.mean([np.mean(all_dices[i]) for i in valid_classes if all_dices[i]])
    mean_precision = np.mean([np.mean(all_precisions[i]) for i in valid_classes if all_precisions[i]])
    mean_recall = np.mean([np.mean(all_recalls[i]) for i in valid_classes if all_recalls[i]])
    mean_f1 = np.mean([np.mean(all_f1s[i]) for i in valid_classes if all_f1s[i]])

    return {
        'mean_iou': mean_iou,
        'mean_dice': mean_dice,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'mean_f1': mean_f1,
        'class_metrics': class_metrics
    }


# ===================== Main function =====================
def main():
    parser = argparse.ArgumentParser(description='ConvNeXt + U-Net ')
    parser.add_argument(
        '--checkpoint',
        '-c',
        type=str,
        default='convnext_tiny_stage1_results/best_convnext_tiny_stage1.pth',
        help='Model'
    )
    parser.add_argument('--convnext_size', type=str, default='tiny', choices=['tiny', 'small', 'base', 'large'], help='ConvNeXt')
    parser.add_argument('--num_classes', type=int, default=1, choices=[1, 2, 4], help=' (1=sigmoid, 2=binary classification, 4=4-class)')
    parser.add_argument('--threshold', type=float, default=0.5, help='binary segmentation threshold')
    parser.add_argument('--tta', action='store_true', help='enable Stage 1 flip TTA')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(PROJECT_ROOT, "image datasets")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = args.num_classes
    if num_classes == 1:
        class_names = {0: 'Background', 1: 'Foreground'}
        task_desc = "Stage 1  (sigmoid)"
        activation = 'sigmoid'
    elif num_classes == 2:
        class_names = {0: 'Background', 1: 'Foreground'}
        task_desc = "Stage 1 binary classification (Foreground)"
        activation = 'softmax'
    else:
        class_names = {0: 'Background', 1: 'Single Layer', 2: 'Contaminant', 3: 'Multilayer'}
        task_desc = "Stage 2 4-class"
        activation = 'softmax'

    if args.convnext_size == 'tiny':
        encoder_name = 'convnext_tiny'
    elif args.convnext_size == 'small':
        encoder_name = 'convnext_small'
    elif args.convnext_size == 'base':
        encoder_name = 'convnext_base'
    else:
        encoder_name = 'convnext_large'

    print("=" * 60)
    print(f"ConvNeXt + U-Net  ({task_desc})")
    print("=" * 60)
    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(base_dir, checkpoint_path)

    print(f"Model: {checkpoint_path}")
    print(f"ConvNeXt : {args.convnext_size.upper()}")
    print(f"Num classes: {num_classes}")
    print(f"Device: {device}")

    # Model
    model = ConvNeXtUNet(
        encoder_name=encoder_name,
        in_channels=3,
        classes=num_classes,
        activation=activation
    )

    # 
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    model = model.to(device)
    model.eval()
    print("Model!")

    # Dataset
    print("\nLoading validation set...")
    
    if num_classes in (1, 2):
        val_dataset = Stage1Dataset(
            image_dir=os.path.join(dataset_root, 'validation dataset'),
            annotation_file=os.path.join(
                dataset_root,
                'annotation COCO dataset',
                'instances_val2019.json'
            )
        )
    else:
        val_dataset = Stage2Dataset(
            image_dir=os.path.join(dataset_root, 'validation dataset'),
            annotation_file=os.path.join(
                dataset_root,
                'annotation COCO dataset',
                'instances_val2019.json'
            )
        )


    print(f"Validation set: {len(val_dataset)} images")

    # 
    results = evaluate_model(
        model,
        val_dataset,
        device,
        num_classes,
        args.threshold,
        use_tta=args.tta,
    )

    # 
    print("\n" + "=" * 60)
    print("")
    print("=" * 60)
    if num_classes in (1, 2):
        print("Mean metrics (Background + Foreground):")
    print(f"Mean IoU: {results['mean_iou']:.4f}")
    print(f"Mean Dice: {results['mean_dice']:.4f}")
    print(f"Mean Precision: {results['mean_precision']:.4f}")
    print(f"Mean Recall: {results['mean_recall']:.4f}")
    print(f"Mean F1: {results['mean_f1']:.4f}")

    print("\nClassMetrics:")
    for cls_id, metrics in results['class_metrics'].items():
        name = class_names.get(cls_id, f'Class {cls_id}')
        print(f"  {name}: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}, "
              f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

    # 
    checkpoint_dir = os.path.dirname(checkpoint_path)
    results_file = os.path.join(checkpoint_dir, 'evaluation_results.txt')

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"ConvNeXt + U-Net  ({task_desc})\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {checkpoint_path}\n")
        f.write(f"ConvNeXt : {args.convnext_size.upper()}\n")
        f.write(f"Num classes: {num_classes}\n")
        if num_classes == 1:
            f.write(f"threshold: {args.threshold}\n")
            f.write(f"Flip TTA: {'enabled' if args.tta else 'disabled'}\n")
        if num_classes in (1, 2):
            f.write("Mean metrics (Background + Foreground):\n")
        f.write(f"Mean IoU: {results['mean_iou']:.4f}\n")
        f.write(f"Mean Dice: {results['mean_dice']:.4f}\n")
        f.write(f"Mean Precision: {results['mean_precision']:.4f}\n")
        f.write(f"Mean Recall: {results['mean_recall']:.4f}\n")
        f.write(f"Mean F1: {results['mean_f1']:.4f}\n")
        f.write("\nClassMetrics:\n")
        for cls_id, metrics in results['class_metrics'].items():
            name = class_names.get(cls_id, f'Class {cls_id}')
            f.write(f"  {name}: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}, "
                    f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}\n")

    print(f"\nResults saved to: {results_file}")
    return results


if __name__ == '__main__':
    main()
# python evaluate_convnext.py -c convnext_tiny_binary_results_convnext/best_convnext_tiny_binary.pth --num_classes 2

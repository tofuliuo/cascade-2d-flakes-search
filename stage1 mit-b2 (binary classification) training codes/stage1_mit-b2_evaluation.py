"""
MiT + U-Net evaluation script Stage 1 
Evaluation metrics: IoU, Dice, Precision, Recall, F1
Run:
    python evaluate_mit.py --checkpoint mit_b0_stage1_results/best_mit_b0_stage1.pth
    python evaluate_mit.py --checkpoint mit_b2_stage1_results/best_mit_b2_stage1.pth
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


# ===================== Model =====================
class MiTUNet(nn.Module):
    """Stage 1/2  MiT + U-Net Model"""

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


# ===================== Dataset =====================
class Stage1Dataset:
    """Stage 1 Dataset: binary classification (Background/Foreground)"""

    def __init__(self, image_dir, annotation_file):
        self.image_dir = image_dir

        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.coco_data = json.load(f)

        self.single_layer_id = next(
            cat['id'] for cat in self.coco_data['categories']
            if cat['name'] == 'single layer'
        )

        self.image_id_to_info = {
            img['id']: img for img in self.coco_data['images']
        }

        self.image_id_to_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append(ann)

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

        if image_id in self.image_id_to_annotations:
            for ann in self.image_id_to_annotations[image_id]:
                if ann['category_id'] != self.single_layer_id:
                    continue
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
                            mask = np.maximum(mask, np.array(temp_mask))

        if image.shape[:2] != (512, 512):
            image = cv2.resize(image, (512, 512))

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
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
        f1_score = (2 * precision * recall + 1e-6) / (precision + recall + 1e-6)

        ious[cls_id] = float(iou)
        dices[cls_id] = float(dice)
        precisions[cls_id] = float(precision)
        recalls[cls_id] = float(recall)
        f1s[cls_id] = float(f1_score)

    return ious, dices, precisions, recalls, f1s


def evaluate_model(model, dataset, device, num_classes, is_binary=True):
    """Model"""
    model.eval()

    all_ious = {i: [] for i in range(num_classes)}
    all_dices = {i: [] for i in range(num_classes)}
    all_precisions = {i: [] for i in range(num_classes)}
    all_recalls = {i: [] for i in range(num_classes)}
    all_f1s = {i: [] for i in range(num_classes)}

    print("\nStart evaluation...")

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Evaluating"):
            image, gt_mask, _ = dataset[i]
            image = image.unsqueeze(0).to(device)

            output = model(image)

            if is_binary:
                pred_mask = (output.squeeze() > 0.5).cpu().numpy().astype(np.uint8)
                gt_mask = gt_mask.squeeze().numpy()
                num_eval_classes = 2
            else:
                pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
                gt_mask = gt_mask.squeeze().numpy()
                num_eval_classes = num_classes



            ious, dices, precisions, recalls, f1s = compute_metrics(pred_mask, gt_mask, num_eval_classes)

            for cls_id in range(num_eval_classes):
                all_ious[cls_id].append(ious[cls_id])
                all_dices[cls_id].append(dices[cls_id])
                all_precisions[cls_id].append(precisions[cls_id])
                all_recalls[cls_id].append(recalls[cls_id])
                all_f1s[cls_id].append(f1s[cls_id])

    class_metrics = None
    if is_binary:
        class_metrics = {}
        for cls_id in range(num_classes):
            class_metrics[cls_id] = {
                'iou': np.mean(all_ious[cls_id]) if all_ious[cls_id] else 0,
                'dice': np.mean(all_dices[cls_id]) if all_dices[cls_id] else 0,
                'precision': np.mean(all_precisions[cls_id]) if all_precisions[cls_id] else 0,
                'recall': np.mean(all_recalls[cls_id]) if all_recalls[cls_id] else 0,
                'f1': np.mean(all_f1s[cls_id]) if all_f1s[cls_id] else 0
            }

    valid_classes = range(num_classes) if is_binary else range(1, num_classes)
    mean_iou = np.mean([np.mean(all_ious[i]) for i in valid_classes if all_ious[i]])
    mean_dice = np.mean([np.mean(all_dices[i]) for i in valid_classes if all_dices[i]])
    mean_precision = np.mean([np.mean(all_precisions[i]) for i in valid_classes if all_precisions[i]])
    mean_recall = np.mean([np.mean(all_recalls[i]) for i in valid_classes if all_recalls[i]])
    mean_f1 = np.mean([np.mean(all_f1s[i]) for i in valid_classes if all_f1s[i]])

    results = {
        'mean_iou': mean_iou,
        'mean_dice': mean_dice,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'mean_f1': mean_f1
    }
    if is_binary:
        results['class_metrics'] = class_metrics
    return results


# ===================== Main function =====================
def main():
    parser = argparse.ArgumentParser(description='MiT + U-Net ')
    parser.add_argument('--checkpoint', '-c', type=str, default=None, help='Model')
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2], help='stage (1=binary classification, 2=4-class)')
    parser.add_argument('--mit_size', type=str, default='b2', choices=['b0', 'b2'], help='MiT')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(PROJECT_ROOT, "image datasets")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ModelConfiguration
    if args.stage == 1:
        num_classes = 2
        is_binary = True
        activation = 'sigmoid'
        encoder_name = f'mit_{args.mit_size}'
        class_names = {0: 'Background', 1: 'Foreground'}
        if args.checkpoint is None:
            args.checkpoint = os.path.join(base_dir, f'mit_{args.mit_size}_stage1_results', f'best_mit_{args.mit_size}_stage1.pth')
    else:
        num_classes = 4
        is_binary = False
        activation = 'softmax'
        encoder_name = f'mit_{args.mit_size}'
        if args.checkpoint is None:
            args.checkpoint = os.path.join(base_dir, f'mit_{args.mit_size}_stage2_results', f'best_mit_{args.mit_size}_stage2.pth')

    print("=" * 60)
    print(f"MiT + U-Net  (Stage {args.stage})")
    print("=" * 60)
    print(f"Model: {args.checkpoint}")
    print(f"MiT : {args.mit_size.upper()}")
    print(f"Num classes: {num_classes}")
    print(f"Device: {device}")

    # Model
    model = MiTUNet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1 if is_binary else num_classes,
        activation=activation
    )

    # 
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print("Model!")

    # Dataset
    print("\nLoading validation set...")
    if args.stage == 1:
        val_dataset = Stage1Dataset(
            image_dir=os.path.join(dataset_root, 'validation dataset'),
            annotation_file=os.path.join(dataset_root, 'annotation COCO dataset', 'instances_val2019.json')
        )
    else:
        val_dataset = Stage2Dataset(
            image_dir=os.path.join(dataset_root, 'validation dataset'),
            annotation_file=os.path.join(dataset_root, 'annotation COCO dataset', 'instances_val2019.json')
        )

    print(f"Validation set: {len(val_dataset)} images")

    # 
    results = evaluate_model(model, val_dataset, device, num_classes, is_binary)

    # 
    print("\n" + "=" * 60)
    print("")
    print("=" * 60)
    if is_binary:
        print("Mean metrics (Background + Foreground):")
    print(f"Mean IoU: {results['mean_iou']:.4f}")
    print(f"Mean Dice: {results['mean_dice']:.4f}")
    print(f"Mean Precision: {results['mean_precision']:.4f}")
    print(f"Mean Recall: {results['mean_recall']:.4f}")
    print(f"Mean F1: {results['mean_f1']:.4f}")

    if is_binary:
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
        f.write(f"MiT + U-Net  (Stage {args.stage})\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {args.checkpoint}\n")
        f.write(f"MiT : {args.mit_size.upper()}\n")
        f.write(f"Num classes: {num_classes}\n")
        if is_binary:
            f.write("Mean metrics (Background + Foreground):\n")
        f.write(f"Mean IoU: {results['mean_iou']:.4f}\n")
        f.write(f"Mean Dice: {results['mean_dice']:.4f}\n")
        f.write(f"Mean Precision: {results['mean_precision']:.4f}\n")
        f.write(f"Mean Recall: {results['mean_recall']:.4f}\n")
        f.write(f"Mean F1: {results['mean_f1']:.4f}\n")
        if is_binary:
            f.write("\nClassMetrics:\n")
            for cls_id, metrics in results['class_metrics'].items():
                name = class_names[cls_id]
                f.write(f"  {name}: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}, "
                        f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}\n")

    print(f"\nResults saved to: {results_file}")
    return results


if __name__ == '__main__':
    main()

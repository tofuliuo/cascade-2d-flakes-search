"""
Mask R-CNN Stage 1 ()
: vs 
 vs ( / + / +)

 torchvision Mask R-CNN (ResNet-50 FPN),COCO 
"""

import os
import sys
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
import albumentations as A

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import box_convert
import torchvision.ops as ops
try:
    import openpyxl
    from openpyxl import Workbook
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

# =====================  =====================
class Config:
    def __init__(self, include_multilayer=False, include_contaminants=False,
                 checkpoint=None, batch_size=4, epochs=50, lr=1e-4,
                 weight_decay=1e-4, backbone='resnet50'):
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATASET_ROOT = os.path.join(PROJECT_ROOT, "image datasets")
        self.CHECKPOINT = checkpoint

        # 
        self.include_multilayer = include_multilayer
        self.include_contaminants = include_contaminants

        # : (0) + (1)
        self.NUM_CLASSES = 2
        self.MODE_SUFFIX = "_maskrcnn_binary"
        self.MODE_DESC = "(Mask R-CNN)"

        # 
        suffix_parts = []
        if include_multilayer:
            suffix_parts.append('ml')
        if include_contaminants:
            suffix_parts.append('ct')
        extra_suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
 
        self.SAVE_DIR = os.path.join(self.BASE_DIR, 'resnet50_maskrcnn_stage1_results')

        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.LEARNING_RATE = lr
        self.WEIGHT_DECAY = weight_decay
        self.IMAGE_SIZE = (512, 512)
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.BACKBONE = backbone

# ===================== COCO → Mask R-CNN  =====================
class MaskRCNNDataset(Dataset):
    """
     COCO  Mask R-CNN 
    ,
    """

    def __init__(self, image_dir, annotation_file, config,
                is_train=True, augment=False):
        self.image_dir = image_dir
        self.config = config
        self.is_train = is_train
        self.augment = augment



        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=15, p=0.3),
            A.RandomScale(scale_limit=0.1, p=0.3),
            A.Resize(
                height=config.IMAGE_SIZE[0],
                width=config.IMAGE_SIZE[1]
            ),
            A.ElasticTransform(alpha=1, sigma=50, p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.3
            ),
            A.GaussNoise(p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.1),
        ])

        # Validation: resize only, no augmentation
        self.resize_transform = A.Compose([
            A.Resize(
                height=config.IMAGE_SIZE[0],
                width=config.IMAGE_SIZE[1]
            ),
        ])

        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.coco_data = json.load(f)

        # 
        self.image_id_to_info = {img['id']: img for img in self.coco_data['images']}
        self.image_id_to_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append(ann)

        # 
        self.category_id_to_name = {}
        for cat in self.coco_data['categories']:
            self.category_id_to_name[cat['id']] = cat['name']

        # 
        self.valid_image_ids = []
        for img_id in self.image_id_to_annotations.keys():
            img_info = self.image_id_to_info[img_id]
            img_path = os.path.join(self.image_dir, img_info['file_name'])
            if os.path.exists(img_path):
                self.valid_image_ids.append(img_id)

        self.valid_image_ids.sort()
        print(f" {len(self.valid_image_ids)} ")

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

    def _get_boxes_and_masks(self, image_id, img_height, img_width):
        """
        COCOboxesmasks
        
        """
        boxes = []
        masks = []
        labels = []

        if image_id not in self.image_id_to_annotations:
            return boxes, masks, labels

        all_masks_per_class = {}

        for ann in self.image_id_to_annotations[image_id]:
            category_id = ann['category_id']
            category_name = self.category_id_to_name.get(category_id, '')

            # 
            is_foreground = False
            if category_name == 'single layer':
                is_foreground = True
            elif category_name == 'multilayer' and self.config.include_multilayer:
                is_foreground = True
            elif category_name == 'contaminants' and self.config.include_contaminants:
                is_foreground = True

            if not is_foreground:
                continue

            if 'segmentation' not in ann or not ann['segmentation']:
                continue

            # 
            class_mask = self._polys_to_mask(ann['segmentation'], img_height, img_width)

            if category_id not in all_masks_per_class:
                all_masks_per_class[category_id] = class_mask.copy()
            else:
                all_masks_per_class[category_id] = np.maximum(
                    all_masks_per_class[category_id], class_mask
                )

        # 
        for category_id, class_mask in all_masks_per_class.items():
            # 
            num_labels, labeled, stats, centroids = cv2.connectedComponentsWithStats(class_mask, connectivity=8)

            for comp_id in range(1, num_labels):
                component_mask = (labeled == comp_id).astype(np.uint8)


                #  (xywh → xyxy)
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
                labels.append(1)  # label=1

        return boxes, masks, labels

    def _rebuild_instances_from_masks(self, masks, labels):
        new_boxes = []
        new_masks = []
        new_labels = []

        for mask, label in zip(masks, labels):
            mask = (np.asarray(mask) > 0).astype(np.uint8)

            ys, xs = np.where(mask > 0)

            if len(xs) == 0 or len(ys) == 0:
                continue

            x1 = xs.min()
            x2 = xs.max()
            y1 = ys.min()
            y2 = ys.max()

            if (x2 - x1) <= 1 or (y2 - y1) <= 1:
                continue

            new_boxes.append([x1, y1, x2, y2])
            new_masks.append(mask)
            new_labels.append(label)

        return new_boxes, new_masks, new_labels

    def __getitem__(self, idx):
        image_id = self.valid_image_ids[idx]
        img_info = self.image_id_to_info[image_id]

        # 
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        try:
            image = np.array(Image.open(img_path).convert('RGB'))
        except Exception:
            image = np.zeros((512, 512, 3), dtype=np.uint8)

        orig_h, orig_w = image.shape[:2]

        if self.is_train:
            boxes, masks, labels = self._get_boxes_and_masks(
                image_id,
                orig_h,
                orig_w
            )

            # Training: full augmentation
            # Validation: resize only
            if self.augment:
                transformed = self.train_transform(
                    image=image,
                    masks=masks
                )
            else:
                transformed = self.resize_transform(
                    image=image,
                    masks=masks
                )

            image = transformed['image']
            masks = transformed['masks']


            # Recalculate boxes from augmented masks
            boxes, masks, labels = self._rebuild_instances_from_masks(
                masks,
                labels
            )

            image_tensor = (
                torch.from_numpy(np.ascontiguousarray(image))
                .permute(2, 0, 1)
                .float()
                / 255.0
            )

            if len(boxes) == 0:
                target = {
                    'boxes': torch.zeros((0, 4), dtype=torch.float32),
                    'labels': torch.zeros((0,), dtype=torch.int64),
                    'masks': torch.zeros(
                        (0, *self.config.IMAGE_SIZE),
                        dtype=torch.float32
                    ),
                }
            else:
                target = {
                    'boxes': torch.tensor(boxes, dtype=torch.float32),
                    'labels': torch.tensor(labels, dtype=torch.int64),
                    'masks': torch.from_numpy(
                        np.stack(masks, axis=0)
                    ).float(),
                }

            return image_tensor, target

        else:
            # Validation / inference: no augmentation
            if image.shape[:2] != self.config.IMAGE_SIZE:
                image = cv2.resize(
                    image,
                    (self.config.IMAGE_SIZE[1], self.config.IMAGE_SIZE[0])
                )

            image_tensor = (
                torch.from_numpy(np.ascontiguousarray(image))
                .permute(2, 0, 1)
                .float()
                / 255.0
            )

            return image_tensor, image_id

    def get_raw_image(self, idx):
        """PIL()"""
        image_id = self.valid_image_ids[idx]
        img_info = self.image_id_to_info[image_id]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        return Image.open(img_path).convert('RGB'), img_info['file_name']


# =====================  =====================
def get_model(num_classes, device):
    """ Mask R-CNN """
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
        trainable_backbone_layers=5,
        min_size=512,
        max_size=512
    )
    # ( +  = 2)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    #  mask 
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model.to(device)


def collate_fn(batch):
    """batch collate"""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def train_model(model, train_loader, val_loader, config):
    """ Mask R-CNN"""
    model.train()

    # 
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    # 5-epoch warm-up + cosine annealing
    warmup_epochs = 5
    total_epochs = config.EPOCHS

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 
    total_steps = len(train_loader) * config.EPOCHS

    history = {
        'train_loss': [], 'val_loss': [],
    }

    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    min_delta = 1e-4

    for epoch in range(config.EPOCHS):
        model.train()
        epoch_loss = 0.0


        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        loss_sum = 0.0
        loss_count = 0

        for images, targets in pbar:
            images = [img.to(config.DEVICE) for img in images]
            targets = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets]

            # 
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # 
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            loss_sum += losses.item()
            loss_count += 1

            pbar.set_postfix({'loss': f'{loss_sum/max(loss_count,1):.4f}'})

        epoch_loss /= len(train_loader)

        # 
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation"):
                images = [img.to(config.DEVICE) for img in images]
                targets = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets]
                # Mask R-CNN returns detections (list) during eval mode, not losses
                # We pass targets during eval to get detection results, but skip loss calculation
                _ = model(images, targets)
                # For validation, we use a forward pass with targets in training mode to get losses
                model.train()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                val_count += 1
                model.eval()

        val_loss = val_loss / max(val_count, 1)

        # Update learning rate
        scheduler.step()

        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)

        print(f"\nEpoch {epoch+1}: train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}")

        # 
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = os.path.join(config.SAVE_DIR, 'best_resnet50_maskrcnn_stage1.pth')
            torch.save(model.state_dict(), best_path)
            print(f": {best_path}")
        else:
            patience_counter += 1
            print(f": {patience_counter}/{patience}")

        # 10epoch
        if (epoch + 1) % 10 == 0:
            plot_history(history, os.path.join(config.SAVE_DIR, 'training_history.png'))

        if patience_counter >= patience:
            print(f"！")
            break

    return history


def plot_history(history, save_path):
    """"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history['train_loss'], label='Train Loss')
    ax.plot(history['val_loss'], label='Val Loss')
    ax.set_title('Training History')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f": {save_path}")


# =====================  =====================
def visualize_inference(model, dataset, output_dir, num_images=8, device='cpu'):
    """"""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    from torchvision.ops import nms

    for i in tqdm(range(min(num_images, len(dataset))), desc=""):
        image, target = dataset[i]
        orig_image, filename = dataset.get_raw_image(i)
        orig_w, orig_h = orig_image.size

        # 
        with torch.no_grad():
            image_norm = image.unsqueeze(0).to(device)
            outputs = model([image_norm])[0]

        # 
        keep = outputs['scores'] > 0.5
        boxes = outputs['boxes'][keep].cpu().numpy()
        masks = outputs['masks'][keep].cpu().numpy()
        scores = outputs['scores'][keep].cpu().numpy()

        # 
        scale_x = orig_w / dataset.config.IMAGE_SIZE[1]
        scale_y = orig_h / dataset.config.IMAGE_SIZE[0]

        orig_image_np = np.array(orig_image)
        overlay = orig_image_np.copy()

        # 
        np.random.seed(42)
        colors = np.random.randint(0, 255, (max(len(boxes), 1), 3)).tolist()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 
        axes[0].imshow(orig_image_np)
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')

        # mask
        for j, (box, mask, score) in enumerate(zip(boxes, masks, scores)):
            color = colors[j % len(colors)]
            color = tuple(c / 255 for c in color)

            # 
            x1, y1, x2, y2 = box * [scale_x, scale_y, scale_x, scale_y]
            mask_resized = cv2.resize(mask[0], (orig_w, orig_h))
            binary_mask = (mask_resized > 0.5).astype(np.uint8)

            # 
            for c in range(3):
                overlay[:, :, c] = np.where(
                    binary_mask > 0,
                    overlay[:, :, c] * 0.5 + color[c] * 255 * 0.5,
                    overlay[:, :, c]
                ).astype(np.uint8)

            # 
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        axes[1].imshow(overlay)
        axes[1].set_title(f'Predictions (n={len(boxes)})', fontsize=12)
        axes[1].axis('off')

        # mask
        mask_overlay = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        for j, (box, mask) in enumerate(zip(boxes, masks)):
            color = colors[j % len(colors)]
            mask_resized = cv2.resize(mask[0], (orig_w, orig_h))
            binary_mask = (mask_resized > 0.5).astype(np.uint8)
            mask_overlay[binary_mask > 0] = color

        axes[2].imshow(mask_overlay)
        axes[2].set_title('Instance Masks', fontsize=12)
        axes[2].axis('off')

        name = os.path.splitext(filename)[0]
        plt.suptitle(f'{name}', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}_visualization.png'), dpi=150, bbox_inches='tight')
        plt.close()

        print(f": {name}_visualization.png")

    print(f": {output_dir}")


# =====================  =====================
def main():
    import torchvision
    parser = argparse.ArgumentParser(description='Mask R-CNN Stage1 ')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    parser.add_argument('--epochs', type=int, default=50, help='')
    parser.add_argument('--batch_size', type=int, default=4, help='')
    parser.add_argument('--lr', type=float, default=1e-4, help='')
    parser.add_argument(
        '--include_multilayer',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='include multilayer as foreground'
    )

    parser.add_argument(
        '--include_contaminants',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='include contaminants as foreground'
    )
    parser.add_argument('--visualize', action='store_true', help='')
    args = parser.parse_args()

    config = Config(
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        include_multilayer=args.include_multilayer,
        include_contaminants=args.include_contaminants,
    )

    print("=" * 60)
    print("Mask R-CNN Stage 1 ")
    print("=" * 60)
    print(f": ", end="")
    if args.include_multilayer:
        print(" + ", end="")
    if args.include_contaminants:
        print(" + ", end="")
    print()
    print(f": {config.DEVICE}")
    print(f": {config.SAVE_DIR}")
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    # 
    print("\n...")
    train_dataset = MaskRCNNDataset(
        image_dir=os.path.join(config.DATASET_ROOT, 'training dataset'),
        annotation_file=os.path.join(
            config.DATASET_ROOT,
            'annotation COCO dataset',
            'instances_train2019.json'
        ),
        config=config,
        is_train=True,
        augment=True,
    )

    val_dataset = MaskRCNNDataset(
        image_dir=os.path.join(config.DATASET_ROOT, 'validation dataset'),
        annotation_file=os.path.join(
            config.DATASET_ROOT,
            'annotation COCO dataset',
            'instances_val2019.json'
        ),
        config=config,
        is_train=True,
        augment=False,
    )


    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                            num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)

    print(f": {len(train_dataset)} , : {len(val_dataset)} ")

    # 
    model = get_model(num_classes=config.NUM_CLASSES, device=config.DEVICE)
    print(f": {config.BACKBONE} FPN + Mask R-CNN")

    # 
    print("\n...")
    history = train_model(model, train_loader, val_loader, config)

    # 
    final_path = os.path.join(config.SAVE_DIR, 'final_resnet50_maskrcnn_stage1.pth')
    torch.save(model.state_dict(), final_path)
    print(f": {final_path}")

    # 
    if args.visualize:
        print("\n...")
        vis_dir = os.path.join(config.SAVE_DIR, 'visualization')
        visualize_inference(model, val_dataset, vis_dir, num_images=8, device=config.DEVICE)

    print("\n!")
    print(f": {min(history['val_loss']):.4f}")


if __name__ == "__main__":
    main()


# # Stage 1
# python maskrcnn_stage1.py --epochs 50


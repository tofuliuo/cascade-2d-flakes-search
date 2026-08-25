"""
Mask R-CNN Stage 2 ()
: vs 
 vs  vs  vs 

,

 torchvision Mask R-CNN (ResNet-50 FPN),COCO 

:
- label 0 = ()
- label 1 = ()
- label 2 = 
- label 3 = 
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
try:
    import openpyxl
    from openpyxl import Workbook
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

# =====================  =====================

class Config:
    def __init__(
        self,
        checkpoint=None,
        batch_size=4,
        epochs=50,
        lr=1e-4,
        weight_decay=1e-4,
        backbone='resnet50'
    ):
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATASET_ROOT = os.path.join(PROJECT_ROOT, "image datasets")
        self.CHECKPOINT = checkpoint

        # Foreground classes:
        # 1 = single layer
        # 2 = contaminant
        # 3 = multilayer
        # Mask R-CNN additionally reserves label 0 for background
        self.NUM_CLASSES = 3
        self.MODE_SUFFIX = "_maskrcnn_4class"
        self.MODE_DESC = "(Mask R-CNN)"

        self.SAVE_DIR = os.path.join(
            self.BASE_DIR,
            'resnet50_maskrcnn_stage2_results'
        )

        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.LEARNING_RATE = lr
        self.WEIGHT_DECAY = weight_decay
        self.IMAGE_SIZE = (512, 512)
        self.DEVICE = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.BACKBONE = backbone

        self.STAGE1_MODEL_PATH = os.path.join(
            PROJECT_ROOT,
            'stage1 maskrcnn (binary classification) training codes',
            'resnet50_maskrcnn_stage1_results',
            'best_resnet50_maskrcnn_stage1.pth',
        )

        self.STAGE1_SCORE_THRESHOLD = 0.5
        self.PRIOR_FOREGROUND_WEIGHT = 1.0
        self.PRIOR_BACKGROUND_WEIGHT = 0.2
        self.PRIOR_UNCERTAINTY_BAND = 0.15
        self.GLOBAL_LOSS_WEIGHT = 0.2


# =====================  =====================
class MaskRCNNStage2Dataset(Dataset):
    """
    Stage 2: 
    - class 0 = ()
    - class 1 =  (single layer)
    - class 2 =  (contaminant)
    - class 3 =  (multilayer)
    """

    def __init__(
        self,
        image_dir,
        annotation_file,
        config,
        is_train=True,
        augment=False
    ):
        self.image_dir = image_dir
        self.config = config
        self.is_train = is_train
        self.augment = augment

        # Same training augmentation recipe used by ConvNeXt/MiT
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

            A.ElasticTransform(
                alpha=1,
                sigma=50,
                p=0.2
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.3
            ),

            A.GaussNoise(p=0.2),
            A.GaussianBlur(
                blur_limit=3,
                p=0.1
            ),
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

        self.image_id_to_info = {img['id']: img for img in self.coco_data['images']}
        self.image_id_to_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append(ann)

        # : single layer → 1, contaminants → 2, multilayer → 3
        # (0  Mask R-CNN )
        self.category_id_to_label = {}
        for cat in self.coco_data['categories']:
            name = cat['name']
            if name == 'single layer':
                self.category_id_to_label[cat['id']] = 1
            elif name == 'contaminants':
                self.category_id_to_label[cat['id']] = 2
            elif name == 'multilayer':
                self.category_id_to_label[cat['id']] = 3

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

    

    def _get_instances(self, image_id, img_height, img_width):
        """
        
        : boxes, masks, labels
        - boxes: List[list[4]] xyxy
        - masks: List[np.array] HxW
        - labels: List[int] 1=, 2=, 3=
        """
        boxes = []
        masks = []
        labels = []

        if image_id not in self.image_id_to_annotations:
            return boxes, masks, labels

        # 
        class_masks = {}  # label → binary mask

        for ann in self.image_id_to_annotations[image_id]:
            category_id = ann['category_id']
            label = self.category_id_to_label.get(category_id, None)
            if label is None:
                continue

            if 'segmentation' not in ann or not ann['segmentation']:
                continue

            class_mask = self._polys_to_mask(ann['segmentation'], img_height, img_width)
            if label not in class_masks:
                class_masks[label] = class_mask.copy()
            else:
                class_masks[label] = np.maximum(class_masks[label], class_mask)

        # ""/
        #  =  -  - 
        if 1 in class_masks and 3 in class_masks:
            class_masks[1] = class_masks[1] - class_masks[3]
            class_masks[1] = np.clip(class_masks[1], 0, 1).astype(np.uint8)
        if 1 in class_masks and 2 in class_masks:
            class_masks[1] = class_masks[1] - class_masks[2]
            class_masks[1] = np.clip(class_masks[1], 0, 1).astype(np.uint8)

        # 
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

                # //
                box_w = cmax - cmin
                box_h = rmax - rmin
                if box_w <= 1 or box_h <= 1:
                    continue

                boxes.append([cmin, rmin, cmax, rmax])
                masks.append(component_mask)
                labels.append(label)

        return boxes, masks, labels

    def _rebuild_instances_from_masks(self, masks, labels):
        """
        Recalculate bounding boxes from transformed instance masks.
        """
        new_boxes = []
        new_masks = []
        new_labels = []

        for mask, label in zip(masks, labels):
            mask = (np.asarray(mask) > 0).astype(np.uint8)

            ys, xs = np.where(mask > 0)

            if len(xs) == 0 or len(ys) == 0:
                continue

            x1 = int(xs.min())
            x2 = int(xs.max())
            y1 = int(ys.min())
            y2 = int(ys.max())

            if (x2 - x1) <= 1 or (y2 - y1) <= 1:
                continue

            new_boxes.append([x1, y1, x2, y2])
            new_masks.append(mask)
            new_labels.append(label)

        return new_boxes, new_masks, new_labels


    def __getitem__(self, idx):
        image_id = self.valid_image_ids[idx]
        img_info = self.image_id_to_info[image_id]

        img_path = os.path.join(
            self.image_dir,
            img_info['file_name']
        )

        try:
            image = np.array(
                Image.open(img_path).convert('RGB')
            )
        except Exception:
            image = np.zeros(
                (512, 512, 3),
                dtype=np.uint8
            )

        orig_h, orig_w = image.shape[:2]

        # -------------------------------------------------
        # Training/validation mode requiring targets
        # -------------------------------------------------
        if self.is_train:

            # Stage-2 GT:
            # clean single layer + contaminant + multilayer
            boxes, masks, labels = self._get_instances(
                image_id,
                orig_h,
                orig_w
            )

            # ---------------------------------------------
            # Training: full augmentation
            # Validation: resize only
            # ---------------------------------------------
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

            # Recalculate bounding boxes after transformation
            boxes, masks, labels = self._rebuild_instances_from_masks(
                masks,
                labels
            )

            # torchvision Mask R-CNN expects [0,1] RGB tensor
            image_tensor = (
                torch.from_numpy(
                    np.ascontiguousarray(image)
                )
                .permute(2, 0, 1)
                .float()
                / 255.0
            )

            if len(boxes) == 0:
                target = {
                    'boxes': torch.zeros(
                        (0, 4),
                        dtype=torch.float32
                    ),
                    'labels': torch.zeros(
                        (0,),
                        dtype=torch.int64
                    ),
                    'masks': torch.zeros(
                        (
                            0,
                            self.config.IMAGE_SIZE[0],
                            self.config.IMAGE_SIZE[1]
                        ),
                        dtype=torch.float32
                    ),
                }

            else:
                target = {
                    'boxes': torch.tensor(
                        boxes,
                        dtype=torch.float32
                    ),
                    'labels': torch.tensor(
                        labels,
                        dtype=torch.int64
                    ),
                    'masks': torch.from_numpy(
                        np.stack(
                            masks,
                            axis=0
                        )
                    ).float(),
                }

            return image_tensor, target

        # -------------------------------------------------
        # Pure inference mode: no augmentation
        # -------------------------------------------------
        else:

            transformed = self.resize_transform(
                image=image
            )

            image = transformed['image']

            image_tensor = (
                torch.from_numpy(
                    np.ascontiguousarray(image)
                )
                .permute(2, 0, 1)
                .float()
                / 255.0
            )

            return image_tensor, image_id



    def get_raw_image(self, idx):
        image_id = self.valid_image_ids[idx]
        img_info = self.image_id_to_info[image_id]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        return Image.open(img_path).convert('RGB'), img_info['file_name']

# =====================  =====================
def get_model(num_classes, device, pretrained=True):
    """
     Mask R-CNN
    num_classes = ( Mask R-CNN )
     num_classes + 1 
    """
    if pretrained:
        model = maskrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
            trainable_backbone_layers=5,
            min_size=512,
            max_size=512
        )
    else:
        model = maskrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            trainable_backbone_layers=5,
            min_size=512,
            max_size=512
        )
    # :  + num_classes 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

    #  mask 
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes + 1)

    return model.to(device)


def load_stage1_model(config):
    """Load and freeze Stage 1 Mask R-CNN for online foreground priors."""
    checkpoint_path = config.STAGE1_MODEL_PATH
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {checkpoint_path}")

    model = get_model(num_classes=1, device=config.DEVICE, pretrained=True)
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print(f"Stage 1 model loaded for online priors: {checkpoint_path}")
    return model


def stage1_outputs_to_priors(outputs, image_size, score_threshold):
    """Merge accepted Stage 1 instance probabilities into foreground maps."""
    priors = []
    for output in outputs:
        prior = torch.zeros(image_size, dtype=torch.float32, device=output['scores'].device)
        keep = (output['labels'] == 1) & (output['scores'] >= score_threshold)
        masks = output['masks'][keep, 0]
        if masks.numel() > 0:
            prior = masks.amax(dim=0)
        priors.append(torch.clamp(prior, 0.0, 1.0))
    return priors


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def compute_prior_focus_gain(priors, config, device):
    prior = torch.stack([p.to(device).float() for p in priors], dim=0)  # [B,H,W]
    prior = torch.clamp(prior, 0.0, 1.0)

    fg_w = getattr(config, "PRIOR_FOREGROUND_WEIGHT", 1.0)
    bg_w = getattr(config, "PRIOR_BACKGROUND_WEIGHT", 0.2)
    unc_band = getattr(config, "PRIOR_UNCERTAINTY_BAND", 0.15)

    raw_pixel_weight = fg_w * prior + bg_w * (1.0 - prior)
    confidence = torch.abs(prior - 0.5)
    confidence = torch.clamp((confidence - unc_band) / (0.5 - unc_band + 1e-6), 0.0, 1.0)
    pixel_weight = 1.0 + (raw_pixel_weight - 1.0) * confidence

    fg_score = (pixel_weight * prior).sum() / (prior.sum() + 1e-6)
    bg_score = (pixel_weight * (1.0 - prior)).sum() / ((1.0 - prior).sum() + 1e-6)
    gain = fg_score / (bg_score + 1e-6)
    return torch.clamp(gain, 1.0, 3.0)


def train_model(model, stage1_model, train_loader, val_loader, config):
    model.train()
    stage1_model.eval()
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

    history = {'train_loss': [], 'val_loss': []}
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

        for i, (images, targets) in enumerate(pbar):
            # :
            for j, target in enumerate(targets):
                boxes = target['boxes']
                if len(boxes) > 0:
                    box_w = boxes[:, 2] - boxes[:, 0]
                    box_h = boxes[:, 3] - boxes[:, 1]
                    invalid = (box_w <= 0) | (box_h <= 0)
                    if invalid.any():
                        print(f"\n[DEBUG] Batch {i}, Target {j}:  {invalid.sum()} ")
                        print(f"  : {boxes[invalid]}")

            images = [img.to(config.DEVICE) for img in images]
            targets = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets]
            with torch.no_grad():
                stage1_outputs = stage1_model(images)
                priors = stage1_outputs_to_priors(
                    stage1_outputs,
                    config.IMAGE_SIZE,
                    config.STAGE1_SCORE_THRESHOLD,
                )
            prior_focus_gain = compute_prior_focus_gain(priors, config, config.DEVICE)

            loss_dict = model(images, targets)
            loss_classifier = loss_dict.get('loss_classifier', torch.tensor(0.0, device=config.DEVICE))
            loss_box_reg = loss_dict.get('loss_box_reg', torch.tensor(0.0, device=config.DEVICE))
            loss_mask = loss_dict.get('loss_mask', torch.tensor(0.0, device=config.DEVICE))
            loss_objectness = loss_dict.get('loss_objectness', torch.tensor(0.0, device=config.DEVICE))
            loss_rpn_box_reg = loss_dict.get('loss_rpn_box_reg', torch.tensor(0.0, device=config.DEVICE))

            local_loss = prior_focus_gain * (loss_classifier + loss_box_reg + loss_mask)
            global_loss = loss_objectness + loss_rpn_box_reg
            lambda_global = getattr(config, "GLOBAL_LOSS_WEIGHT", 0.2)
            losses = (1.0 - lambda_global) * local_loss + lambda_global * global_loss

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                stage1_outputs = stage1_model(images)
                priors = stage1_outputs_to_priors(
                    stage1_outputs,
                    config.IMAGE_SIZE,
                    config.STAGE1_SCORE_THRESHOLD,
                )
                prior_focus_gain = compute_prior_focus_gain(priors, config, config.DEVICE)
                # Mask R-CNN returns detections (list) during eval mode, not losses
                # We pass targets during eval to get detection results, but skip loss calculation
                _ = model(images, targets)
                # For validation, we use a forward pass with targets in training mode to get losses
                model.train()
                loss_dict = model(images, targets)
                loss_classifier = loss_dict.get('loss_classifier', torch.tensor(0.0, device=config.DEVICE))
                loss_box_reg = loss_dict.get('loss_box_reg', torch.tensor(0.0, device=config.DEVICE))
                loss_mask = loss_dict.get('loss_mask', torch.tensor(0.0, device=config.DEVICE))
                loss_objectness = loss_dict.get('loss_objectness', torch.tensor(0.0, device=config.DEVICE))
                loss_rpn_box_reg = loss_dict.get('loss_rpn_box_reg', torch.tensor(0.0, device=config.DEVICE))

                local_loss = prior_focus_gain * (loss_classifier + loss_box_reg + loss_mask)
                global_loss = loss_objectness + loss_rpn_box_reg
                lambda_global = getattr(config, "GLOBAL_LOSS_WEIGHT", 0.2)
                losses = (1.0 - lambda_global) * local_loss + lambda_global * global_loss
                val_loss += losses.item()
                val_count += 1
                model.eval()

        val_loss = val_loss / max(val_count, 1)

        scheduler.step()

        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)

        print(f"\nEpoch {epoch+1}: train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = os.path.join(config.SAVE_DIR, 'best_resnet50_maskrcnn_stage2.pth')
            torch.save(model.state_dict(), best_path)
            print(f": {best_path}")
        else:
            patience_counter += 1
            print(f": {patience_counter}/{patience}")

        if (epoch + 1) % 10 == 0:
            plot_history(history, os.path.join(config.SAVE_DIR, 'training_history.png'))

        if patience_counter >= patience:
            print(f"！")
            break

    return history


def plot_history(history, save_path):
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
CLASS_NAMES = {
    0: 'Background',
    1: 'Single Layer',
    2: 'Contaminant',
    3: 'Multilayer',
}
CLASS_COLORS = {
    1: (0, 255, 0),      #  - 
    2: (255, 0, 255),     #  - 
    3: (255, 255, 0),     #  - 
}


def visualize_inference(model, dataset, output_dir, num_images=8, device='cpu'):
    """"""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    np.random.seed(42)

    for i in tqdm(range(min(num_images, len(dataset))), desc=""):
        image, target = dataset[i]
        orig_image, filename = dataset.get_raw_image(i)
        orig_w, orig_h = orig_image.size

        with torch.no_grad():
            image_norm = image.unsqueeze(0).to(device)
            outputs = model([image_norm])[0]

        keep = outputs['scores'] > 0.5
        boxes = outputs['boxes'][keep].cpu().numpy()
        masks = outputs['masks'][keep].cpu().numpy()
        labels = outputs['labels'][keep].cpu().numpy()
        scores = outputs['scores'][keep].cpu().numpy()

        scale_x = orig_w / 512
        scale_y = orig_h / 512

        orig_image_np = np.array(orig_image)
        overlay = orig_image_np.copy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(orig_image_np)
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')

        # 
        for box, mask, label, score in zip(boxes, masks, labels, scores):
            color = CLASS_COLORS.get(int(label), (128, 128, 128))
            color_norm = tuple(c / 255 for c in color)

            x1, y1, x2, y2 = box * [scale_x, scale_y, scale_x, scale_y]
            mask_resized = cv2.resize(mask[0], (orig_w, orig_h))
            binary_mask = (mask_resized > 0.5).astype(np.uint8)

            for c in range(3):
                overlay[:, :, c] = np.where(
                    binary_mask > 0,
                    (overlay[:, :, c] * 0.5 + color[c] * 0.5).astype(np.uint8),
                    overlay[:, :, c]
                )
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            cx = int((x1 + x2) / 2)
            cy = int(y1) - 5
            cv2.putText(overlay, f'{CLASS_NAMES.get(int(label), "Unknown")}:{score:.2f}',
                       (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        axes[1].imshow(overlay)
        axes[1].set_title(f'Instances (n={len(boxes)})', fontsize=12)
        axes[1].axis('off')

        # mask
        mask_overlay = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        for box, mask, label in zip(boxes, masks, labels):
            color = CLASS_COLORS.get(int(label), (128, 128, 128))
            mask_resized = cv2.resize(mask[0], (orig_w, orig_h))
            binary_mask = (mask_resized > 0.5).astype(np.uint8)
            mask_overlay[binary_mask > 0] = color

        axes[2].imshow(mask_overlay)
        axes[2].set_title('Instance Masks\nGreen=SL, Purple=CT, Yellow=ML', fontsize=10)
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
    parser = argparse.ArgumentParser(description='Mask R-CNN Stage2 ')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    parser.add_argument('--epochs', type=int, default=50, help='')
    parser.add_argument('--batch_size', type=int, default=4, help='')
    parser.add_argument('--lr', type=float, default=1e-4, help='')
    parser.add_argument('--visualize', action='store_true', help='')
    parser.add_argument('--stage1_score_thresh', type=float, default=0.5,
                        help='minimum Stage 1 detection score used in the foreground prior')
    parser.add_argument('--prior_fg_weight', type=float, default=1.0,
                        help='loss weight for Stage1 predicted foreground')
    parser.add_argument('--prior_bg_weight', type=float, default=0.2,
                        help='loss weight for Stage1 predicted background')
    parser.add_argument('--prior_uncertainty_band', type=float, default=0.15,
                        help='do not strongly reweight uncertain Stage1 area around 0.5')
    parser.add_argument('--global_loss_weight', type=float, default=0.2,
                        help='global loss anchor weight to preserve global context')
    args = parser.parse_args()

    config = Config(
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
    )
    config.STAGE1_SCORE_THRESHOLD = args.stage1_score_thresh
    config.PRIOR_FOREGROUND_WEIGHT = args.prior_fg_weight
    config.PRIOR_BACKGROUND_WEIGHT = args.prior_bg_weight
    config.PRIOR_UNCERTAINTY_BAND = args.prior_uncertainty_band
    config.GLOBAL_LOSS_WEIGHT = args.global_loss_weight

    print("=" * 60)
    print("Mask R-CNN Stage 2 ")
    print("=" * 60)
    print(": 0=, 1=(), 2=(), 3=()")
    print(f": {config.DEVICE}")
    print(f": {config.SAVE_DIR}")
    print(f"Stage 1 checkpoint: {config.STAGE1_MODEL_PATH}")
    print(f"Stage 1 score threshold: {config.STAGE1_SCORE_THRESHOLD}")
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    print("\n...")
    train_dataset = MaskRCNNStage2Dataset(
        image_dir=os.path.join(
            config.DATASET_ROOT,
            'training dataset'
        ),
        annotation_file=os.path.join(
            config.DATASET_ROOT,
            'annotation COCO dataset',
            'instances_train2019.json'
        ),
        config=config,
        is_train=True,
        augment=True,
    )

    val_dataset = MaskRCNNStage2Dataset(
        image_dir=os.path.join(
            config.DATASET_ROOT,
            'validation dataset'
        ),
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

    model = get_model(num_classes=config.NUM_CLASSES, device=config.DEVICE)
    stage1_model = load_stage1_model(config)
    print(f": {config.BACKBONE} FPN + Mask R-CNN ({config.NUM_CLASSES} +  = {config.NUM_CLASSES + 1})")

    print("\n...")
    history = train_model(model, stage1_model, train_loader, val_loader, config)

    final_path = os.path.join(config.SAVE_DIR, 'final_resnet50_maskrcnn_stage2.pth')
    torch.save(model.state_dict(), final_path)
    print(f": {final_path}")

    if args.visualize:
        print("\n...")
        vis_dir = os.path.join(config.SAVE_DIR, 'visualization')
        visualize_inference(model, val_dataset, vis_dir, num_images=8, device=config.DEVICE)

    print("\n!")
    print(f": {min(history['val_loss']):.4f}")


if __name__ == "__main__":
    main()

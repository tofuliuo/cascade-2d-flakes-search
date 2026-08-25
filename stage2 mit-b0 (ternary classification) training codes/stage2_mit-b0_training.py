"""
MiT + U-Net Stage 2 
:label 0=, label 1=, label 2=, label 3=
 stage_2.py , MiT-B0/B2 
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
os.environ['HF_HOME'] = BASE_DIR

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
try:
    import openpyxl
    from openpyxl import Workbook
    OPENPYXL_AVAILABLE = True
except ImportError:
    print(": openpyxl,Excel")
    OPENPYXL_AVAILABLE = False
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    print(": timm,MiT")
    TIMM_AVAILABLE = False

# =====================  =====================
def parse_args():
    parser = argparse.ArgumentParser(description='MiT + U-Net Stage2 ')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='checkpoint')
    parser.add_argument('--epochs', type=int, default=50,
                       help=' (: 50)')
    parser.add_argument('--mit_size', type=str, default='b0',
                       choices=['b0', 'b2'],
                       help='MiT (: b0)')
    parser.add_argument('--dropout', action='store_true',
                       help='Dropout')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'],
                       help=' (: adamw)')
    parser.add_argument('--prior_fg_weight', type=float, default=1.0,
                       help='loss weight for Stage1 predicted foreground')
    parser.add_argument('--prior_bg_weight', type=float, default=0.2,
                       help='loss weight for Stage1 predicted background')
    parser.add_argument('--prior_uncertainty_band', type=float, default=0.15,
                       help='do not strongly reweight uncertain Stage1 area around 0.5')
    parser.add_argument('--global_loss_weight', type=float, default=0.2,
                       help='global CE anchor weight to preserve global context')
    return parser.parse_args()

# =====================  =====================
class Config:
    def __init__(self, checkpoint=None, mit_size='b0', use_dropout=False, optimizer_type='adam'):
        self.BASE_DIR = BASE_DIR
        self.DATASET_ROOT = os.path.join(PROJECT_ROOT, "image datasets")
        self.CHECKPOINT = checkpoint

        # MiT 
        self.MIT_SIZE = mit_size
        if mit_size == 'b0':
            self.ENCODER_NAME = 'mit_b0'
        else:  # b2
            self.ENCODER_NAME = 'mit_b2'

        # Stage 2 :
        # label 0 = ()
        # label 1 = ()
        # label 2 = 
        # label 3 = 
        self.CLASSES = 4
        self.ACTIVATION = None
        self.NUM_CLASSES = 4
        self.MODE_SUFFIX = "_stage2_mit_4class"
        self.MODE_DESC = "Stage2(MiT)"

        self.MODEL_NAME = f'{self.ENCODER_NAME}_unet{self.MODE_SUFFIX}'
        self.ENCODER_WEIGHTS = 'imagenet'  #  ImageNet 
        self.IN_CHANNELS = 3
        self.USE_DROPOUT = use_dropout
        self.optimizer_type = optimizer_type

        if use_dropout:
            self.MODEL_NAME += '_dropout'


        self.SAVE_DIR = os.path.join(self.BASE_DIR, 'mit_b0_stage2_results')
        self.STAGE1_MODEL_PATH = os.path.join(
            PROJECT_ROOT,
            f'stage1 mit-{mit_size} (binary classification) training codes',
            f'mit_{mit_size}_stage1_results',
            f'best_mit_{mit_size}_stage1.pth',
        )

    # 
    BATCH_SIZE = 4
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    IMAGE_SIZE = (512, 512)
    PRIOR_FOREGROUND_WEIGHT = 1.0
    PRIOR_BACKGROUND_WEIGHT = 0.2
    PRIOR_UNCERTAINTY_BAND = 0.15
    GLOBAL_LOSS_WEIGHT = 0.2
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =====================  =====================
class Stage2Dataset(Dataset):
    """Stage 2 :"""

    def __init__(self, image_dir, annotation_file, config, transform=None, is_train=True):
        self.image_dir = image_dir
        self.config = config
        self.transform = transform
        self.is_train = is_train

        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.coco_data = json.load(f)

        self.image_id_to_annotations = {}
        self.image_id_to_info = {}

        for img in self.coco_data['images']:
            self.image_id_to_info[img['id']] = img

        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append(ann)

        #  -  1=, 2=, 3=
        self.category_id_to_name = {}
        self.category_id_mapping = {}

        for cat in self.coco_data['categories']:
            orig_id = cat['id']
            name = cat['name']
            self.category_id_to_name[orig_id] = name

            if name == 'single layer':
                self.category_id_mapping[orig_id] = 1
            elif name == 'contaminants':
                self.category_id_mapping[orig_id] = 2
            elif name == 'multilayer':
                self.category_id_mapping[orig_id] = 3

        self.valid_image_ids = list(self.image_id_to_annotations.keys())
        self.valid_image_ids.sort()
        print(f" {len(self.valid_image_ids)} ")

    def __len__(self):
        return len(self.valid_image_ids)

    def __getitem__(self, idx):
        image_id = self.valid_image_ids[idx]
        img_info = self.image_id_to_info[image_id]

        # 
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        try:
            image = np.array(Image.open(img_path).convert('RGB'))
        except (FileNotFoundError, OSError) as e:
            print(f" {img_path},: {e}")
            image = np.zeros((512, 512, 3), dtype=np.uint8)

        # ============================================================
        #  mask
        # label 0 = ()
        # label 1 = ()
        # label 2 = 
        # label 3 = 
        # ============================================================
        mask = np.zeros((512, 512), dtype=np.uint8)  #  label 0 = 
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

                            if category_id == 1:  # single layer → mask
                                single_layer_mask = np.maximum(single_layer_mask, temp_mask_array)
                            elif category_id == 2:  # contaminants → mask
                                contaminants_mask = np.maximum(contaminants_mask, temp_mask_array)
                            elif category_id == 3:  # multilayer → mask
                                multilayer_mask = np.maximum(multilayer_mask, temp_mask_array)

        #  =  -  - (/)
        single_layer_clean = single_layer_mask.copy()
        single_layer_clean[(multilayer_mask > 0) | (contaminants_mask > 0)] = 0

        mask = np.where(multilayer_mask > 0, 3, mask)        # 
        mask = np.where(contaminants_mask > 0, 2, mask)     # 
        mask = np.where(single_layer_clean > 0, 1, mask)    # 
        #  0 = 

        # 512x512
        if image.shape[:2] != (512, 512):
            image = cv2.resize(image, (512, 512))

        # 
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask.long()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).long()

        return image, mask

# ===================== MiT + U-Net  =====================
class MiTUNet(nn.Module):
    """MiT + U-Net"""

    def __init__(self, encoder_name='mit_b0', encoder_weights=None, in_channels=3, classes=4, activation=None):
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
            print(f"MiT {encoder_name} : {encoder_channels}")

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

# =====================  =====================
def create_model(config):
    """MiT + U-Net"""
    try:
        print(f" {config.MODEL_NAME} ...")
        print(f" MiT {config.ENCODER_NAME} ")

        model = MiTUNet(
            encoder_name=config.ENCODER_NAME,
            encoder_weights=config.ENCODER_WEIGHTS,
            in_channels=config.IN_CHANNELS,
            classes=config.CLASSES,
            activation=config.ACTIVATION
        )

        if config.CHECKPOINT:
            if os.path.exists(config.CHECKPOINT):
                print(f"checkpoint: {config.CHECKPOINT}")
                checkpoint = torch.load(config.CHECKPOINT, map_location='cpu')

                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint

                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    print(f": {len(missing_keys)} ")
                if unexpected_keys:
                    print(f": {len(unexpected_keys)} ")
                print("Checkpoint")
            else:
                print(f":Checkpoint: {config.CHECKPOINT}")

        print(f"{config.MODEL_NAME} ")
        return model

    except Exception as e:
        raise RuntimeError(f" {config.ENCODER_NAME} : {e}") from e


def load_stage1_model(config):
    """Load and freeze the matching Stage 1 model for online prior inference."""
    checkpoint_path = config.STAGE1_MODEL_PATH
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {checkpoint_path}")

    model = MiTUNet(
        encoder_name=config.ENCODER_NAME,
        encoder_weights=None,
        in_channels=config.IN_CHANNELS,
        classes=1,
        activation='sigmoid',
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']
    model.load_state_dict(checkpoint, strict=True)
    model = model.to(config.DEVICE)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print(f"Stage 1 model loaded for online priors: {checkpoint_path}")
    return model

# =====================  =====================
def train_model(model, stage1_model, train_loader, val_loader, config):
    """"""
    model = model.to(config.DEVICE)
    stage1_model = stage1_model.to(config.DEVICE)
    stage1_model.eval()

    # 
    def combined_loss(outputs, targets, stage1_prior):
        """
        :
        label 0 = :  1
        label 1 = :  10
        label 2 = :  50
        label 3 = :  10
        """
        targets_all = targets.squeeze(1)  # [B, H, W]
        stage1_prior = stage1_prior.to(outputs.device).float()
        if stage1_prior.ndim == 4:
            stage1_prior = stage1_prior.squeeze(1)
        stage1_prior = torch.clamp(stage1_prior, 0.0, 1.0)

        # 4: (0)=1, (1)=10, (2)=50, (3)=10
        #  stage_2.py 
        class_weights = torch.tensor([1.0, 10.0, 50.0, 10.0], device=outputs.device)
        fg_w = getattr(config, "PRIOR_FOREGROUND_WEIGHT", 1.0)
        bg_w = getattr(config, "PRIOR_BACKGROUND_WEIGHT", 0.2)
        unc_band = getattr(config, "PRIOR_UNCERTAINTY_BAND", 0.15)
        lambda_global = getattr(config, "GLOBAL_LOSS_WEIGHT", 0.2)

        raw_pixel_weight = fg_w * stage1_prior + bg_w * (1.0 - stage1_prior)
        confidence = torch.abs(stage1_prior - 0.5)
        confidence = torch.clamp((confidence - unc_band) / (0.5 - unc_band + 1e-6), 0.0, 1.0)
        pixel_weight = 1.0 + (raw_pixel_weight - 1.0) * confidence
        pixel_weight = pixel_weight / (pixel_weight.mean() + 1e-6)

        # 1) Weighted CrossEntropy + global CE anchor
        ce_map = F.cross_entropy(outputs, targets_all, weight=class_weights, reduction='none')
        ce_loss_weighted = (ce_map * pixel_weight).mean()
        ce_loss_global = F.cross_entropy(outputs, targets_all, weight=class_weights, reduction='mean')

        # 2) Dice Loss (4)
        probs = torch.softmax(outputs, dim=1)
        num_classes = outputs.shape[1]
        dice_loss = 0.0
        for c in range(num_classes):
            pred_c = probs[:, c, :, :]
            target_c = (targets_all == c).float()
            intersection = (pred_c * target_c * pixel_weight).sum()
            union = (pred_c * pixel_weight).sum() + (target_c * pixel_weight).sum()
            dice = (2. * intersection + 1e-6) / (union + 1e-6)
            dice_loss += 1 - dice
        dice_loss = dice_loss / num_classes

        # 3) Weighted Focal-like loss
        log_probs = F.log_softmax(outputs, dim=1)
        probs_for_focal = log_probs.exp()
        gather_index = targets_all.unsqueeze(1)
        log_pt = torch.gather(log_probs, 1, gather_index).squeeze(1)
        pt = torch.gather(probs_for_focal, 1, gather_index).squeeze(1)
        focal_map = -((1.0 - pt).pow(2.0)) * log_pt
        focal_loss = (focal_map * pixel_weight).mean()

        # 4) FP : (2)
        impurity_class_id = 2
        p_imp = probs[:, impurity_class_id, :, :]
        neg_imp = (targets_all != impurity_class_id)
        neg_imp_prob = p_imp.clone()
        neg_imp_prob = torch.clamp(neg_imp_prob, min=1e-6, max=1 - 1e-6)
        fp_penalty = (
            -torch.log(1 - neg_imp_prob) * neg_imp.float() * pixel_weight
        ).sum() / ((neg_imp.float() * pixel_weight).sum() + 1e-6)

        lambda_fp = 0.3
        local_loss = dice_loss + focal_loss + ce_loss_weighted + lambda_fp * fp_penalty
        total_loss = (1.0 - lambda_global) * local_loss + lambda_global * ce_loss_global
        return total_loss

    # :
    def predict_with_confidence_threshold(outputs, imp_threshold=0.6, imp_margin=0.15):
        """
        : argmax,
        label 0=, label 1=, label 2=, label 3=
        """
        probs = torch.softmax(outputs, dim=1)
        pred_masks = torch.argmax(outputs, dim=1)

        p_impurity = probs[:, 2, :, :]
        abs_pass = p_impurity > imp_threshold
        rel_pass = (p_impurity - probs[:, 0, :, :]) > imp_margin
        keep_impurity = abs_pass & rel_pass

        pred_masks_fixed = pred_masks.clone()
        pred_masks_fixed[(pred_masks == 2) & ~keep_impurity] = 0  # →

        return pred_masks_fixed

    # 
    if hasattr(config, 'optimizer_type') and config.optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                               weight_decay=config.WEIGHT_DECAY, betas=(0.9, 0.999))
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                              weight_decay=config.WEIGHT_DECAY)

    # 
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
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    min_delta = 1e-4

    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': [],
        'train_dice': [], 'val_dice': []
    }

    best_model_path = os.path.join(config.SAVE_DIR, 'best_mit_b0_stage2.pth')

    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")

        # 
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        train_dice = 0.0

        for batch_data in tqdm(train_loader, desc="Training"):
            images, masks = batch_data
            images = images.to(config.DEVICE)
            masks = masks.to(config.DEVICE)

            with torch.no_grad():
                stage1_prior = stage1_model(images)

            optimizer.zero_grad()
            outputs = model(images)
            loss = combined_loss(outputs, masks, stage1_prior)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # 3 IoU/Dice
            with torch.no_grad():
                pred_masks = predict_with_confidence_threshold(outputs,
                                                                imp_threshold=0.6, imp_margin=0.15)
                masks_squeezed = masks.squeeze(1)

                # label 0=, label 1=, label 2=, label 3=
                # ""
                batch_iou = []
                batch_dice = []

                for class_id in range(4):
                    pred_c = (pred_masks == class_id).float()
                    mask_c = (masks_squeezed == class_id).float()
                    inter = (pred_c * mask_c).sum(dim=(1, 2))
                    union = pred_c.sum(dim=(1, 2)) + mask_c.sum(dim=(1, 2)) - inter
                    iou = (inter / (union + 1e-6)).mean().item()
                    dice = (2 * inter / (pred_c.sum(dim=(1, 2)) + mask_c.sum(dim=(1, 2)) + 1e-6)).mean().item()
                    batch_iou.append(iou)
                    batch_dice.append(dice)

                # ""(class 1)
                train_iou += batch_iou[1]
                train_dice += batch_dice[1]

        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        train_dice /= len(train_loader)

        # 
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc="Validation"):
                images, masks = batch_data
                images = images.to(config.DEVICE)
                masks = masks.to(config.DEVICE)

                stage1_prior = stage1_model(images)

                outputs = model(images)
                loss = combined_loss(outputs, masks, stage1_prior)
                val_loss += loss.item()

                pred_masks = predict_with_confidence_threshold(outputs,
                                                                imp_threshold=0.6, imp_margin=0.15)
                masks_squeezed = masks.squeeze(1)

                batch_iou = []
                batch_dice = []

                for class_id in range(4):
                    pred_c = (pred_masks == class_id).float()
                    mask_c = (masks_squeezed == class_id).float()
                    inter = (pred_c * mask_c).sum(dim=(1, 2))
                    union = pred_c.sum(dim=(1, 2)) + mask_c.sum(dim=(1, 2)) - inter
                    iou = (inter / (union + 1e-6)).mean().item()
                    dice = (2 * inter / (pred_c.sum(dim=(1, 2)) + mask_c.sum(dim=(1, 2)) + 1e-6)).mean().item()
                    batch_iou.append(iou)
                    batch_dice.append(dice)

                val_iou += batch_iou[1]
                val_dice += batch_dice[1]

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_dice /= len(val_loader)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)

        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, Val Dice: {val_dice:.4f}")

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f": {best_model_path}")
        else:
            patience_counter += 1
            print(f",: {patience_counter}/{patience}")

        if (epoch + 1) % 10 == 0:
            print(f"... (Epoch {epoch + 1})")
            history_path = os.path.join(config.SAVE_DIR, 'training_history.png')
            plot_training_history(history, history_path)

        excel_path = os.path.join(config.SAVE_DIR, 'training_history.xlsx')
        save_history_to_excel(history, excel_path)

        if patience_counter >= patience:
            print(f"！{patience}epoch")
            print(f": {best_val_loss:.4f}")
            break

    return history

# =====================  =====================
def plot_training_history(history, save_path):
    """"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history['train_iou'], label='Train IoU')
    axes[0, 1].plot(history['val_iou'], label='Val IoU')
    axes[0, 1].set_title('Training and Validation IoU')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(history['train_dice'], label='Train Dice')
    axes[1, 0].plot(history['val_dice'], label='Val Dice')
    axes[1, 0].set_title('Training and Validation Dice')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(history['val_iou'], label='Val IoU', color='blue')
    axes[1, 1].plot(history['val_dice'], label='Val Dice', color='red')
    axes[1, 1].set_title('Validation Metrics Comparison')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ===================== Excel =====================
def save_history_to_excel(history, save_path):
    """Excel"""
    if not OPENPYXL_AVAILABLE:
        print(": openpyxl,Excel")
        return

    wb = Workbook()
    ws = wb.active
    ws.title = "Training History"

    headers = ['Epoch', 'Train Loss', 'Val Loss', 'Train IoU', 'Val IoU', 'Train Dice', 'Val Dice']
    ws.append(headers)

    num_epochs = len(history['train_loss'])
    for epoch in range(num_epochs):
        row = [
            epoch + 1,
            history['train_loss'][epoch] if epoch < len(history['train_loss']) else None,
            history['val_loss'][epoch] if epoch < len(history['val_loss']) else None,
            history['train_iou'][epoch] if epoch < len(history['train_iou']) else None,
            history['val_iou'][epoch] if epoch < len(history['val_iou']) else None,
            history['train_dice'][epoch] if epoch < len(history['train_dice']) else None,
            history['val_dice'][epoch] if epoch < len(history['val_dice']) else None,
        ]
        ws.append(row)

    ws.append([])
    ws.append(['Best Metrics'])
    ws.append(['Best Val Loss', min(history['val_loss']) if history['val_loss'] else None])
    ws.append(['Best Val IoU', max(history['val_iou']) if history['val_iou'] else None])
    ws.append(['Best Val Dice', max(history['val_dice']) if history['val_dice'] else None])

    wb.save(save_path)
    print(f"Excel: {save_path}")


# =====================  =====================
def main():
    """"""
    args = parse_args()
    config = Config(
        checkpoint=args.checkpoint,
        mit_size=args.mit_size,
        use_dropout=args.dropout,
        optimizer_type=args.optimizer
    )

    config.EPOCHS = args.epochs
    config.PRIOR_FOREGROUND_WEIGHT = args.prior_fg_weight
    config.PRIOR_BACKGROUND_WEIGHT = args.prior_bg_weight
    config.PRIOR_UNCERTAINTY_BAND = args.prior_uncertainty_band
    config.GLOBAL_LOSS_WEIGHT = args.global_loss_weight

    print(f" MiT-{args.mit_size} + U-Net {config.MODE_DESC} ")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")

    print(f": label 0=, label 1=, label 2=, label 3=")
    print(f": {config.CLASSES}")
    print(f": {config.ACTIVATION}")
    print(f": {config.DATASET_ROOT}")
    print(f": {config.MODEL_NAME}")
    print(f": {config.IMAGE_SIZE}")
    print(f": {config.DEVICE}")
    print(f": {config.SAVE_DIR}")
    print(f"Stage 1 checkpoint: {config.STAGE1_MODEL_PATH}")
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    # 
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=15, p=0.3),
        A.RandomScale(scale_limit=0.1, p=0.3),
        A.Resize(height=config.IMAGE_SIZE[0], width=config.IMAGE_SIZE[1]),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussNoise(p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(height=config.IMAGE_SIZE[0], width=config.IMAGE_SIZE[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # 
    print("\n...")

    train_dataset = Stage2Dataset(
        image_dir=os.path.join(config.DATASET_ROOT, 'training dataset'),
        annotation_file=os.path.join(config.DATASET_ROOT, 'annotation COCO dataset', 'instances_train2019.json'),
        config=config,
        transform=train_transform,
        is_train=True
    )

    val_dataset = Stage2Dataset(
        image_dir=os.path.join(config.DATASET_ROOT, 'validation dataset'),
        annotation_file=os.path.join(config.DATASET_ROOT, 'annotation COCO dataset', 'instances_val2019.json'),
        config=config,
        transform=val_transform,
        is_train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    print(f": {len(train_dataset)} ")
    print(f": {len(val_dataset)} ")

    # 
    model = create_model(config)
    model = model.to(config.DEVICE)
    stage1_model = load_stage1_model(config)

    # 
    print("\n...")
    print(":Dice + Focal + CE + FP,")
    history = train_model(model, stage1_model, train_loader, val_loader, config)

    # 
    final_model_path = os.path.join(config.SAVE_DIR, 'final_mit_b0_stage2.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f": {final_model_path}")

    print("\n！")
    print(f": {min(history['val_loss']):.4f}")
    print(f"IoU: {max(history['val_iou']):.4f}")
    print(f"Dice: {max(history['val_dice']):.4f}")

if __name__ == "__main__":
    main()
    
# # MiT-B0()
# python stage_2_mit.py

# # MiT-B2
# python stage_2_mit.py --mit_size b2

# python stage_2_contrast_mit.py --epochs 100

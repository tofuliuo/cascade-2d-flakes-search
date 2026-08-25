"""
MiT + U-Net 

"""

import os
# HF_HOME,timm

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
matplotlib.use('Agg')  # GUI,Qt
import matplotlib.pyplot as plt
from tqdm import tqdm
import segmentation_models_pytorch as smp
from sklearn.metrics import precision_score, recall_score, f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
from datetime import datetime
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
    parser = argparse.ArgumentParser(description='MiT + U-Net ')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='checkpoint (pth)')
    parser.add_argument('--epochs', type=int, default=50,
                        help=' (: 50)')
    parser.add_argument('--multi_class', action='store_true',
                       help=' (+++), (vs)')
    parser.add_argument('--include_multilayer', action='store_true',
                       help=' ()')
    parser.add_argument('--include_contaminants', action='store_true',
                       help=' ()')
    parser.add_argument('--mit_size', type=str, default='b2',
                       choices=['b0', 'b2'],
                       help='MiT (: b0)')
    parser.add_argument('--dropout', action='store_true',
                       help='Dropout')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'],
                       help=' (: adamw)')
    return parser.parse_args()

# =====================  =====================
class Config:
    def __init__(self, checkpoint=None, multi_class=False, include_multilayer=False, include_contaminants=False,
                 mit_size='b0', use_dropout=False, optimizer_type='adam'):
        # ()
        self.BASE_DIR = BASE_DIR

        # (BASE_DIR)
        self.DATASET_ROOT = os.path.join(PROJECT_ROOT, "image datasets")

        # Checkpoint
        self.CHECKPOINT = checkpoint

        # MiT
        self.MIT_SIZE = mit_size
        if mit_size == 'b0':
            self.ENCODER_NAME = 'mit_b0'
        else:  # b2
            self.ENCODER_NAME = 'mit_b2'

        # 
        self.MODEL_NAME = f'{self.ENCODER_NAME}_unet'
        self.ENCODER_WEIGHTS = 'imagenet'
        self.IN_CHANNELS = 3
        self.USE_DROPOUT = use_dropout
        self.optimizer_type = optimizer_type

        # 
        self.multi_class = multi_class
        self.include_multilayer = include_multilayer
        self.include_contaminants = include_contaminants

        if multi_class:
            # : +  +  + 
            self.CLASSES = 4
            self.ACTIVATION = 'softmax'
            self.NUM_CLASSES = 4
            self.MODE_SUFFIX = "_multiclass"
            self.MODE_DESC = ""
        else:
            # : vs 
            self.CLASSES = 1
            self.ACTIVATION = 'sigmoid'
            self.NUM_CLASSES = 2
            self.MODE_SUFFIX = "_binary"
            self.MODE_DESC = ""

        self.SAVE_DIR = os.path.join(self.BASE_DIR, 'mit_b2_stage1_results')

    # 
    BATCH_SIZE = 4
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # 
    IMAGE_SIZE = (512, 512)

    # 
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =====================  =====================
class GrapheneDataset(Dataset):
    """Graphene:"""

    def __init__(self, image_dir, annotation_file, config, transform=None, is_train=True):
        self.image_dir = image_dir
        self.config = config
        self.transform = transform
        self.is_train = is_train

        # COCO
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.coco_data = json.load(f)

        # ID
        self.image_id_to_annotations = {}
        self.image_id_to_info = {}

        for img in self.coco_data['images']:
            self.image_id_to_info[img['id']] = img

        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append(ann)

        # (,)
        self.category_id_to_name = {}
        for cat in self.coco_data['categories']:
            self.category_id_to_name[cat['id']] = cat['name']

        # ID()
        self.valid_image_ids = []
        for img_id in self.image_id_to_annotations.keys():
            img_info = self.image_id_to_info[img_id]
            img_path = os.path.join(self.image_dir, img_info['file_name'])
            if os.path.exists(img_path):
                self.valid_image_ids.append(img_id)
            else:
                print(f":: {img_path}")

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
            print(f": {img_path},: {e}")
            # 
            image = np.zeros((512, 512, 3), dtype=np.uint8)

        # mask
        if self.config.multi_class:
            # :(0) + (1) + (2) + (3)
            mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        else:
            # :(0) vs (1)
            mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

        # ,mask
        if np.all(image == 0):
            # ,mask
            pass  # mask0

        if image_id in self.image_id_to_annotations:
            for ann in self.image_id_to_annotations[image_id]:
                category_id = ann['category_id']
                category_name = self.category_id_to_name.get(category_id, '')

                # 
                if self.config.multi_class:
                    # :IDmask
                    # =0, =1, =2, =3
                    if category_name == 'single layer':
                        mask_value = 1
                    elif category_name == 'contaminants':
                        mask_value = 2
                    elif category_name == 'multilayer':
                        mask_value = 3
                    else:
                        continue  # 
                else:
                    # :
                    is_foreground = False
                    if category_name == 'single layer':
                        is_foreground = True
                    elif category_name == 'multilayer' and self.config.include_multilayer:
                        is_foreground = True
                    elif category_name == 'contaminants' and self.config.include_contaminants:
                        is_foreground = True

                    if not is_foreground:
                        continue

                    mask_value = 1  # 
                    # COCOsegmentationmask
                    if 'segmentation' in ann and ann['segmentation']:
                        for seg in ann['segmentation']:
                            if len(seg) >= 6:  # 3
                                # 
                                poly_coords = []
                                for i in range(0, len(seg), 2):
                                    poly_coords.append((int(seg[i]), int(seg[i+1])))

                                # mask
                                from PIL import ImageDraw
                                temp_mask = Image.new('L', (img_info['width'], img_info['height']), 0)
                                draw = ImageDraw.Draw(temp_mask)
                                draw.polygon(poly_coords, outline=mask_value, fill=mask_value)
                                temp_mask_array = np.array(temp_mask)

                                # mask
                                if self.config.multi_class:
                                    # :,
                                    mask = np.maximum(mask, temp_mask_array)
                                else:
                                    # :
                                    mask = np.maximum(mask, temp_mask_array)

        # mask512x512
        if image.shape[:2] != (512, 512):
            image = cv2.resize(image, (512, 512))
        if mask.shape[:2] != (512, 512):
            mask = cv2.resize(mask.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)

        # 
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            # maskfloat
            mask = mask.float()
            return image, mask
        else:
            # 
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()

            return image, mask

# ===================== MiT + U-Net  =====================
class MiTUNet(nn.Module):
    """MiT + U-Net - ConvNeXtSkip Connections"""

    def __init__(self, encoder_name='mit_b0', encoder_weights=None, in_channels=3, classes=1, activation='sigmoid'):
        super().__init__()

        # SMPMiT,forward6:
        # [, dummy(1/2), 1/4, 1/8, 1/16, 1/32]
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights
        )

        # 
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, 512, 512)
            features = self.encoder(dummy_input)[2:]
            encoder_channels = [f.shape[1] for f in features]
            feature_sizes = [f.shape[2:] for f in features]
            print(f"MiT {encoder_name} : {encoder_channels}")
            print(f"MiT {encoder_name} : {feature_sizes}")

        # skip connections
        self.encoder_channels = encoder_channels

        #  - Skip ConnectionsU-Net
        # : features[0]=128x128, features[1]=64x64, features[2]=32x32, features[3]=16x16
        # : 16x16->32x32, 32x32->64x64, 64x64->128x128, 128x128->256x256, 256x256->512x512

        # 1: 16x16 -> 32x32 (features[2])
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[3], encoder_channels[2], kernel_size=2, stride=2),
            nn.BatchNorm2d(encoder_channels[2]),
            nn.ReLU(inplace=True)
        )
        # : 384+384=768 -> 384
        self.decoder1_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[2] * 2, encoder_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_channels[2]),
            nn.ReLU(inplace=True)
        )

        # 2: 32x32 -> 64x64 (features[1])
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[2], encoder_channels[1], kernel_size=2, stride=2),
            nn.BatchNorm2d(encoder_channels[1]),
            nn.ReLU(inplace=True)
        )
        # : 192+192=384 -> 192
        self.decoder2_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[1] * 2, encoder_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_channels[1]),
            nn.ReLU(inplace=True)
        )

        # 3: 64x64 -> 128x128 (features[0])
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[1], encoder_channels[0], kernel_size=2, stride=2),
            nn.BatchNorm2d(encoder_channels[0]),
            nn.ReLU(inplace=True)
        )
        # : 96+96=192 -> 96
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

        # 
        self.final_conv = nn.Conv2d(encoder_channels[0] // 4, classes, kernel_size=1)

        # 
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        else:
            self.activation = None

    def forward(self, x):
        # ,4
        # features[0]: 128x128, features[1]: 64x64, features[2]: 32x32, features[3]: 16x16
        features = self.encoder(x)[2:]

        #  - Skip Connections
        # 
        x = features[3]  # 16x16, 768 channels

        # 1: 16x16 -> 32x32,features[2]
        x = self.decoder1(x)
        x = torch.cat([x, features[2]], dim=1)  # 384 + 384 = 768 channels
        x = self.decoder1_conv(x)  # 768 -> 384

        # 2: 32x32 -> 64x64,features[1]
        x = self.decoder2(x)
        x = torch.cat([x, features[1]], dim=1)  # 192 + 192 = 384 channels
        x = self.decoder2_conv(x)  # 384 -> 192

        # 3: 64x64 -> 128x128,features[0]
        x = self.decoder3(x)
        x = torch.cat([x, features[0]], dim=1)  # 96 + 96 = 192 channels
        x = self.decoder3_conv(x)  # 192 -> 96

        # 4: 128x128 -> 256x256
        x = self.decoder4(x)

        # 5: 256x256 -> 512x512
        x = self.decoder5(x)

        # 
        x = self.final_conv(x)
        if self.activation:
            x = self.activation(x)

        return x

# =====================  =====================
def create_model(config):
    """MiT + U-Net"""
    try:
        print(f" {config.MODEL_NAME} ...")
        print(f"MiT + U-Net: {config.ENCODER_NAME}")

        model = MiTUNet(
            encoder_name=config.ENCODER_NAME,
            encoder_weights=config.ENCODER_WEIGHTS,
            in_channels=config.IN_CHANNELS,
            classes=config.CLASSES,
            activation=config.ACTIVATION
        )

        # checkpoint,
        if config.CHECKPOINT:
            if os.path.exists(config.CHECKPOINT):
                print(f"checkpoint: {config.CHECKPOINT}")
                checkpoint = torch.load(config.CHECKPOINT, map_location='cpu')

                # checkpoint
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint

                # 
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

# =====================  =====================
def train_model(model, train_loader, val_loader, config):
    """"""
    model = model.to(config.DEVICE)

    # 
    if config.multi_class:
        # :
        dice_loss = smp.losses.DiceLoss(mode='multiclass')
        ce_loss = nn.CrossEntropyLoss()

        # Focal Loss
        try:
            focal_loss = smp.losses.FocalLoss(mode='multiclass', alpha=0.25, gamma=2.0)
            combined_loss = lambda pred, target: 0.5 * dice_loss(pred, target) + 0.3 * focal_loss(pred, target) + 0.2 * ce_loss(pred, target)
        except:
            combined_loss = lambda pred, target: 0.7 * dice_loss(pred, target) + 0.3 * ce_loss(pred, target)

        # 
        # class_weights = torch.tensor([0.1, 0.8, 3.0, 2.5]).to(config.DEVICE)  # , , , 
        # ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    else:
        # :
        dice_loss = smp.losses.DiceLoss(mode='binary')
        bce_loss = nn.BCEWithLogitsLoss()

        # Focal LossTversky Loss
        try:
            focal_loss = smp.losses.FocalLoss(mode='binary', alpha=0.25, gamma=2.0)
            tversky_loss = smp.losses.TverskyLoss(mode='binary', alpha=0.3, beta=0.7)
            combined_loss = lambda pred, target: 0.4 * dice_loss(pred, target) + 0.3 * focal_loss(pred, target) + 0.2 * tversky_loss(pred, target) + 0.1 * bce_loss(pred, target)
        except:
            combined_loss = lambda pred, target: 0.6 * dice_loss(pred, target) + 0.4 * bce_loss(pred, target)

    # 
    if not config.multi_class:
        binary_loss = bce_loss  # 

    # 
    if hasattr(config, 'optimizer_type') and config.optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                               weight_decay=config.WEIGHT_DECAY, betas=(0.9, 0.999))
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                              weight_decay=config.WEIGHT_DECAY)

    #  - 
    warmup_epochs = 5
    total_epochs = config.EPOCHS

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # :
            return (epoch + 1) / warmup_epochs
        else:
            # 
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 
    best_val_loss = float('inf')
    patience = 15  # 15epoch
    patience_counter = 0
    min_delta = 1e-4  # 

    # 
    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': [],
        'train_dice': [], 'val_dice': []
    }

    best_val_loss = float('inf')
    best_model_path = os.path.join(config.SAVE_DIR, 'best_mit_b2_stage1.pth')

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
            # mask [B, 1, H, W]
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)

            if config.multi_class:
                #  - 
                loss = combined_loss(outputs, masks.long())
            else:
                #  - 
                # masks[B, 1, H, W],outputs
                loss = combined_loss(outputs, masks)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # IoUDice
            if config.multi_class:
                # :IoU ()
                pred_masks = torch.argmax(outputs, dim=1)  # [B, H, W]
                masks = masks.long()  # long

                # IoU
                iou_per_class = []
                dice_per_class = []

                for class_idx in range(config.NUM_CLASSES):
                    pred_class = (pred_masks == class_idx).float()
                    mask_class = (masks == class_idx).float()

                    intersection = (pred_class * mask_class).sum(dim=(1,2))
                    union = pred_class.sum(dim=(1,2)) + mask_class.sum(dim=(1,2)) - intersection

                    # IoU,
                    iou_batch = (intersection / (union + 1e-6)).mean().item()
                    dice_batch = (2 * intersection / (pred_class.sum(dim=(1,2)) + mask_class.sum(dim=(1,2)) + 1e-6)).mean().item()

                    iou_per_class.append(iou_batch)
                    dice_per_class.append(dice_batch)

                # 
                iou = sum(iou_per_class) / len(iou_per_class)
                dice = sum(dice_per_class) / len(dice_per_class)
            else:
                # 
                pred_masks = (outputs > 0.5).float()
                intersection = (pred_masks * masks).sum(dim=(1,2,3))
                union = pred_masks.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3)) - intersection
                iou = (intersection / (union + 1e-6)).mean().item()
                dice = (2 * intersection / (pred_masks.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3)) + 1e-6)).mean().item()

            train_iou += iou
            train_dice += dice

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
                # mask [B, 1, H, W]
                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)

                outputs = model(images)
                if config.multi_class:
                    loss = combined_loss(outputs, masks.long())
                else:
                    loss = combined_loss(outputs, masks)

                val_loss += loss.item()

                # IoUDice
                if config.multi_class:
                    # :IoU ()
                    pred_masks = torch.argmax(outputs, dim=1)  # [B, H, W]
                    masks = masks.long()  # long

                    # IoU
                    iou_per_class = []
                    dice_per_class = []

                    for class_idx in range(config.NUM_CLASSES):
                        pred_class = (pred_masks == class_idx).float()
                        mask_class = (masks == class_idx).float()

                        intersection = (pred_class * mask_class).sum(dim=(1,2))
                        union = pred_class.sum(dim=(1,2)) + mask_class.sum(dim=(1,2)) - intersection

                        # IoU,
                        iou_batch = (intersection / (union + 1e-6)).mean().item()
                        dice_batch = (2 * intersection / (pred_class.sum(dim=(1,2)) + mask_class.sum(dim=(1,2)) + 1e-6)).mean().item()

                        iou_per_class.append(iou_batch)
                        dice_per_class.append(dice_batch)

                    # 
                    iou = sum(iou_per_class) / len(iou_per_class)
                    dice = sum(dice_per_class) / len(dice_per_class)
                else:
                    # 
                    pred_masks = (outputs > 0.5).float()
                    intersection = (pred_masks * masks).sum(dim=(1,2,3))
                    union = pred_masks.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3)) - intersection
                    iou = (intersection / (union + 1e-6)).mean().item()
                    dice = (2 * intersection / (pred_masks.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3)) + 1e-6)).mean().item()

                val_iou += iou
                val_dice += dice

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_dice /= len(val_loader)

        # 
        scheduler.step()

        # 
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)

        print(".4f")
        print(".4f")
        print(".4f")
        # 
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f": {best_model_path}")
        else:
            patience_counter += 1
            print(f",: {patience_counter}/{patience}")

        # 10epoch
        if (epoch + 1) % 10 == 0:
            print(f"... (Epoch {epoch + 1})")
            history_path = os.path.join(config.SAVE_DIR, 'training_history.png')
            plot_training_history(history, history_path)
            print(f": {history_path}")

        # epochExcel
        excel_path = os.path.join(config.SAVE_DIR, 'training_history.xlsx')
        save_history_to_excel(history, excel_path)

        # 
        if patience_counter >= patience:
            print(f"！{patience}epoch")
            print(".4f")
            break

    return history

# =====================  =====================
def plot_training_history(history, save_path):
    """"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # IoU
    axes[0, 1].plot(history['train_iou'], label='Train IoU')
    axes[0, 1].plot(history['val_iou'], label='Val IoU')
    axes[0, 1].set_title('Training and Validation IoU')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Dice
    axes[1, 0].plot(history['train_dice'], label='Train Dice')
    axes[1, 0].plot(history['val_dice'], label='Val Dice')
    axes[1, 0].set_title('Training and Validation Dice')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 
    axes[1, 1].plot(history['val_iou'], label='Val IoU', color='blue')
    axes[1, 1].plot(history['val_dice'], label='Val Dice', color='red')
    axes[1, 1].set_title('Validation Metrics Comparison')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 

# ===================== Excel =====================
def save_history_to_excel(history, save_path):
    """Excel"""
    if not OPENPYXL_AVAILABLE:
        print(": openpyxl,Excel")
        return

    wb = Workbook()
    ws = wb.active
    ws.title = "Training History"

    # 
    headers = ['Epoch', 'Train Loss', 'Val Loss', 'Train IoU', 'Val IoU', 'Train Dice', 'Val Dice']
    ws.append(headers)

    # 
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

    # 
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
        multi_class=args.multi_class,
        include_multilayer=args.include_multilayer,
        include_contaminants=args.include_contaminants,
        mit_size=args.mit_size,
        use_dropout=args.dropout,
        optimizer_type=args.optimizer
    )

    config.EPOCHS = args.epochs

    mode_name = "" if args.checkpoint else ""
    print(f"{mode_name}MiT-{args.mit_size} + U-Net{config.MODE_DESC}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")

    if config.multi_class:
        print(f": (0) + (1) + (2) + (3)")
        print(f": {config.CLASSES}")
        print(f": {config.ACTIVATION}")
    else:
        foreground_classes = [""]
        if config.include_multilayer:
            foreground_classes.append("")
        if config.include_contaminants:
            foreground_classes.append("")
        print(f": {', '.join(foreground_classes)}")
        print(f": {config.CLASSES} ({config.MODE_DESC})")
        print(f": {config.ACTIVATION}")

    print(f": {config.DATASET_ROOT}")
    print(f": {config.MODEL_NAME}")
    print(f": {config.IMAGE_SIZE}")
    print(f": {config.DEVICE}")
    print(f": {config.SAVE_DIR}")
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    # 
    train_transform = A.Compose([
        # 
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=15, p=0.3),

        # 
        A.RandomScale(scale_limit=0.1, p=0.3),
        A.Resize(height=config.IMAGE_SIZE[0], width=config.IMAGE_SIZE[1]),

        # ()
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),

        # 
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussNoise(p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.1),

        # 
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()  # mask[512, 512]
    ])

    val_transform = A.Compose([
        A.Resize(height=config.IMAGE_SIZE[0], width=config.IMAGE_SIZE[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(transpose_mask=True)  # mask
    ])

    # 
    print("\n...")

    # 
    train_dataset = GrapheneDataset(
        image_dir=os.path.join(config.DATASET_ROOT, 'training dataset'),
        annotation_file=os.path.join(config.DATASET_ROOT, 'annotation COCO dataset', 'instances_train2019.json'),
        config=config,
        transform=train_transform,
        is_train=True
    )

    # 
    val_dataset = GrapheneDataset(
        image_dir=os.path.join(config.DATASET_ROOT, 'validation dataset'),
        annotation_file=os.path.join(config.DATASET_ROOT, 'annotation COCO dataset', 'instances_val2019.json'),
        config=config,
        transform=val_transform,
        is_train=False
    )

    # 
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    print(f": {len(train_dataset)} ")
    print(f": {len(val_dataset)} ")

    # 
    model = create_model(config)
    model = model.to(config.DEVICE)

    # 
    print("\n...")
    print(" Dice + Focal + CE ")
    history = train_model(model, train_loader, val_loader, config)

    # 
    final_model_path = os.path.join(config.SAVE_DIR, 'final_mit_b2_stage1.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f": {final_model_path}")

    print("\n！")
    print(f": {min(history['val_loss']):.4f}")
    print(f"IoU: {max(history['val_iou']):.4f}")
    print(f"Dice: {max(history['val_dice']):.4f}")

if __name__ == "__main__":
    main()

"""
:

# 
python stage_1.py

# MiT-B2
python stage_1_mit.py --mit_size b2


"""

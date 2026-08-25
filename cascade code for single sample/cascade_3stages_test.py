import os
import sys

# Get base directory (script directory or executable directory)
def get_base_dir():
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        return os.path.dirname(sys.executable)
    else:
        # Running as script
        return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_base_dir()
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Set timm model cache path to use local cached models
# This avoids downloading models from the internet
# HF_HOME should point to the parent directory containing models--timm--xxx
os.environ['HF_HOME'] = PROJECT_ROOT
os.environ['HF_DATASETS_CACHE'] = os.path.join(PROJECT_ROOT, 'image datasets')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(PROJECT_ROOT, 'models--timm--convnext_tiny.in12k_ft_in1k')

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import glob
import argparse
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
try:
    from skimage.morphology import skeletonize
    HAS_SKIMAGE = True
except:
    HAS_SKIMAGE = False


# ===================== Configuration =====================
class Config:
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = BASE_DIR
        
        self.BASE_DIR = base_dir
        
        # Model paths
        self.STAGE1_MODEL_PATH = os.path.join(PROJECT_ROOT, 'stage1 (binary classification) training codes', 'convnext_tiny_stage1_results', 'best_convnext_tiny_stage1.pth')
        # Stage 2: 4-class model (0=background, 1=single layer, 2=contaminant, 3=multilayer)
        self.STAGE2_MODEL_PATH = os.path.join(PROJECT_ROOT, 'stage2 (ternary classification) training codes', 'convnext_tiny_stage2_results', 'best_convnext_tiny_stage2.pth')
        self.STAGE2_CLASSES = 4  # background (0) + single layer (1) + contaminant (2) + multilayer (3)
        
        # Data paths
        self.PROCESSING_DIR = os.path.join(BASE_DIR, 'BF and DF images for tests')
        self.PREDICT_DIR = os.path.join(self.PROCESSING_DIR, 'BF')
        self.PREDICT_DARK_DIR = os.path.join(self.PROCESSING_DIR, 'DF')
        
        # Output directory structure
        self.OUTPUT_DIR = os.path.join(base_dir, 'output')
        self.VISUALIZATION_OUTPUT = os.path.join(self.OUTPUT_DIR, 'visualizations')

        # Stage 1 output
        self.STAGE1_OUTPUT = os.path.join(self.VISUALIZATION_OUTPUT, 'foreground')

        # Stage 2 outputs (single/contaminant/multilayer)
        self.STAGE2_SINGLE_OUTPUT = os.path.join(self.VISUALIZATION_OUTPUT, 'single_layer')
        self.STAGE2_OVERLAY_OUTPUT = os.path.join(self.VISUALIZATION_OUTPUT, 'stage2_overlay')
        
        # Stage 3: Post-processing
        self.POSTPROCESS_OUTPUT = os.path.join(self.OUTPUT_DIR, 'postprocess_results')
        
        # Device
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Parameters
        self.BLUE_THRESHOLD = 80


# ===================== Model Definition =====================
class ConvNeXtUNet(nn.Module):
    def __init__(self, encoder_name='convnext_tiny', in_channels=3, classes=1, activation='sigmoid'):
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm not available")
        
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
            nn.Conv2d(encoder_channels[2] * 2, encoder_channels[2], 3, padding=1),
            nn.BatchNorm2d(encoder_channels[2]), nn.ReLU(inplace=True)
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[2], encoder_channels[1], 2, 2),
            nn.BatchNorm2d(encoder_channels[1]), nn.ReLU(inplace=True)
        )
        self.decoder2_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[1] * 2, encoder_channels[1], 3, padding=1),
            nn.BatchNorm2d(encoder_channels[1]), nn.ReLU(inplace=True)
        )
        
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[1], encoder_channels[0], 2, 2),
            nn.BatchNorm2d(encoder_channels[0]), nn.ReLU(inplace=True)
        )
        self.decoder3_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[0] * 2, encoder_channels[0], 3, padding=1),
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


# ===================== Helper Functions =====================
def load_model(model_path, classes=1, activation='sigmoid', device='cpu'):
    """Load model from checkpoint"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model: {model_path}")
    
    model = ConvNeXtUNet(
        encoder_name='convnext_tiny',
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
    
    print("Model loaded successfully")
    return model


def get_transform():
    """Get image preprocessing transform"""
    return A.Compose([
        A.Resize(height=512, width=512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


# ===================== Stage 1: Foreground Mask Generation =====================
def generate_foreground_masks(config):
    """Generate foreground masks using Stage 1 model"""
    print("\n" + "="*60)
    print("Stage 1: Generating Foreground Masks")
    print("="*60)
    
    # Check for existing masks
    os.makedirs(config.STAGE1_OUTPUT, exist_ok=True)
    existing_masks = glob.glob(os.path.join(config.STAGE1_OUTPUT, '*_foreground.png'))
    if existing_masks:
        print(f"Found {len(existing_masks)} existing foreground masks, skipping generation")
        return True
    
    # Load model
    model = load_model(config.STAGE1_MODEL_PATH, classes=1, activation='sigmoid', device=config.DEVICE)
    
    # Get image list
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(config.PREDICT_DIR, ext)))
    
    if not image_files:
        print(f"Error: No images found in {config.PREDICT_DIR}")
        return False
    
    print(f"Found {len(image_files)} images")
    
    transform = get_transform()
    
    for image_path in tqdm(image_files, desc="Generating foreground masks"):
        try:
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            
            if image.shape[:2] != (512, 512):
                image = cv2.resize(image, (512, 512))
            
            transformed = transform(image=image)
            image_tensor = transformed['image'].unsqueeze(0).to(config.DEVICE)
            
            with torch.no_grad():
                output = model(image_tensor)
                pred_mask = (output > 0.5).float().squeeze().cpu().numpy().astype(np.uint8)
            
            filename = os.path.basename(image_path)
            name_without_ext = os.path.splitext(filename)[0]
            mask_filename = name_without_ext + '_foreground.png'
            mask_path = os.path.join(config.STAGE1_OUTPUT, mask_filename)
            
            mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8), mode='L')
            mask_img.save(mask_path)

            # Also save a transparent RGBA overlay on top of the original image
            overlay_rgb = image.copy()
            fg = pred_mask == 1
            overlay_rgb[fg] = (
                overlay_rgb[fg].astype(np.float32) * 0.5 + np.array([0, 255, 0], dtype=np.float32) * 0.5
            ).astype(np.uint8)
            alpha = np.full(pred_mask.shape, 255, dtype=np.uint8)
            alpha[fg] = 180
            rgba = np.dstack([overlay_rgb, alpha])
            overlay_filename = name_without_ext + '_foreground_overlay.png'
            overlay_path = os.path.join(config.STAGE1_OUTPUT, overlay_filename)
            Image.fromarray(rgba, mode='RGBA').save(overlay_path)
            
        except Exception as e:
            print(f"Failed {os.path.basename(image_path)}: {e}")
    
    print("Stage 1 complete")
    return True


# ===================== Stage 2: 4-Class Inference =====================
def stage2_inference(config):
    """
    Stage 2: 4-class inference
    Labels: 0=background, 1=single layer, 2=contaminant, 3=multilayer
    """
    print("\n" + "="*60)
    print("Stage 2: 4-Class Segmentation Inference")
    print("Classes: 0=Background, 1=Single Layer, 2=Contaminant, 3=Multilayer")
    print("Single output = Stage1 foreground - contaminant - multilayer (inside foreground)")
    print("="*60)
    
    # Load Stage 1 model (foreground)
    stage1_model = load_model(
        config.STAGE1_MODEL_PATH,
        classes=1,
        activation='sigmoid',
        device=config.DEVICE
    )

    # Load Stage 2 4-class model
    model = load_model(
        config.STAGE2_MODEL_PATH,
        classes=config.STAGE2_CLASSES,
        activation=None,
        device=config.DEVICE
    )
    
    os.makedirs(config.STAGE2_SINGLE_OUTPUT, exist_ok=True)
    os.makedirs(config.STAGE2_OVERLAY_OUTPUT, exist_ok=True)
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(config.PREDICT_DIR, ext)))
    
    if not image_files:
        print(f"Error: No images found in {config.PREDICT_DIR}")
        return False
    
    print(f"Found {len(image_files)} images")
    
    transform = get_transform()
    
    
    for image_path in tqdm(image_files, desc="Stage 2 inference"):
        try:
            filename = os.path.basename(image_path)
            name_without_ext = os.path.splitext(filename)[0]
            
            # Read image
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            
            if image.shape[:2] != (512, 512):
                image = cv2.resize(image, (512, 512))
            
            transformed = transform(image=image)
            image_tensor = transformed['image'].to(config.DEVICE)
            
            # Inference
            with torch.no_grad():
                image_tensor = image_tensor.unsqueeze(0)

                # Stage 1 foreground
                stage1_output = stage1_model(image_tensor)
                stage1_fg = (stage1_output.squeeze() > 0.5).float().cpu().numpy().astype(np.uint8)

                # Stage 2 class prediction
                output = model(image_tensor)
                probs = torch.softmax(output, dim=1)
                pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            
            
            # Foreground-constrained impurity/multilayer masks
            impurity_mask = ((pred_mask == 2).astype(np.uint8) & stage1_fg).astype(np.uint8)
            multilayer_mask = ((pred_mask == 3).astype(np.uint8) & stage1_fg).astype(np.uint8)

            # Single layer = Stage1 foreground - impurity - multilayer
            single_layer_mask = (
                stage1_fg.astype(bool)
                & ~impurity_mask.astype(bool)
                & ~multilayer_mask.astype(bool)
            ).astype(np.uint8)
            
            # Save single-layer result (blue) for post-processing
            single_filename = name_without_ext + '_stage2_pred.png'
            single_path = os.path.join(config.STAGE2_SINGLE_OUTPUT, single_filename)
            single_color = np.zeros((512, 512, 3), dtype=np.uint8)
            single_color[single_layer_mask == 1] = [0, 0, 255]  # Blue = single layer
            Image.fromarray(single_color).save(single_path)

            # Save a 3-class color overlay segmentation map (single layer/contaminant/multilayer)
            overlay = image.copy()
            overlay[single_layer_mask == 1] = (
                overlay[single_layer_mask == 1].astype(np.float32) * 0.5
                + np.array([0, 0, 255], dtype=np.float32) * 0.5
            ).astype(np.uint8)  # Blue = Single
            overlay[impurity_mask == 1] = (
                overlay[impurity_mask == 1].astype(np.float32) * 0.5
                + np.array([255, 0, 255], dtype=np.float32) * 0.5
            ).astype(np.uint8)  # Magenta = Contaminant
            overlay[multilayer_mask == 1] = (
                overlay[multilayer_mask == 1].astype(np.float32) * 0.5
                + np.array([0, 255, 0], dtype=np.float32) * 0.5
            ).astype(np.uint8)  # Green = Multilayer
            overlay_filename = name_without_ext + '_stage2_overlay.png'
            overlay_path = os.path.join(config.STAGE2_OVERLAY_OUTPUT, overlay_filename)
            Image.fromarray(overlay).save(overlay_path)
            
        except Exception as e:
            print(f"Failed {os.path.basename(image_path)}: {e}")
    
    print("Stage 2 complete")
    print(f"  - Single layer mask: {config.STAGE2_SINGLE_OUTPUT}")
    print(f"  - 3-class overlay: {config.STAGE2_OVERLAY_OUTPUT}")
    return True


# ===================== Visualization: Comparison Charts =====================
def generate_visualizations(config):
    """
    Generate visualization panels - 4-class version + Stage1-Stage2 comparison
    Labels: 0=background (gray), 1=single layer (blue), 2=contaminant (purple), 3=multilayer (green)
    """
    print("\n" + "="*60)
    print("Generating 4-Class Visualization + Stage1-Stage2 Comparison")
    print("="*60)
    
    os.makedirs(config.VISUALIZATION_OUTPUT, exist_ok=True)
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(config.PREDICT_DIR, ext)))
    
    # Filter out non-image files
    image_files = [f for f in image_files if not f.endswith('_foreground.png') 
                   and not f.endswith('_stage2_pred.png') and not f.endswith('_stage1_minus_stage2.png')]
    
    if not image_files:
        print("No images found")
        return
    
    print(f"Found {len(image_files)} images")
    
    transform = get_transform()
    
    # Load models
    stage1_model = load_model(
        config.STAGE1_MODEL_PATH,
        classes=1,
        activation='sigmoid',
        device=config.DEVICE
    )
    
    stage2_model = load_model(
        config.STAGE2_MODEL_PATH,
        classes=config.STAGE2_CLASSES,
        activation=None,
        device=config.DEVICE
    )
    
    stage1_threshold = 0.5
    
    for image_path in tqdm(image_files, desc="Generating visualizations"):
        try:
            filename = os.path.basename(image_path)
            name_without_ext = os.path.splitext(filename)[0]
            
            # Load original image
            image = Image.open(image_path).convert('RGB')
            original_image = np.array(image)
            
            if original_image.shape[:2] != (512, 512):
                original_image = cv2.resize(original_image, (512, 512))
            
            transformed = transform(image=original_image)
            image_tensor = transformed['image'].to(config.DEVICE)
            
            with torch.no_grad():
                image_batch = image_tensor.unsqueeze(0)
                
                # Stage 1: Foreground
                stage1_output = stage1_model(image_batch)
                stage1_pred = (stage1_output.squeeze() > stage1_threshold).float().cpu().numpy().astype(np.uint8)
                
                # Stage 2: 4-class
                output = stage2_model(image_batch)
                probs = torch.softmax(output, dim=1)
                pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            
            
            # 4-class masks
            single_layer_mask = (pred_mask == 1).astype(np.uint8)
            impurity_mask = (pred_mask == 2).astype(np.uint8)
            multilayer_mask = (pred_mask == 3).astype(np.uint8)
            
            # Stage1 - Stage2 = single layer: foreground - contaminant - multilayer
            stage1_minus_stage2 = stage1_pred.copy()
            stage1_minus_stage2[(pred_mask == 2) | (pred_mask == 3)] = 0
            
            # Save Stage1-Stage2 result
            s12_filename = name_without_ext + '_stage1_minus_stage2.png'
            s12_path = os.path.join(config.VISUALIZATION_OUTPUT, s12_filename)
            s12_image = np.zeros((512, 512, 3), dtype=np.uint8)
            s12_image[stage1_minus_stage2 == 1] = [0, 255, 0]  # Green = single layer
            Image.fromarray(s12_image).save(s12_path)
            
            # Overlay helper
            def create_overlay(img, mask, color):
                overlay = img.copy()
                mask_3ch = np.stack([mask, mask, mask], axis=-1)
                overlay = np.where(mask_3ch > 0,
                                   (overlay * 0.5 + np.array(color) * 0.5).astype(np.uint8),
                                   overlay)
                return overlay
            
            # 2. Stage 2: 4-class result
            combined = original_image.copy()
            combined[single_layer_mask == 1] = (combined[single_layer_mask == 1] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)
            combined[multilayer_mask == 1] = (combined[multilayer_mask == 1] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
            combined[impurity_mask == 1] = (combined[impurity_mask == 1] * 0.5 + np.array([255, 0, 255]) * 0.5).astype(np.uint8)
            plt.figure(figsize=(12, 10))
            plt.imshow(combined)
            plt.title('Stage 2: 4-Class\nBlue=Single, Green=Multilayer, Magenta=Contaminant',
                     fontsize=14, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(config.VISUALIZATION_OUTPUT, f'{name_without_ext}_classification.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            # 3. Stage1 - Stage2 = Single Layer (Green)
            s12_overlay = create_overlay(original_image, stage1_minus_stage2, [0, 255, 0])
            plt.figure(figsize=(12, 10))
            plt.imshow(s12_overlay)
            plt.title('Stage1 - Stage2: Single Layer (Green)\n(Foreground - Contaminant - Multilayer)', 
                     fontsize=14, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(config.VISUALIZATION_OUTPUT, f'{name_without_ext}_output.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Visualization failed {filename}: {e}")
    
    print(f"Visualization complete, saved to: {config.VISUALIZATION_OUTPUT}")


# ===================== Post-Processing: Boundary Removal =====================
def extract_dark_field_from_zong(dark_img, blue_threshold, clip_limit=3.0,
                                 tile_grid_size=(8, 8)):
    """Port the active DF enhancement and contour filling from zong.py."""
    blue_ch, green_ch, red_ch = cv2.split(dark_img)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    blue_enhanced = clahe.apply(blue_ch)
    _, blue_binary = cv2.threshold(
        blue_enhanced, blue_threshold, 255, cv2.THRESH_BINARY
    )

    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(
        blue_binary, cv2.MORPH_CLOSE, kernel, iterations=1
    )
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros_like(closed)
    for contour in contours:
        if cv2.contourArea(contour) > 4:
            cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel)

    return red_ch, green_ch, blue_ch, blue_enhanced, blue_binary, filled


def postprocess_single_layer(config):
    """Post-process: Remove boundaries using dark field images"""
    print("\n" + "="*60)
    print("Stage 3: Post-processing (Boundary Removal)")
    print("="*60)
    
    # Check for dark field images (both .jpeg and .jpg)
    dark_images = []
    for ext in ['*.jpeg', '*.jpg', '*.JPEG', '*.JPG']:
        dark_images.extend(glob.glob(os.path.join(config.PREDICT_DARK_DIR, ext)))
    
    if not dark_images:
        print("Warning: No dark field images found, skipping post-processing")
        return False
    
    # Create output subdirectories
    enhanced_dir = os.path.join(config.POSTPROCESS_OUTPUT, 'enhanced_channels')
    boundary_dir = os.path.join(config.POSTPROCESS_OUTPUT, 'dark_boundaries')
    final_dir = os.path.join(config.POSTPROCESS_OUTPUT, 'final_results')
    overlay_dir = os.path.join(config.POSTPROCESS_OUTPUT, 'visualization_overlay')
    
    os.makedirs(enhanced_dir, exist_ok=True)
    os.makedirs(boundary_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    
    # Get stage 2 single-layer predictions
    pred_files = glob.glob(os.path.join(config.STAGE2_SINGLE_OUTPUT, '*_stage2_pred.png'))
    
    if not pred_files:
        print("Error: No Stage 2 predictions found")
        return False
    
    print(f"Found {len(pred_files)} predictions to post-process")
    
    for pred_path in tqdm(pred_files, desc="Post-processing"):
        try:
            filename = os.path.basename(pred_path)
            name_without_ext = os.path.splitext(filename)[0].replace('_stage2_pred', '')
            
            # Corresponding dark field image (check both .jpeg and .jpg)
            dark_path = None
            for ext in ['.jpeg', '.jpg', '.JPEG', '.JPG']:
                potential_path = os.path.join(config.PREDICT_DARK_DIR, f'{name_without_ext}{ext}')
                if os.path.exists(potential_path):
                    dark_path = potential_path
                    break
            
            if dark_path is None:
                continue
            
            # ===== Step 1: Read and enhance dark field blue channel =====
            dark_img = cv2.imread(dark_path)
            if dark_img is None:
                print(f"Warning: Unable to read dark-field image: {dark_path}")
                continue

            red_ch, green_ch, blue_ch, blue_enhanced, blue_binary, filled = (
                extract_dark_field_from_zong(dark_img, config.BLUE_THRESHOLD)
            )
            
            # Save original channels
            cv2.imwrite(os.path.join(enhanced_dir, f'{name_without_ext}_red.jpg'), red_ch)
            cv2.imwrite(os.path.join(enhanced_dir, f'{name_without_ext}_green.jpg'), green_ch)
            cv2.imwrite(os.path.join(enhanced_dir, f'{name_without_ext}_blue_original.jpg'), blue_ch)
            cv2.imwrite(os.path.join(enhanced_dir, f'{name_without_ext}_blue_enhanced.jpg'), blue_enhanced)
            cv2.imwrite(os.path.join(enhanced_dir, f'{name_without_ext}_blue_binary.jpg'), blue_binary)
            
            # Save filled result (for debugging)
            cv2.imwrite(os.path.join(boundary_dir, f'{name_without_ext}_filled.jpg'), filled)
            
            # ===== Step 3: Read single layer prediction =====
            pred_img = cv2.imread(pred_path)
            single_layer = pred_img[:, :, 0]  # Blue channel
            
            # Ensure single_layer has the same size as filled (from dark field processing)
            if single_layer.shape != filled.shape:
                single_layer = cv2.resize(single_layer, (filled.shape[1], filled.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
            
            # ===== Step 4: Subtract boundary =====
            # Step 4a: Binarize filled and single layer
            _, filled_binary = cv2.threshold(filled, 127, 255, cv2.THRESH_BINARY)
            _, single_binary = cv2.threshold(single_layer, 127, 255, cv2.THRESH_BINARY)
            
            # Step 4b: Intersection = filled AND single_layer (only process where both exist)
            intersection = cv2.bitwise_and(filled_binary, single_binary)
            cv2.imwrite(os.path.join(boundary_dir, f'{name_without_ext}_intersection.jpg'), intersection)
            
            # Step 4c: Subtract blue binary boundary from intersection
            blue_binary_inv = cv2.bitwise_not(blue_binary)
            final_result = cv2.bitwise_and(intersection, blue_binary_inv)
            cv2.imwrite(os.path.join(boundary_dir, f'{name_without_ext}_subtracted.jpg'), final_result)
            
            # Remove small white components without filling internal DF boundaries.
            num_cc, labels, stats, _ = cv2.connectedComponentsWithStats(
                final_result, connectivity=8
            )
            cleaned_result = np.zeros_like(final_result)
            for component_id in range(1, num_cc):
                if stats[component_id, cv2.CC_STAT_AREA] > 50:
                    cleaned_result[labels == component_id] = 255
            
            # Step 7: Extra subtraction of stage2 predicted blue regions
            # If final is white but stage2 predicts blue, remove it
            # Condition: final white(255) AND stage2 not blue(0) → invert(final) = black(0)
            # Other cases keep original
            stage2_img = cv2.imread(pred_path)
            if stage2_img is not None and len(stage2_img.shape) == 3:
                stage2_blue = stage2_img[:, :, 0]  # Blue channel = single layer
                
                # Ensure same size
                if stage2_blue.shape != cleaned_result.shape:
                    stage2_blue = cv2.resize(stage2_blue, (cleaned_result.shape[1], cleaned_result.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)
                
                _, stage2_binary = cv2.threshold(stage2_blue, 127, 255, cv2.THRESH_BINARY)
                
                stage2_inv = cv2.bitwise_not(stage2_binary)
                condition = cv2.bitwise_and(cleaned_result, stage2_inv)  # positions meeting condition are white(255)
                final_result = cleaned_result.copy()
                final_result[condition == 255] = 0  # set black for positions meeting condition
            else:
                final_result = cleaned_result
            
            # Save final result
            final_path = os.path.join(final_dir, f'{name_without_ext}_final.png')
            cv2.imwrite(final_path, final_result)
            
            # ===== Step 5: Create visualization overlay =====
            # Align both arrays to the same size before blending.
            target_h, target_w = cleaned_result.shape[:2]
            dark_resized = cv2.resize(dark_img, (target_w, target_h))
            mask_color = cv2.cvtColor(cleaned_result, cv2.COLOR_GRAY2BGR)
            if mask_color.shape[:2] != dark_resized.shape[:2]:
                mask_color = cv2.resize(mask_color, (dark_resized.shape[1], dark_resized.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
            mask_color[:, :, 1] = 0  # Green
            mask_color[:, :, 2] = 0  # Red, keep blue
            
            overlay = cv2.addWeighted(dark_resized, 0.7, mask_color, 0.3, 0)
            overlay_path = os.path.join(overlay_dir, f'{name_without_ext}_overlay.jpg')
            cv2.imwrite(overlay_path, overlay)
            
        except Exception as e:
            print(f"Post-process failed {filename}: {e}")
    
    print(f"Post-processing complete")
    print(f"  - Enhanced channels: {enhanced_dir}")
    print(f"  - Dark boundaries: {boundary_dir}")
    print(f"  - Final results: {final_dir}")
    print(f"  - Visualization overlay: {overlay_dir}")
    return True


# ===================== Main Function =====================
def main():
    import sys
    
    parser = argparse.ArgumentParser(description='Graphene Single Layer Segmentation Pipeline')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3], default=None,
                        help='Run a single sub-task: 1=foreground, 2=single/contaminant/multilayer, 3=postprocess')
    parser.add_argument('--skip-stage1', action='store_true', help='Skip Stage 1 (foreground masks already exist)')
    parser.add_argument('--no-postprocess', action='store_true', help='Skip post-processing')
    parser.add_argument('--no-visualization', action='store_true', help='Skip visualization')
    parser.add_argument('--metrics', action='store_true', help='Run metrics evaluation (requires *_gt_single_layer.png masks)')
    parser.add_argument('--base-dir', type=str, default=None, help='Base directory (default: script directory)')
    parser.add_argument('--blue-threshold', type=int, default=80, help='Blue channel threshold for dark field')
    
    args = parser.parse_args()
    
    # Initialize config
    config = Config(args.base_dir)
    config.BLUE_THRESHOLD = args.blue_threshold
    
    print("="*60)
    print("Graphene Single Layer Segmentation Pipeline")
    print("="*60)
    print(f"Base directory: {config.BASE_DIR}")
    print(f"Device: {config.DEVICE}")
    print(f"Stage 1 model: {config.STAGE1_MODEL_PATH}")
    print(f"Stage 2 model: {config.STAGE2_MODEL_PATH} (4-class: 0=Background, 1=Single, 2=Contaminant, 3=Multilayer)")
    print(f"Input images: {config.PREDICT_DIR}")
    if args.stage is not None:
        print(f"Selected sub-task stage: {args.stage}")
    
    # Check models for relevant stages
    needs_stage1_model = args.stage in (None, 1, 2)
    needs_stage2_model = args.stage in (None, 2)

    if needs_stage1_model and not os.path.exists(config.STAGE1_MODEL_PATH):
        print(f"\nError: Stage 1 model not found: {config.STAGE1_MODEL_PATH}")
        return
    
    if needs_stage2_model and not os.path.exists(config.STAGE2_MODEL_PATH):
        print(f"\nError: Stage 2 model not found: {config.STAGE2_MODEL_PATH}")
        return
    
    # Check input directory
    if not os.path.exists(config.PREDICT_DIR):
        print(f"\nError: Input directory not found: {config.PREDICT_DIR}")
        print("Please put images in: datasets/processing/predict/")
        return
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Run single selected sub-task
    if args.stage == 1:
        generate_foreground_masks(config)
        print(f"\nStage 1 output: {config.STAGE1_OUTPUT}")
        return
    if args.stage == 2:
        stage2_inference(config)
        print(f"\nStage 2 outputs:")
        print(f"  - {config.STAGE2_SINGLE_OUTPUT}")
        print(f"  - {config.STAGE2_OVERLAY_OUTPUT}")
        return
    if args.stage == 3:
        postprocess_single_layer(config)
        print(f"\nStage 3 output: {config.POSTPROCESS_OUTPUT}")
        return
    
    # Execute pipeline
    if not args.skip_stage1:
        generate_foreground_masks(config)
    else:
        print("\nSkipping Stage 1")
    
    stage2_inference(config)
    
    if not args.no_visualization:
        generate_visualizations(config)
    else:
        print("\nSkipping visualization")
    
    if not args.no_postprocess:
        postprocess_single_layer(config)
    else:
        print("\nSkipping post-processing")
    

    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print(f"Output directory: {config.OUTPUT_DIR}")
    print(f"\nOutput structure:")
    print(f"  - output/visualizations/foreground   # Stage 1 foreground")
    print(f"  - output/visualizations/single_layer # Stage 2 single layer mask (for stage 3)")
    print(f"  - output/visualizations/stage2_overlay # Stage 2 colored overlay")
    print(f"  - output/postprocess_results/")
    print(f"      - enhanced_channels/              # Blue channel processing steps")
    print(f"      - dark_boundaries/                # Boundary extraction steps")
    print(f"      - final_results/                  # Final cleaned single layer")
    print(f"      - visualization_overlay/           # Overlay with dark field")


if __name__ == "__main__":
    main()

# python run_inference.py --stage 1
# python run_inference.py --stage 2
# python run_inference.py --stage 3

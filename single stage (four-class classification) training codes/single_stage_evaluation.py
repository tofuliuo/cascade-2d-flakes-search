"""Single-stage four-class comparison model using the original Figure 3 scale."""
import os
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import timm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve()
def seed_everything(seed):
    if seed is None:
        return  # Match original training: leave RNG and backend defaults untouched.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

def widths(scale):
    return [max(8, round(c * scale / 8) * 8) for c in (384, 192, 96, 48, 24)]

def decoder_parameters(ws, skip=True, classes=4):
    total, prev = 0, 768
    for i, w in enumerate(ws):
        extra = (384, 192, 96)[i] if skip and i < 3 else 0
        total += prev * w * 4 + w + 2 * w
        total += (w + extra) * w * 9 + w + 2 * w
        prev = w
    return total + prev * classes + classes

class Model(nn.Module):
    def __init__(self, variant, pretrained=False):
        super().__init__()
        self.encoder = timm.create_model('convnext_tiny.fb_in1k', pretrained=pretrained,
                                        features_only=True, out_indices=(0, 1, 2, 3))
        encoder_n = sum(p.numel() for p in self.encoder.parameters())
        self.target = 2 * encoder_n + decoder_parameters(widths(1), classes=1) + decoder_parameters(widths(1))
        if variant != 'A':
            raise ValueError('Only the original-scale single-stage comparison is supported')
        self.widths = widths(1)
        self.skip = True
        self.up, self.fuse = nn.ModuleList(), nn.ModuleList()
        prev = 768
        for i, w in enumerate(self.widths):
            extra = (384, 192, 96)[i] if self.skip and i < 3 else 0
            self.up.append(nn.Sequential(nn.ConvTranspose2d(prev, w, 2, 2), nn.BatchNorm2d(w), nn.ReLU(True)))
            self.fuse.append(nn.Sequential(nn.Conv2d(w + extra, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(True)))
            prev = w
        self.head = nn.Conv2d(prev, 4, 1)

    def forward(self, x):
        fs = self.encoder(x)
        x = fs[-1]
        for i, (up, fuse) in enumerate(zip(self.up, self.fuse)):
            x = up(x)
            if self.skip and i < 3:
                x = torch.cat((x, fs[2-i]), dim=1)
            x = fuse(x)
        return self.head(x)

def loss_fn(logits, target):
    """Original Stage 2 composite loss with spatial prior weights set to one."""
    weights = logits.new_tensor([1., 10., 50., 10.])
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    ce_local = F.cross_entropy(logits, target, weight=weights, reduction='none').mean()
    # Equivalent weighted mean without CUDA's nondeterministic NLL mean kernel.
    ce_global = ce_local * target.numel() / weights[target].sum()
    dice = 0.
    for c in range(4):
        gt = (target == c).float()
        dice += 1 - (2 * (p[:, c] * gt).sum() + 1e-6) / (p[:, c].sum() + gt.sum() + 1e-6)
    pt = p.gather(1, target[:, None]).squeeze(1)
    focal = (-((1-pt)**2) * logp.gather(1, target[:, None]).squeeze(1)).mean()
    neg = (target != 2).float()
    fp = (-torch.log(1-p[:, 2].clamp(1e-6, 1-1e-6)) * neg).sum() / (neg.sum()+1e-6)
    return .8 * (dice/4 + focal + ce_local + .3*fp) + .2*ce_global

class Stage2Dataset(Dataset):
    """: - ///"""

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

        #  -  1=, 2=, 3=
        # : 1=single layer, 2=contaminants, 3=multilayer
        self.category_id_to_name = {}
        self.category_id_mapping = {}  # ID -> ID (1,2,3)
        
        for cat in self.coco_data['categories']:
            orig_id = cat['id']
            name = cat['name']
            self.category_id_to_name[orig_id] = name
            
            # ID
            if name == 'single layer':
                self.category_id_mapping[orig_id] = 1  # 
            elif name == 'contaminants':
                self.category_id_mapping[orig_id] = 2  # 
            elif name == 'multilayer':
                self.category_id_mapping[orig_id] = 3  # 

        # ID(mask)
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
        # :0=, 1=(), 2=, 3=
        # ============================================================
        mask = np.zeros((512, 512), dtype=np.uint8)  # =0
        multilayer_mask = np.zeros((512, 512), dtype=np.uint8)
        contaminants_mask = np.zeros((512, 512), dtype=np.uint8)
        single_layer_mask = np.zeros((512, 512), dtype=np.uint8)

        # mask
        if image_id in self.image_id_to_annotations:
            for ann in self.image_id_to_annotations[image_id]:
                category_id = ann['category_id']
                # ID(ID)
                category_id = self.category_id_mapping.get(category_id, category_id)

                if 'segmentation' in ann and ann['segmentation']:
                    for seg in ann['segmentation']:
                        if len(seg) >= 6:  # 3
                            poly_coords = []
                            for i in range(0, len(seg), 2):
                                # 512x512
                                x = int(seg[i] * 512 / img_info['width'])
                                y = int(seg[i+1] * 512 / img_info['height'])
                                poly_coords.append((x, y))

                            # mask
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

        # mask512x512
        if image.shape[:2] != (512, 512):
            image = cv2.resize(image, (512, 512))

        # 
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask.long()
        else:
            # 
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).long()

        return image, mask

def dataset(split, seed=None):
    aug = []
    if split == 'training':
        aug = [A.HorizontalFlip(p=.5), A.VerticalFlip(p=.5), A.RandomRotate90(p=.5),
               A.Rotate(limit=15, p=.3), A.RandomScale(scale_limit=.1, p=.3),
               A.Resize(512, 512), A.ElasticTransform(alpha=1, sigma=50, p=.2),
               A.RandomBrightnessContrast(brightness_limit=.2, contrast_limit=.2, p=.3),
               A.GaussNoise(p=.2), A.GaussianBlur(blur_limit=3, p=.1)]
    transform = A.Compose(aug + [A.Resize(512, 512),
        A.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]), ToTensorV2()],
        **({'seed': seed} if seed is not None else {}))
    ann = ROOT / 'image datasets/annotation COCO dataset' / ('instances_train2019.json' if split == 'training' else 'instances_val2019.json')
    ds = Stage2Dataset(str(ROOT / 'image datasets' / (split+' dataset')), str(ann), SimpleNamespace(), transform)
    # Fail rather than silently substitute a black image, as the legacy reader does.
    from PIL import Image
    for image_id in ds.valid_image_ids:
        with Image.open(Path(ds.image_dir) / ds.image_id_to_info[image_id]['file_name']) as im:
            im.verify()
    return ds

def evaluate(model, loader, device):
    model.eval()
    rows, losses = [], []
    with torch.no_grad():
        for images, gt in loader:
            images, gt = images.to(device), gt.to(device)
            logits = model(images)
            losses.append(loss_fn(logits, gt).item())
            pred = logits.argmax(1)
            for a, b in zip(pred, gt):
                classes = []
                for c in range(4):
                    tp = ((a == c) & (b == c)).sum().item()
                    fp = ((a == c) & (b != c)).sum().item()
                    fn = ((a != c) & (b == c)).sum().item()
                    div = lambda n, d: n/d if d else 0.
                    classes.append(dict(iou=div(tp,tp+fp+fn), dice=div(2*tp,2*tp+fp+fn),
                                        precision=div(tp,tp+fp), recall=div(tp,tp+fn), f1=div(2*tp,2*tp+fp+fn)))
                rows.append(classes)
    if not rows:
        raise ValueError('Empty validation dataset')
    metrics = [{k: float(np.mean([r[c][k] for r in rows])) for k in rows[0][c]} for c in range(4)]
    return float(np.mean(losses)), dict(classes=metrics, single_layer=metrics[1], per_image=rows)

def save_evaluation_report(metrics, out):
    title = 'Single-stage ConvNeXt-Tiny: Direct Four-Class Segmentation = Single Layer'
    r = metrics['single_layer']
    line = (f"Single Layer: IoU={r['iou']:.4f}, Dice={r['dice']:.4f}, "
            f"P={r['precision']:.4f}, R={r['recall']:.4f}, F1={r['f1']:.4f}")
    text = '=' * 60 + '\n(' + title + ')\n' + '=' * 60 + '\n\n' + line + '\n'
    (out/'evaluation_results.txt').write_text(text, encoding='utf8')
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'convnext_tiny_single_stage_results')
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--seed', type=int, default=None, help='Optional; no fixed seed by default')
    args = parser.parse_args()
    seed_everything(args.seed)
    checkpoint_path = args.checkpoint or args.output/'best_convnext_tiny_single_stage.pth'
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint missing: {checkpoint_path}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model('A').to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt.get('state_dict', ckpt), strict=True)
    val = dataset('validation', args.seed)
    _, metrics = evaluate(model, DataLoader(val, batch_size=1, shuffle=False, num_workers=0), device)
    metrics['image_ids'] = val.valid_image_ids
    args.output.mkdir(parents=True, exist_ok=True)
    save_evaluation_report(metrics, args.output)

if __name__ == '__main__':
    main()

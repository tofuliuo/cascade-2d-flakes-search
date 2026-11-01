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
VARIANTS = ('A',)


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


def save_training_outputs(history, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from openpyxl import Workbook
    epochs = [r['epoch'] for r in history]
    series = {
        'train_loss': [r['train_loss'] for r in history],
        'val_loss': [r['val_loss'] for r in history],
        'train_iou': [r.get('train_iou', float('nan')) for r in history],
        'val_iou': [r['single_layer']['iou'] for r in history],
        'train_dice': [r.get('train_dice', float('nan')) for r in history],
        'val_dice': [r['single_layer']['dice'] for r in history],
    }
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    pairs = [('train_loss','val_loss','Loss'), ('train_iou','val_iou','IoU'),
             ('train_dice','val_dice','Dice'), ('val_iou','val_dice','Score')]
    for ax, (first, second, label) in zip(axes.flat, pairs):
        for key in (first, second):
            if np.isfinite(series[key]).any():
                ax.plot(epochs, series[key], label=key.replace('_',' ').title().replace('Iou','IoU'))
        ax.set_xlabel('Epoch'); ax.set_ylabel(label); ax.legend(); ax.grid(True)
        ax.set_title('Validation Metrics Comparison' if label=='Score' else 'Training and Validation '+label)
        if not np.isfinite(series[first]).any():
            ax.text(.5,.1, first.replace('_',' ').title()+' not recorded',transform=ax.transAxes,ha='center')
    fig.tight_layout(); fig.savefig(out/'training_history.png',dpi=300,bbox_inches='tight'); plt.close(fig)
    wb = Workbook(); ws = wb.active; ws.title = 'Training History'
    ws.append(['Epoch','Train Loss','Val Loss','Train IoU','Val IoU','Train Dice','Val Dice'])
    for i, epoch in enumerate(epochs):
        ws.append([epoch]+[float(series[k][i]) if np.isfinite(series[k][i]) else None for k in series])
    ws.append([]); ws.append(['Best Metrics'])
    ws.append(['Best Val Loss',min(series['val_loss'])]); ws.append(['Best Val IoU',max(series['val_iou'])]);ws.append(['Best Val Dice',max(series['val_dice'])])
    wb.save(out/'training_history.xlsx')



def save_evaluation_report(metrics, out):
    title = 'Single-stage ConvNeXt-Tiny: Direct Four-Class Segmentation = Single Layer'
    r = metrics['single_layer']
    line = (f"Single Layer: IoU={r['iou']:.4f}, Dice={r['dice']:.4f}, "
            f"P={r['precision']:.4f}, R={r['recall']:.4f}, F1={r['f1']:.4f}")
    text = '=' * 60 + '\n(' + title + ')\n' + '=' * 60 + '\n\n' + line + '\n'
    (out/'evaluation_results.txt').write_text(text, encoding='utf8')
    print(text)


def main(fixed_action=None, fixed_variant=None, result_dir=None):
    ap = argparse.ArgumentParser(description=__doc__)
    if fixed_action is None:
        ap.add_argument('action', choices=['profile', 'train', 'evaluate'])
        ap.add_argument('--variant', choices=VARIANTS, default='A')
    else:
        ap.set_defaults(action=fixed_action, variant=fixed_variant)
    ap.add_argument('--seed', type=int, default=None,
                    help='Optional fixed seed; omitted by default to match original experiments')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--scratch', action='store_true', help='Disable ImageNet initialization for all controls when comparing scratch runs')
    if result_dir is None:
        ap.add_argument('--output', type=Path, default=Path(__file__).parent / 'results')
    else:
        ap.add_argument('--output', type=Path, default=Path(result_dir), help='This experiment result directory')
    ap.add_argument('--checkpoint', type=Path, default=None, help='Evaluation checkpoint override')
    args = ap.parse_args()
    seed_everything(args.seed)
    if args.action == 'profile':
        report = {}
        for v in VARIANTS:
            m = Model(v)
            report[v] = dict(parameters=sum(p.numel() for p in m.parameters()), decoder_widths=m.widths,
                             skip=m.skip, cascade_parameter_target=m.target)
            del m
        print(json.dumps(report, indent=2))
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/'parameters.json').write_text(json.dumps(report, indent=2), encoding='utf8')
        return
    if args.epochs < 1 or args.batch_size < 1:
        ap.error('epochs and batch-size must be positive')
    out = args.output if result_dir is not None else args.output / args.variant / (f'seed_{args.seed}' if args.seed is not None else 'unseeded')
    checkpoint_path = args.checkpoint or out/'best_convnext_tiny_single_stage.pth'
    if args.action == 'evaluate' and not checkpoint_path.is_file():
        raise FileNotFoundError(f'Train this experiment first; checkpoint missing: {out}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(args.variant, pretrained=args.action == 'train' and not args.scratch).to(device)
    val = dataset('validation', args.seed)
    val_loader = DataLoader(val, batch_size=args.batch_size if args.action == 'train' else 1,
                            shuffle=False, num_workers=0)
    if args.action == 'evaluate':
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt.get('state_dict', ckpt), strict=True)
        out.mkdir(parents=True, exist_ok=True)
        _, metrics = evaluate(model, val_loader, device)
        metrics['image_ids'] = val.valid_image_ids
        save_evaluation_report(metrics, out)
        return
    train = dataset('training', args.seed)
    loader_options = {} if args.seed is None else {'generator': torch.Generator().manual_seed(args.seed)}
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0, **loader_options)
    out.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    schedule = lambda e: (e+1)/5 if e < 5 else .5*(1+math.cos(math.pi*(e-5)/max(1,args.epochs-5)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    best, stale, history = float('inf'), 0, []
    for epoch in range(args.epochs):
        model.train()
        total = 0.
        train_iou_total = train_dice_total = 0.
        for step, (images, gt) in enumerate(loader, 1):
            opt.zero_grad(set_to_none=True)
            images, gt = images.to(device), gt.to(device)
            logits = model(images)
            loss = loss_fn(logits, gt)
            with torch.no_grad():
                pred = logits.argmax(1) == 1
                truth = gt == 1
                inter = (pred & truth).sum(dim=(1,2)).float()
                union = (pred | truth).sum(dim=(1,2)).float()
                denom = pred.sum(dim=(1,2)) + truth.sum(dim=(1,2))
                train_iou_total += (inter/(union+1e-6)).mean().item()
                train_dice_total += (2*inter/(denom+1e-6)).mean().item()
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            opt.step()
            total += loss.item()
            if step == 1 or step % 20 == 0:
                print(f'epoch={epoch+1} step={step}/{len(loader)} loss={loss.item():.6f}', flush=True)
        val_loss, metrics = evaluate(model, val_loader, device)
        history.append(dict(epoch=epoch+1, train_loss=total/len(loader), val_loss=val_loss,
                            train_iou=train_iou_total/len(loader), train_dice=train_dice_total/len(loader),
                            lr=opt.param_groups[0]['lr'], single_layer=metrics['single_layer']))
        print(history[-1], flush=True)
        save_training_outputs(history, out)
        if val_loss < best-1e-4:
            best, stale = val_loss, 0
            torch.save(model.state_dict(), out/'best_convnext_tiny_single_stage.pth')
        else:
            stale += 1
        sched.step()
        if stale >= 15:
            break
    torch.save(model.state_dict(), out/'final_convnext_tiny_single_stage.pth')


if __name__ == '__main__':
    main(fixed_action='train', fixed_variant='A',
         result_dir=Path(__file__).resolve().parent/'convnext_tiny_single_stage_results')

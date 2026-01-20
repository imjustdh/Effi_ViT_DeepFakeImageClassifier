"""
Deepfake Detection - 48GB VRAM + 1시간 최적화 버전

=============================================================================
최적화 요약:
=============================================================================
VRAM: 48GB
  - batch_size=32 (checkpoint 불필요)
  - use_checkpoint=False → 속도 2배
  - num_workers=8

시간: 1시간 제한
  - max_train_samples=40,000 (15-20 에폭 가능)
  - patience=5 (빠른 early stopping)

이미지 증강 (평가 기준 핵심):
  - Deepfake 특화: JPEG, Blur, Downscale, Noise
  - 기하학적: Perspective, Rotation, Affine
  - 색상: ColorJitter, Grayscale, Contrast
  - 가림: GridMask, RandomErasing
  - 배치: MixUp, CutMix
=============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import io
import random
from datetime import datetime
import json
import time
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import warnings

# AMP
try:
    from torch.amp import autocast, GradScaler
    AMP_V2 = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AMP_V2 = False

# Pillow compatibility
try:
    BILINEAR = Image.Resampling.BILINEAR
    BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    BILINEAR = Image.BILINEAR
    BICUBIC = Image.BICUBIC

# 48GB optimized EfficientNet
from torchvision.models import resnet50

# torchvision 0.13+ (weights enum). Fallback for older versions.
try:
    from torchvision.models import ResNet50_Weights
except Exception:  # pragma: no cover
    ResNet50_Weights = None

# ============================================================================
# 48GB + 1시간 최적 설정
# ============================================================================

CONFIG = {
    # Model
    'num_classes': 2,

    # 48GB VRAM 최적화
    'batch_size': 32,
    'use_checkpoint': False,
    'num_workers': 8,

    # 1시간 제한 최적화
    'max_train_samples': 40000,
    'max_val_samples': 4000,
    'num_epochs': 20,
    'patience': 5,
    'warmup_epochs': 2,

    # Optimizer
    'learning_rate': 2e-4,
    'weight_decay': 1e-5,
    'label_smoothing': 0.1,

    # Image
    'image_size': 224,

    # Augmentation (강화)
    'aug_strength': 'strong',
    'use_mixup': True,
    'mixup_alpha': 0.4,
    'cutmix_alpha': 1.0,

    # Save
    'save_dir': 'checkpoints_48gb',
    'seed': 42,
}


# ============================================================================
# Seed
# ============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)

# ============================================================================
# Helpers
# ============================================================================

def setup_for_48gb():
    """48GB VRAM 환경에서 안전하게 성능을 끌어올리는 torch 설정."""
    # 로그를 지저분하게 만드는 torchvision 경고는 숨김(동작에는 영향 없음)
    try:
        import warnings as _warnings
        _warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
    except Exception:
        pass

    if torch.cuda.is_available():
        # 속도/정확도 균형: TF32 허용(암시적으로도 켜져 있을 수 있음)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

        # cudnn autotune
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        # torch 2.x matmul precision 힌트
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass


def get_model_info(model: nn.Module):
    """모델 파라미터 수 등 간단한 정보 반환."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'parameters': int(total),
        'trainable': int(trainable),
        'parameters_m': total / 1e6,
        'trainable_m': trainable / 1e6,
    }


# ============================================================================
# 이미지 증강 (평가 기준 대응 - 강화)
# ============================================================================

class JPEGCompression:
    """JPEG 압축 아티팩트 시뮬레이션"""
    def __init__(self, quality=(20, 100), p=0.6):
        self.quality = quality
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            try:
                q = random.randint(*self.quality)
                buf = io.BytesIO()
                img.save(buf, 'JPEG', quality=q)
                buf.seek(0)
                img = Image.open(buf)
                img.load()
                img = img.convert('RGB')
            except:
                pass
        return img


class GaussianBlur:
    """가우시안 블러"""
    def __init__(self, radius=(0.1, 3.0), p=0.4):
        self.radius = radius
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            try:
                r = random.uniform(*self.radius)
                img = img.filter(ImageFilter.GaussianBlur(r))
            except:
                pass
        return img


class Downscale:
    """해상도 저하 시뮬레이션"""
    def __init__(self, scale=(0.3, 0.9), p=0.4):
        self.scale = scale
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            try:
                w, h = img.size
                s = random.uniform(*self.scale)
                nw, nh = max(8, int(w*s)), max(8, int(h*s))
                img = img.resize((nw, nh), BILINEAR).resize((w, h), BICUBIC)
            except:
                pass
        return img


class Noise:
    """노이즈 추가"""
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            try:
                arr = np.array(img, dtype=np.float32)
                std = random.uniform(5, 25)
                arr += np.random.normal(0, std, arr.shape)
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                img = Image.fromarray(arr)
            except:
                pass
        return img


class Sharpen:
    """선명도 조절"""
    def __init__(self, factor=(0.3, 2.5), p=0.3):
        self.factor = factor
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            try:
                f = random.uniform(*self.factor)
                img = ImageEnhance.Sharpness(img).enhance(f)
            except:
                pass
        return img


class Perspective:
    """원근 변환"""
    def __init__(self, scale=0.15, p=0.3):
        self.scale = scale
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            try:
                w, h = img.size
                s = self.scale
                pts = [
                    [random.uniform(0, s*w/2), random.uniform(0, s*h/2)],
                    [random.uniform(w-s*w/2, w), random.uniform(0, s*h/2)],
                    [random.uniform(w-s*w/2, w), random.uniform(h-s*h/2, h)],
                    [random.uniform(0, s*w/2), random.uniform(h-s*h/2, h)]
                ]
                src = [[0,0], [w,0], [w,h], [0,h]]
                coeffs = self._coeffs(pts, src)
                img = img.transform((w, h), Image.PERSPECTIVE, coeffs, BICUBIC)
            except:
                pass
        return img

    def _coeffs(self, dst, src):
        matrix = []
        for s, d in zip(src, dst):
            matrix.append([d[0], d[1], 1, 0, 0, 0, -s[0]*d[0], -s[0]*d[1]])
            matrix.append([0, 0, 0, d[0], d[1], 1, -s[1]*d[0], -s[1]*d[1]])
        A = np.array(matrix, dtype=np.float64)
        B = np.array(src).flatten()
        return np.linalg.lstsq(A, B, rcond=None)[0].tolist()


class GridMask:
    """GridMask 가림"""
    def __init__(self, d=(20, 60), ratio=0.5, p=0.2):
        self.d = d
        self.ratio = ratio
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            try:
                w, h = img.size
                d = random.randint(*self.d)
                mask = np.ones((h, w), dtype=np.float32)
                for i in range(0, h, d):
                    for j in range(0, w, d):
                        mask[i:min(i+int(d*self.ratio), h),
                             j:min(j+int(d*self.ratio), w)] = 0
                arr = np.array(img, dtype=np.float32)
                for c in range(3):
                    arr[:,:,c] *= mask
                img = Image.fromarray(arr.astype(np.uint8))
            except:
                pass
        return img


def get_train_transforms(size=224):
    """학습용 변환 (강화된 증강)"""
    return transforms.Compose([
        transforms.Resize((size, size)),
        # Deepfake 특화
        JPEGCompression(quality=(20, 100), p=0.6),
        GaussianBlur(radius=(0.1, 3.0), p=0.4),
        Downscale(scale=(0.3, 0.9), p=0.4),
        Noise(p=0.3),
        Sharpen(factor=(0.3, 2.5), p=0.3),
        # 기하학적
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=10),
        Perspective(scale=0.15, p=0.3),
        # 색상
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.RandomGrayscale(0.1),
        # 가림
        GridMask(d=(20, 60), ratio=0.5, p=0.2),
        # Tensor
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def get_val_transforms(size=380):
    """검증용 변환"""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# ============================================================================
# MixUp / CutMix
# ============================================================================

def mixup(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam


def cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    idx = torch.randperm(x.size(0), device=x.device)

    W, H = x.size(2), x.size(3)
    cut = np.sqrt(1 - lam)
    cw, ch = int(W * cut), int(H * cut)
    cx, cy = np.random.randint(W), np.random.randint(H)

    x1 = np.clip(cx - cw//2, 0, W)
    y1 = np.clip(cy - ch//2, 0, H)
    x2 = np.clip(cx + cw//2, 0, W)
    y2 = np.clip(cy + ch//2, 0, H)

    x[:, :, x1:x2, y1:y2] = x[idx, :, x1:x2, y1:y2]
    lam = 1 - (x2-x1)*(y2-y1)/(W*H)

    return x, y, y[idx], lam


def mix_loss(criterion, pred, ya, yb, lam):
    return lam * criterion(pred, ya) + (1 - lam) * criterion(pred, yb)


# ============================================================================
# Dataset
# ============================================================================

class DeepfakeDataset(Dataset):
    def __init__(self, hf_data, transform, size=380):
        self.data = hf_data
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        try:
            item = self.data[i]
            img = item['image']
            if img.mode != 'RGB':
                img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, item['label']
        except:
            return torch.zeros(3, self.size, self.size), 0


def load_data(cfg):
    """데이터 로드 (1시간 최적화)"""
    from datasets import load_dataset

    print("=" * 70)
    print("Loading Dataset (1-Hour Optimized)")
    print("=" * 70)

    set_seed(cfg['seed'])

    ds = load_dataset("JamieWithofs/Deepfake-and-real-images")
    print(f"\nFull: Train={len(ds['train'])}, Val={len(ds['validation'])}, Test={len(ds['test'])}")

    # Balance & limit
    def sample(data, n, seed):
        random.seed(seed)
        labels = data['label']
        fake = [i for i, l in enumerate(labels) if l == 0]
        real = [i for i, l in enumerate(labels) if l == 1]
        per_class = min(len(fake), len(real), n // 2)
        indices = random.sample(fake, per_class) + random.sample(real, per_class)
        random.shuffle(indices)
        print(f"  Sampled: {per_class}x2 = {len(indices)}")
        return data.select(indices)

    print(f"\n[Train] Target: {cfg['max_train_samples']}")
    train_ds = sample(ds['train'], cfg['max_train_samples'], cfg['seed'])

    print(f"\n[Val] Target: {cfg['max_val_samples']}")
    val_ds = sample(ds['validation'], cfg['max_val_samples'], cfg['seed'])

    print(f"\n[Test] Full: {len(ds['test'])}")
    test_ds = ds['test']

    # Transforms
    train_tf = get_train_transforms(cfg['image_size'])
    val_tf = get_val_transforms(cfg['image_size'])

    # DataLoaders
    train_loader = DataLoader(
        DeepfakeDataset(train_ds, train_tf, cfg['image_size']),
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg['num_workers'] > 0,
        prefetch_factor=4 if cfg['num_workers'] > 0 else None
    )

    val_loader = DataLoader(
        DeepfakeDataset(val_ds, val_tf, cfg['image_size']),
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=True,
        persistent_workers=cfg['num_workers'] > 0
    )

    test_loader = DataLoader(
        DeepfakeDataset(test_ds, val_tf, cfg['image_size']),
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=True,
        persistent_workers=cfg['num_workers'] > 0
    )

    print(f"\nLoaders: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)} batches")

    # Time estimate
    est = len(train_loader) * 0.12  # ~0.12s/batch with 48GB
    print(f"Estimated: {est/60:.1f}min/epoch, {est*15/60:.1f}min for 15 epochs")
    print("=" * 70)

    return train_loader, val_loader, test_loader


# ============================================================================
# Training
# ============================================================================

class Meter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(1, self.count)


def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch, cfg):
    model.train()
    loss_m, acc_m = Meter(), Meter()

    pbar = tqdm(loader, desc=f"Epoch {epoch}", ncols=100)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        bs = images.size(0)

        # Mix augmentation
        use_mix = cfg['use_mixup'] and random.random() < 0.5
        if use_mix:
            if random.random() < 0.5:
                images, ya, yb, lam = mixup(images, labels, cfg['mixup_alpha'])
            else:
                images, ya, yb, lam = cutmix(images, labels, cfg['cutmix_alpha'])

        optimizer.zero_grad()

        # Forward
        if AMP_V2:
            with autocast(device_type='cuda', dtype=torch.float16):
                out = model(images)
                loss = mix_loss(criterion, out, ya, yb, lam) if use_mix else criterion(out, labels)
        else:
            with autocast():
                out = model(images)
                loss = mix_loss(criterion, out, ya, yb, lam) if use_mix else criterion(out, labels)

        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        with torch.no_grad():
            pred = out.argmax(1)
            if use_mix:
                acc = (lam * pred.eq(ya).float() + (1-lam) * pred.eq(yb).float()).sum().item()
            else:
                acc = pred.eq(labels).sum().item()

        loss_m.update(loss.item(), bs)
        acc_m.update(acc / bs, bs)

        pbar.set_postfix(loss=f"{loss_m.avg:.4f}", acc=f"{acc_m.avg:.4f}")

    return loss_m.avg, acc_m.avg


def validate(model, loader, criterion, device):
    model.eval()
    loss_m, acc_m = Meter(), Meter()
    preds, labels, probs = [], [], []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Val", ncols=100):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            bs = images.size(0)

            if AMP_V2:
                with autocast(device_type='cuda', dtype=torch.float16):
                    out = model(images)
                    loss = criterion(out, targets)
            else:
                with autocast():
                    out = model(images)
                    loss = criterion(out, targets)

            prob = torch.softmax(out.float(), 1)
            pred = prob.argmax(1)

            loss_m.update(loss.item(), bs)
            acc_m.update(pred.eq(targets).sum().item() / bs, bs)

            preds.extend(pred.cpu().numpy())
            labels.extend(targets.cpu().numpy())
            probs.extend(prob[:, 1].cpu().numpy())

    return loss_m.avg, acc_m.avg, preds, labels, probs


# ============================================================================
# Main
# ============================================================================

def train(cfg=None):
    """48GB + 1시간 최적화 학습"""
    if cfg is None:
        cfg = CONFIG.copy()

    start = time.time()
    set_seed(cfg['seed'])
    setup_for_48gb()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 70)
    print("Deepfake Detection (48GB VRAM, 1-Hour)")
    print("=" * 70)
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    print(f"\nConfig:")
    print(f"  batch_size={cfg['batch_size']}, use_checkpoint={cfg['use_checkpoint']}")
    print(f"  train_samples={cfg['max_train_samples']}, epochs={cfg['num_epochs']}")
    print(f"  augmentation=STRONG, mixup/cutmix=ON")

    # Directory
    os.makedirs(cfg['save_dir'], exist_ok=True)
    exp_dir = os.path.join(cfg['save_dir'], datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(exp_dir)

    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

    # Data
    train_loader, val_loader, test_loader = load_data(cfg)

    # Model
    print("\n[Model] EfficientNet-B4")
    if ResNet50_Weights is not None:
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        model = resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, cfg['num_classes'])
    model = model.to(device)
    info = get_model_info(model)
    print(f"  Params: {info['parameters_m']:.2f}M")
    # Loss, Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
    optimizer = optim.SGD(model.parameters(), lr=cfg['learning_rate'], momentum=0.9, weight_decay=cfg['weight_decay'])
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # Scheduler
    warmup = cfg['warmup_epochs']
    def lr_fn(ep):
        if ep < warmup:
            return (ep + 1) / max(1, warmup)
        progress = (ep - warmup) / max(1, cfg['num_epochs'] - warmup)
        return max(0.01, 0.5 * (1 + np.cos(np.pi * min(1, progress))))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    # History
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'lr': [], 'time': []}

    best_acc, best_f1, best_ep = 0, 0, 0
    patience_cnt = 0

    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)

    for epoch in range(1, cfg['num_epochs'] + 1):
        ep_start = time.time()
        elapsed = time.time() - start

        print(f"\nEpoch {epoch}/{cfg['num_epochs']} | Elapsed: {elapsed/60:.1f}min | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Time check (55분에 중단)
        if elapsed > 55 * 60:
            print("\n[!] 55min reached. Stopping.")
            break

        # Train
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, cfg)

        # Validate
        val_loss, val_acc, val_preds, val_labels, _ = validate(model, val_loader, criterion, device)
        val_f1 = f1_score(val_labels, val_preds, average='macro')

        scheduler.step()

        ep_time = time.time() - ep_start

        # History
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['time'].append(ep_time)

        print(f"  Train: loss={tr_loss:.4f}, acc={tr_acc:.4f}")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.4f}, f1={val_f1:.4f}")
        print(f"  Time:  {ep_time:.1f}s")

        # Best
        if val_acc > best_acc:
            best_acc, best_f1, best_ep = val_acc, val_f1, epoch
            patience_cnt = 0
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1
            }, os.path.join(exp_dir, 'best.pth'))
            print(f"  [*] Best! acc={val_acc:.4f}, f1={val_f1:.4f}")
        else:
            patience_cnt += 1
            print(f"  No improvement ({patience_cnt}/{cfg['patience']})")
            if patience_cnt >= cfg['patience']:
                print(f"\n[!] Early stopping at epoch {epoch}")
                break

    total_time = time.time() - start

    print("\n" + "=" * 70)
    print("Training Done!")
    print("=" * 70)
    print(f"Total: {total_time/60:.1f}min")
    print(f"Best: epoch={best_ep}, acc={best_acc:.4f}, f1={best_f1:.4f}")

    # Test
    print("\n[Testing]")
    ckpt = torch.load(os.path.join(exp_dir, 'best.pth'))
    model.load_state_dict(ckpt['model'])

    test_loss, test_acc, test_preds, test_labels, test_probs = validate(model, test_loader, criterion, device)
    test_f1 = f1_score(test_labels, test_preds, average='macro')

    try:
        test_auc = roc_auc_score(test_labels, test_probs)
    except:
        test_auc = 0

    print(f"\nTest: acc={test_acc:.4f}, f1={test_f1:.4f}, auc={test_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=['Fake', 'Real'], digits=4))

    cm = confusion_matrix(test_labels, test_preds)
    print("Confusion Matrix:")
    print(f"           Pred")
    print(f"         Fake  Real")
    print(f"  Fake  {cm[0][0]:5} {cm[0][1]:5}")
    print(f"  Real  {cm[1][0]:5} {cm[1][1]:5}")

    # Save
    results = {
        'total_time_min': total_time / 60,
        'best_epoch': best_ep,
        'best_val_acc': best_acc,
        'best_val_f1': best_f1,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_auc': test_auc,
        'confusion_matrix': cm.tolist()
    }

    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(exp_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Plot
    try:
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))

        ep = range(1, len(history['train_loss']) + 1)

        ax[0,0].plot(ep, history['train_loss'], 'b-', label='Train')
        ax[0,0].plot(ep, history['val_loss'], 'r-', label='Val')
        ax[0,0].set_title('Loss')
        ax[0,0].legend()
        ax[0,0].grid(alpha=0.3)

        ax[0,1].plot(ep, history['train_acc'], 'b-', label='Train')
        ax[0,1].plot(ep, history['val_acc'], 'r-', label='Val')
        ax[0,1].set_title('Accuracy')
        ax[0,1].legend()
        ax[0,1].grid(alpha=0.3)

        ax[1,0].plot(ep, history['val_f1'], 'g-')
        ax[1,0].set_title('Val F1')
        ax[1,0].grid(alpha=0.3)

        ax[1,1].bar(ep, history['time'], color='purple', alpha=0.7)
        ax[1,1].set_title('Time/Epoch (s)')
        ax[1,1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'curves.png'), dpi=200)
        plt.close()
    except Exception as e:
        print(f"Plot error: {e}")

    print(f"\nSaved to: {exp_dir}")
    print("=" * 70)

    return model, history, results


# ============================================================================
# Entry
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("Deepfake Detection - 48GB VRAM, 1-Hour Limit")
    print("=" * 70)
    print("\n[Settings]")
    print(f"  VRAM: 48GB → batch=32, checkpoint=OFF")
    print(f"  Time: 1hr → 40K samples, 15-20 epochs")
    print(f"  Augmentation: STRONG (Deepfake + Mix)")

    train()

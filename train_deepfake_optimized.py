"""
Deepfake Detection using EfficientNet-B4 (Optimized Version)

Dataset: JamieWithofs/Deepfake-and-real-images from HuggingFace (140K images)
Model: EfficientNet-B4 (optimized from-scratch implementation)
Task: Binary classification (Fake vs Real)
  - Label 0: Fake (딥페이크 이미지)
  - Label 1: Real (진짜 이미지)

Optimizations for RunPod (36GB RAM):
1. Mixed Precision Training (AMP) - 50% memory reduction, 2x speed
2. Gradient Accumulation - effective larger batch sizes
3. Gradient Checkpointing - memory efficient backprop
4. Early Stopping - prevent overfitting
5. Proper seed setting - full reproducibility
6. Deepfake-specific augmentations (JPEG, Blur, etc.)
7. Original test set distribution - realistic evaluation
8. Memory-optimized DataLoader settings
9. Full PyTorch version compatibility (1.9+, 2.0+)
10. CPU/GPU compatible AMP handling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageFilter
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import io
import random
from datetime import datetime
import json
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings

# Version-compatible AMP imports
try:
    from torch.amp import autocast as torch_autocast
    from torch.amp import GradScaler
    AMP_VERSION = 2  # PyTorch 2.0+
except ImportError:
    try:
        from torch.cuda.amp import autocast as torch_autocast
        from torch.cuda.amp import GradScaler
        AMP_VERSION = 1  # PyTorch 1.6+
    except ImportError:
        AMP_VERSION = 0
        warnings.warn("AMP not available. Mixed precision training disabled.")

# Pillow version compatibility
try:
    RESAMPLING_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLING_BILINEAR = Image.BILINEAR

# Import optimized EfficientNet
from efficientnet_optimized import efficientnet_b4, count_parameters, get_model_info

# PyTorch version info
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])


# ============================================================================
# 0. Reproducibility
# ============================================================================

def set_seed(seed: int = 42):
    """Set all random seeds for full reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ============================================================================
# 1. AMP Compatibility Wrapper
# ============================================================================

class AMPContext:
    """
    Version-compatible AMP context manager
    Handles PyTorch 1.x and 2.x differences, and CPU fallback
    """
    def __init__(self, device: torch.device, enabled: bool = True):
        self.device = device
        self.enabled = enabled and AMP_VERSION > 0 and device.type == 'cuda'
        self.device_type = device.type

    def autocast(self):
        """Return appropriate autocast context"""
        if not self.enabled:
            return NullContext()

        if AMP_VERSION == 2:
            return torch_autocast(device_type=self.device_type, dtype=torch.float16)
        else:
            return torch_autocast()

    def get_scaler(self):
        """Return GradScaler if AMP is enabled"""
        if self.enabled:
            return GradScaler()
        return None


class NullContext:
    """Null context manager for when AMP is disabled"""
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ============================================================================
# 2. Deepfake-Specific Data Augmentations
# ============================================================================

class JPEGCompression:
    """
    Simulate JPEG compression artifacts
    Deepfakes often have different compression signatures
    """
    def __init__(self, quality_range=(30, 95), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            try:
                quality = random.randint(*self.quality_range)
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=quality)
                buffer.seek(0)
                img = Image.open(buffer)
                img.load()  # Force load to release buffer
                img = img.convert('RGB')
                buffer.close()
            except Exception:
                pass  # Return original image on error
        return img


class GaussianBlur:
    """
    Apply Gaussian blur to simulate image quality variations
    """
    def __init__(self, radius_range=(0.5, 2.0), p=0.3):
        self.radius_range = radius_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            try:
                radius = random.uniform(*self.radius_range)
                img = img.filter(ImageFilter.GaussianBlur(radius=radius))
            except Exception:
                pass
        return img


class RandomDownscale:
    """
    Random downscale and upscale to simulate resolution variations
    """
    def __init__(self, scale_range=(0.5, 0.9), p=0.3):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            try:
                w, h = img.size
                if w > 0 and h > 0:
                    scale = random.uniform(*self.scale_range)
                    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                    img = img.resize((new_w, new_h), RESAMPLING_BILINEAR)
                    img = img.resize((w, h), RESAMPLING_BILINEAR)
            except Exception:
                pass
        return img


def get_transforms(image_size: int = 380, is_training: bool = True):
    """
    Get data augmentation transforms with deepfake-specific augmentations

    Args:
        image_size: target image size (EfficientNet-B4 uses 380x380)
        is_training: whether for training or validation

    Returns:
        torchvision transforms
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            # Deepfake-specific augmentations
            JPEGCompression(quality_range=(30, 95), p=0.5),
            GaussianBlur(radius_range=(0.5, 2.0), p=0.3),
            RandomDownscale(scale_range=(0.5, 0.9), p=0.3),
            # Standard augmentations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05)
            ),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


# ============================================================================
# 3. Dataset Preparation
# ============================================================================

class DeepfakeDataset(Dataset):
    """
    Custom Dataset for Deepfake Detection
    Memory-efficient implementation with lazy loading and error handling
    """
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]

            # Get image and label
            image = item['image']
            label = item['label']  # 0: Fake, 1: Real

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Apply transforms
            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            # Return a placeholder on error (black image with random label)
            warnings.warn(f"Error loading image at index {idx}: {e}")
            placeholder = torch.zeros(3, 380, 380)
            return placeholder, 0


def balance_dataset(dataset, label_column='label', seed=42):
    """
    Balance dataset to have equal number of samples per class
    """
    random.seed(seed)

    labels = dataset[label_column]
    fake_indices = [i for i, label in enumerate(labels) if label == 0]
    real_indices = [i for i, label in enumerate(labels) if label == 1]

    num_fake = len(fake_indices)
    num_real = len(real_indices)

    # Avoid division by zero
    total = num_fake + num_real
    if total == 0:
        raise ValueError("Dataset is empty!")

    print(f"\n  Original distribution:")
    print(f"    Fake (label=0): {num_fake} ({num_fake/total*100:.1f}%)")
    print(f"    Real (label=1): {num_real} ({num_real/total*100:.1f}%)")

    min_samples = min(num_fake, num_real)
    if min_samples == 0:
        raise ValueError("One class has zero samples!")

    balanced_fake_indices = random.sample(fake_indices, min_samples)
    balanced_real_indices = random.sample(real_indices, min_samples)

    balanced_indices = balanced_fake_indices + balanced_real_indices
    random.shuffle(balanced_indices)

    balanced_dataset = dataset.select(balanced_indices)

    print(f"  Balanced distribution:")
    print(f"    Fake (label=0): {min_samples} (50.0%)")
    print(f"    Real (label=1): {min_samples} (50.0%)")
    print(f"    Total: {len(balanced_dataset)}")

    return balanced_dataset


def load_deepfake_dataset(
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 380,
    pin_memory: bool = True
):
    """
    Load Deepfake dataset with optimized settings for RunPod

    Key optimizations:
    - Training/Validation balanced, Test set keeps original distribution
    - Efficient DataLoader settings
    - drop_last=True for training
    - Windows compatibility for num_workers
    """
    # Import here to avoid import error if datasets not installed
    from datasets import load_dataset

    print("=" * 80)
    print("Loading Deepfake Dataset from HuggingFace")
    print("=" * 80)

    print("\nDownloading dataset (140K images)...")
    dataset = load_dataset("JamieWithofs/Deepfake-and-real-images")

    print(f"\nDataset loaded:")
    print(f"  Train samples: {len(dataset['train'])}")
    print(f"  Validation samples: {len(dataset['validation'])}")
    print(f"  Test samples: {len(dataset['test'])}")

    # Balance training set
    print("\n" + "-" * 40)
    print("Balancing Training Dataset")
    train_dataset = balance_dataset(dataset['train'])

    # Balance validation set
    print("\n" + "-" * 40)
    print("Balancing Validation Dataset")
    val_dataset = balance_dataset(dataset['validation'])

    # Keep test set with ORIGINAL distribution (FIX)
    print("\n" + "-" * 40)
    print("Test Dataset (Original Distribution - for realistic evaluation)")
    test_dataset = dataset['test']
    labels = test_dataset['label']
    num_fake = sum(1 for l in labels if l == 0)
    num_real = sum(1 for l in labels if l == 1)
    total = len(labels)
    if total > 0:
        print(f"  Fake (label=0): {num_fake} ({num_fake/total*100:.1f}%)")
        print(f"  Real (label=1): {num_real} ({num_real/total*100:.1f}%)")
    print(f"  Total: {total}")

    # Create PyTorch datasets
    print("\n" + "-" * 40)
    print("Creating DataLoaders")

    train_transforms = get_transforms(image_size, is_training=True)
    val_transforms = get_transforms(image_size, is_training=False)

    train_pytorch_dataset = DeepfakeDataset(train_dataset, train_transforms)
    val_pytorch_dataset = DeepfakeDataset(val_dataset, val_transforms)
    test_pytorch_dataset = DeepfakeDataset(test_dataset, val_transforms)

    # Windows compatibility: num_workers=0 if on Windows and not in __main__
    effective_num_workers = num_workers
    if os.name == 'nt' and num_workers > 0:
        # Check if multiprocessing is safe
        import sys
        if not hasattr(sys.modules['__main__'], '__file__'):
            effective_num_workers = 0
            print("  Note: Setting num_workers=0 for Windows compatibility")

    # DataLoader settings
    persistent = effective_num_workers > 0 and TORCH_VERSION >= (1, 8)

    train_loader = DataLoader(
        train_pytorch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=effective_num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=persistent
    )

    val_loader = DataLoader(
        val_pytorch_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=persistent
    )

    test_loader = DataLoader(
        test_pytorch_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=persistent
    )

    print(f"\nDataLoaders created:")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Num workers: {effective_num_workers}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print("=" * 80)

    return train_loader, val_loader, test_loader


# ============================================================================
# 4. Training Functions with Mixed Precision
# ============================================================================

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    amp_context: AMPContext,
    scaler,
    accumulation_steps: int = 1
):
    """
    Train for one epoch with mixed precision and gradient accumulation

    Args:
        model: EfficientNet model
        train_loader: training data loader
        criterion: loss function
        optimizer: optimizer
        device: cuda or cpu
        epoch: current epoch number
        amp_context: AMP context for device-compatible autocast
        scaler: GradScaler for mixed precision (None if disabled)
        accumulation_steps: gradient accumulation steps

    Returns:
        avg_loss, avg_acc
    """
    model.train()

    losses = AverageMeter()
    accuracies = AverageMeter()

    optimizer.zero_grad()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

    num_batches = len(train_loader)

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.size(0)

        # Mixed precision forward pass
        with amp_context.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps  # Scale for accumulation

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation step
        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        # Handle remaining gradients at the end of epoch (FIX)
        elif batch_idx == num_batches - 1:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        # Calculate accuracy
        with torch.no_grad():
            _, predicted = outputs.max(1)
            correct = predicted.eq(labels).sum().item()
            accuracy = correct / batch_size

        # Update metrics (scale loss back for logging)
        losses.update(loss.item() * accumulation_steps, batch_size)
        accuracies.update(accuracy, batch_size)

        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{accuracies.avg:.4f}'
        })

    return losses.avg, accuracies.avg


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_context: AMPContext,
    epoch: int = None
):
    """
    Validate the model with mixed precision

    Returns:
        avg_loss, avg_acc, all_preds, all_labels, all_probs
    """
    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()

    all_preds = []
    all_labels = []
    all_probs = []

    desc = f"Epoch {epoch} [Val]" if epoch is not None else "Validation"
    pbar = tqdm(val_loader, desc=desc)

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = images.size(0)

            # Mixed precision forward
            with amp_context.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Calculate accuracy (convert to float32 for softmax stability)
            probs = torch.softmax(outputs.float(), dim=1)
            _, predicted = probs.max(1)
            correct = predicted.eq(labels).sum().item()
            accuracy = correct / batch_size

            # Update metrics
            losses.update(loss.item(), batch_size)
            accuracies.update(accuracy, batch_size)

            # Store predictions
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of Real

            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{accuracies.avg:.4f}'
            })

    return losses.avg, accuracies.avg, all_preds, all_labels, all_probs


# ============================================================================
# 5. Training Pipeline with All Optimizations
# ============================================================================

def train_deepfake_detector(
    num_epochs: int = 30,
    batch_size: int = 16,
    accumulation_steps: int = 2,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    image_size: int = 380,
    save_dir: str = 'checkpoints',
    patience: int = 7,
    use_checkpoint: bool = True,
    device: torch.device = None,
    seed: int = 42
):
    """
    Complete training pipeline with all optimizations for RunPod (36GB RAM)

    Optimizations:
    1. Mixed Precision (AMP) - 50% VRAM reduction
    2. Gradient Accumulation - effective batch_size = batch_size * accumulation_steps
    3. Gradient Checkpointing - trade compute for memory
    4. Early Stopping - prevent overfitting
    5. Cosine Annealing with Warmup - better convergence
    6. Gradient Clipping - training stability
    7. Full PyTorch version compatibility
    8. CPU fallback support

    Args:
        num_epochs: max training epochs
        batch_size: batch size per step
        accumulation_steps: gradient accumulation (effective batch = batch_size * steps)
        learning_rate: initial learning rate
        weight_decay: L2 regularization
        image_size: input image size (380 for B4)
        save_dir: checkpoint directory
        patience: early stopping patience
        use_checkpoint: enable gradient checkpointing
        device: cuda or cpu
        seed: random seed
    """
    # Set seed for reproducibility
    set_seed(seed)

    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup AMP context
    amp_context = AMPContext(device, enabled=True)
    scaler = amp_context.get_scaler()

    print("\n" + "=" * 80)
    print("Deepfake Detection Training (Optimized)")
    print("=" * 80)
    print(f"\nSystem Info:")
    print(f"  PyTorch Version: {torch.__version__}")
    print(f"  AMP Version: {AMP_VERSION} ({'2.0+' if AMP_VERSION == 2 else '1.x' if AMP_VERSION == 1 else 'disabled'})")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nConfiguration:")
    print(f"  Model: EfficientNet-B4 (gradient checkpointing: {use_checkpoint})")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size} x {accumulation_steps} = {batch_size * accumulation_steps} effective")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Mixed Precision: {'Enabled' if amp_context.enabled else 'Disabled'}")
    print(f"  Seed: {seed}")

    # Create experiment directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(save_dir, f"efficientnet_b4_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Save config
    config = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'accumulation_steps': accumulation_steps,
        'effective_batch_size': batch_size * accumulation_steps,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'image_size': image_size,
        'patience': patience,
        'use_checkpoint': use_checkpoint,
        'seed': seed,
        'device': str(device),
        'torch_version': torch.__version__,
        'amp_version': AMP_VERSION
    }
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Load dataset
    train_loader, val_loader, test_loader = load_deepfake_dataset(
        batch_size=batch_size,
        num_workers=4,
        image_size=image_size
    )

    # Create model with gradient checkpointing
    print("\nBuilding Model...")
    model = efficientnet_b4(num_classes=2, use_checkpoint=use_checkpoint)
    model = model.to(device)

    model_info = get_model_info(model)
    print(f"  Model: EfficientNet-B4")
    print(f"  Parameters: {model_info['parameters_m']:.2f}M")
    print(f"  Recommended resolution: {model_info['resolution']}")

    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler with warmup
    warmup_epochs = min(3, num_epochs // 2)  # At most half of total epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        else:
            # Cosine annealing after warmup
            remaining = max(1, num_epochs - warmup_epochs)
            progress = (epoch - warmup_epochs) / remaining
            return max(0.0, 0.5 * (1 + np.cos(np.pi * min(1.0, progress))))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'lr': []
    }

    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    # Training loop
    print("\n" + "=" * 80)
    print("Training Started")
    print("=" * 80)

    for epoch in range(1, num_epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}/{num_epochs} | LR: {current_lr:.6f}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            amp_context, scaler, accumulation_steps
        )

        # Validate
        val_loss, val_acc, val_preds, val_labels, _ = validate(
            model, val_loader, criterion, device, amp_context, epoch
        )

        # Calculate F1 score
        val_f1 = f1_score(val_labels, val_preds, average='macro')

        # Update scheduler
        scheduler.step()

        # Save history
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        history['val_f1'].append(float(val_f1))
        history['lr'].append(float(current_lr))

        # Print summary
        print(f"\n  Summary:")
        print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"    Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1: {val_f1:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0

            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_loss': val_loss,
            }
            if scaler is not None:
                save_dict['scaler_state_dict'] = scaler.state_dict()

            torch.save(save_dict, os.path.join(experiment_dir, 'best_model.pth'))
            print(f"    [*] New best model saved! (Val Acc: {val_acc:.4f}, F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"    No improvement for {patience_counter}/{patience} epochs")

        # Early stopping
        if patience_counter >= patience:
            print(f"\n[!] Early stopping triggered at epoch {epoch}")
            break

        # Checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(experiment_dir, f'checkpoint_epoch_{epoch}.pth'))

    # Training completed
    print("\n" + "=" * 80)
    print("Training Completed!")
    print("=" * 80)
    print(f"\nBest Results:")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Best Val Acc: {best_val_acc:.4f}")
    print(f"  Best Val F1: {best_val_f1:.4f}")

    # Load best model for testing
    print("\nEvaluating on Test Set...")
    best_model_path = os.path.join(experiment_dir, 'best_model.pth')

    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_loss, test_acc, test_preds, test_labels, test_probs = validate(
            model, test_loader, criterion, device, amp_context
        )

        print(f"\nTest Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Acc:  {test_acc:.4f}")

        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            test_labels,
            test_preds,
            target_names=['Fake (0)', 'Real (1)'],
            digits=4
        ))

        # Confusion matrix
        cm = confusion_matrix(test_labels, test_preds)
        print("\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              Fake    Real")
        print(f"  Actual Fake  {cm[0][0]:5d}   {cm[0][1]:5d}")
        print(f"  Actual Real  {cm[1][0]:5d}   {cm[1][1]:5d}")

        # Save final results
        results = {
            'best_epoch': best_epoch,
            'best_val_acc': float(best_val_acc),
            'best_val_f1': float(best_val_f1),
            'test_acc': float(test_acc),
            'test_loss': float(test_loss),
            'confusion_matrix': cm.tolist()
        }
    else:
        results = {
            'best_epoch': best_epoch,
            'best_val_acc': float(best_val_acc),
            'best_val_f1': float(best_val_f1),
            'test_acc': None,
            'test_loss': None,
            'error': 'Best model not found'
        }

    with open(os.path.join(experiment_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Save history
    with open(os.path.join(experiment_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    # Plot training curves
    try:
        plot_training_curves(history, experiment_dir)
    except Exception as e:
        warnings.warn(f"Failed to plot training curves: {e}")

    print(f"\nAll results saved to: {experiment_dir}")
    print("=" * 80)

    return model, history


# ============================================================================
# 6. Visualization
# ============================================================================

def plot_training_curves(history: dict, save_dir: str):
    """Plot and save training curves"""
    if not history.get('train_loss'):
        warnings.warn("No training history to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # F1 Score
    axes[1, 0].plot(epochs, history['val_f1'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1 Score', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Learning Rate
    axes[1, 1].plot(epochs, history['lr'], 'purple', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining curves saved to: {save_path}")
    plt.close()


# ============================================================================
# 7. Inference
# ============================================================================

def predict_image(
    model: nn.Module,
    image_path: str,
    device: torch.device,
    image_size: int = 380
) -> tuple:
    """
    Predict if an image is real or fake

    Args:
        model: trained EfficientNet model
        image_path: path to image file
        device: cuda or cpu
        image_size: input image size

    Returns:
        prediction: 0 for Fake, 1 for Real
        confidence: prediction confidence score
    """
    model.eval()

    amp_context = AMPContext(device, enabled=True)
    transform = get_transforms(image_size, is_training=False)

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        with amp_context.autocast():
            output = model(image_tensor)
        probabilities = torch.softmax(output.float(), dim=1)
        confidence, predicted = probabilities.max(1)

    return predicted.item(), confidence.item()


def predict_batch(
    model: nn.Module,
    image_paths: list,
    device: torch.device,
    image_size: int = 380,
    batch_size: int = 16
) -> list:
    """
    Batch prediction for multiple images

    Returns:
        List of (image_path, prediction, confidence) tuples
        Failed images will have prediction=-1, confidence=0.0
    """
    model.eval()
    amp_context = AMPContext(device, enabled=True)
    transform = get_transforms(image_size, is_training=False)

    results = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_tensors = []
        valid_paths = []

        for path in batch_paths:
            try:
                image = Image.open(path).convert('RGB')
                tensor = transform(image)
                batch_tensors.append(tensor)
                valid_paths.append(path)
            except Exception as e:
                warnings.warn(f"Failed to load {path}: {e}")
                results.append((path, -1, 0.0))  # Mark as failed

        if not batch_tensors:
            continue

        batch = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            with amp_context.autocast():
                outputs = model(batch)
            probabilities = torch.softmax(outputs.float(), dim=1)
            confidences, predictions = probabilities.max(1)

        for path, pred, conf in zip(valid_paths, predictions.cpu().numpy(), confidences.cpu().numpy()):
            results.append((path, int(pred), float(conf)))

    return results


# ============================================================================
# 8. Main
# ============================================================================

if __name__ == '__main__':
    # Optimal configuration for RunPod (36GB RAM, single GPU)
    # Effective batch size = 16 * 2 = 32
    # With gradient checkpointing, can fit B4 at 380x380

    config = {
        'num_epochs': 30,
        'batch_size': 16,
        'accumulation_steps': 2,  # Effective batch = 32
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'image_size': 380,  # EfficientNet-B4 default
        'save_dir': 'checkpoints_deepfake',
        'patience': 7,
        'use_checkpoint': True,  # Gradient checkpointing
        'seed': 42
    }

    print("\n" + "=" * 80)
    print("Deepfake Detection with EfficientNet-B4 (Optimized)")
    print("=" * 80)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Train
    model, history = train_deepfake_detector(**config)

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print("\nUsage:")
    print("  from train_deepfake_optimized import predict_image, efficientnet_b4")
    print("  ")
    print("  model = efficientnet_b4(num_classes=2)")
    print("  model.load_state_dict(torch.load('best_model.pth')['model_state_dict'])")
    print("  ")
    print("  prediction, confidence = predict_image(model, 'image.jpg', device)")
    print("  label = 'Fake' if prediction == 0 else 'Real'")
    print("  print(f'{label}: {confidence:.2%}')")
    print("=" * 80)

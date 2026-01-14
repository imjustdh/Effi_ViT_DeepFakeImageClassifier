"""
EfficientNet Optimized Implementation from Scratch in PyTorch

Reference:
- Paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
- https://arxiv.org/abs/1905.11946

Optimizations Applied:
1. Fixed total_blocks calculation with depth_mult
2. Dynamic stage indices for feature extraction
3. Correct SE block reduction based on input channels
4. Resolution config integrated with model
5. Memory-efficient implementation for 36GB RAM
6. Gradient checkpointing support for large models
7. Full PyTorch version compatibility (1.9+, 2.0+)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math
import warnings

# Version-compatible checkpoint import
try:
    from torch.utils.checkpoint import checkpoint as torch_checkpoint
    CHECKPOINT_AVAILABLE = True
except ImportError:
    CHECKPOINT_AVAILABLE = False
    warnings.warn("torch.utils.checkpoint not available. Gradient checkpointing disabled.")


# ============================================================================
# Configuration
# ============================================================================

EFFICIENTNET_CONFIGS: Dict[str, Dict] = {
    'b0': {'width': 1.0, 'depth': 1.0, 'resolution': 224, 'dropout': 0.2},
    'b1': {'width': 1.0, 'depth': 1.1, 'resolution': 240, 'dropout': 0.2},
    'b2': {'width': 1.1, 'depth': 1.2, 'resolution': 260, 'dropout': 0.3},
    'b3': {'width': 1.2, 'depth': 1.4, 'resolution': 300, 'dropout': 0.3},
    'b4': {'width': 1.4, 'depth': 1.8, 'resolution': 380, 'dropout': 0.4},
    'b5': {'width': 1.6, 'depth': 2.2, 'resolution': 456, 'dropout': 0.4},
    'b6': {'width': 1.8, 'depth': 2.6, 'resolution': 528, 'dropout': 0.5},
    'b7': {'width': 2.0, 'depth': 3.1, 'resolution': 600, 'dropout': 0.5},
}

# PyTorch version check for compatibility
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])
SUPPORTS_USE_REENTRANT = TORCH_VERSION >= (1, 11)


# ============================================================================
# Helper Functions
# ============================================================================

def _checkpoint(function, *args, use_reentrant: bool = False):
    """
    Version-compatible gradient checkpointing wrapper
    """
    if not CHECKPOINT_AVAILABLE:
        return function(*args)

    if SUPPORTS_USE_REENTRANT:
        return torch_checkpoint(function, *args, use_reentrant=use_reentrant)
    else:
        return torch_checkpoint(function, *args)


# ============================================================================
# 1. Activation Functions
# ============================================================================

class Swish(nn.Module):
    """
    Swish activation function: x * sigmoid(x)
    Using nn.SiLU for better performance when available
    """
    def __init__(self, inplace: bool = False):
        super().__init__()
        # Note: inplace=False is safer with gradient checkpointing
        self.silu = nn.SiLU(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.silu(x)


# ============================================================================
# 2. Squeeze-and-Excitation Block
# ============================================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block with correct reduction base

    Architecture:
    Input -> Global Average Pool -> FC -> ReLU -> FC -> Sigmoid -> Scale Input

    Fix: SE reduction is based on input channels, not expanded channels
    """
    def __init__(
        self,
        channels: int,
        se_ratio: float = 0.25,
        base_channels: Optional[int] = None
    ):
        super().__init__()
        # Use base_channels for reduction calculation (paper specification)
        reduction_channels = base_channels if base_channels is not None else channels
        squeezed_channels = max(1, int(reduction_channels * se_ratio))

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, squeezed_channels, 1, bias=True),
            nn.SiLU(inplace=False),  # inplace=False for gradient safety
            nn.Conv2d(squeezed_channels, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


# ============================================================================
# 3. MBConv Block (Mobile Inverted Bottleneck Convolution)
# ============================================================================

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution Block

    Optimizations:
    - Fused operations where possible
    - Safe activations for gradient checkpointing (no inplace)
    - Correct stochastic depth implementation
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 6,
        se_ratio: float = 0.25,
        drop_connect_rate: float = 0.2
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate
        self.expand_ratio = expand_ratio

        # Skip connection only when stride=1 and same dimensions
        self.use_residual = (stride == 1 and in_channels == out_channels)

        # Hidden dimension (expansion)
        hidden_dim = in_channels * expand_ratio

        # Build layers
        layers = []

        # 1. Expansion phase (only if expand_ratio != 1)
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim, momentum=0.01, eps=1e-3))
            layers.append(nn.SiLU(inplace=False))

        # 2. Depthwise convolution phase
        layers.append(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=hidden_dim,
                bias=False
            )
        )
        layers.append(nn.BatchNorm2d(hidden_dim, momentum=0.01, eps=1e-3))
        layers.append(nn.SiLU(inplace=False))

        self.conv = nn.Sequential(*layers)

        # 3. Squeeze-and-Excitation (using input channels as base)
        self.se = SEBlock(hidden_dim, se_ratio, base_channels=in_channels)

        # 4. Projection phase (pointwise convolution)
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # Main path
        x = self.conv(x)
        x = self.se(x)
        x = self.project(x)

        # Skip connection with stochastic depth
        if self.use_residual:
            if self.training and self.drop_connect_rate > 0:
                x = self._drop_connect(x)
            x = x + identity

        return x

    def _drop_connect(self, x: torch.Tensor) -> torch.Tensor:
        """
        Stochastic Depth (Drop Connect)
        Randomly drop the entire residual branch during training
        """
        if self.drop_connect_rate <= 0 or self.drop_connect_rate >= 1:
            return x

        keep_prob = 1.0 - self.drop_connect_rate

        # Generate random mask for each sample in batch
        batch_size = x.shape[0]
        random_tensor = torch.empty(
            (batch_size, 1, 1, 1),
            dtype=x.dtype,
            device=x.device
        ).uniform_(0, 1)

        binary_mask = (random_tensor < keep_prob).float()

        # Scale to maintain expected value
        if keep_prob > 0:
            x = x / keep_prob * binary_mask

        return x


# ============================================================================
# 4. EfficientNet Architecture
# ============================================================================

class EfficientNet(nn.Module):
    """
    EfficientNet: Scalable Convolutional Neural Network

    Optimizations:
    - Correct total_blocks calculation with depth_mult
    - Dynamic stage indices for feature extraction
    - Gradient checkpointing support for memory efficiency
    - Configurable resolution tracking
    - Full PyTorch version compatibility
    """

    # Base block configuration: [expand_ratio, channels, repeats, stride, kernel_size]
    BASE_BLOCK_CONFIGS = [
        [1, 16, 1, 1, 3],   # Stage 1
        [6, 24, 2, 2, 3],   # Stage 2
        [6, 40, 2, 2, 5],   # Stage 3
        [6, 80, 3, 2, 3],   # Stage 4
        [6, 112, 3, 1, 5],  # Stage 5
        [6, 192, 4, 2, 5],  # Stage 6
        [6, 320, 1, 1, 3],  # Stage 7
    ]

    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        dropout_rate: float = 0.2,
        drop_connect_rate: float = 0.2,
        use_checkpoint: bool = False
    ):
        """
        Args:
            num_classes: Number of output classes
            width_mult: Width multiplier (beta)
            depth_mult: Depth multiplier (alpha)
            dropout_rate: Dropout rate before final FC
            drop_connect_rate: Drop connect rate for MBConv blocks
            use_checkpoint: Enable gradient checkpointing for memory efficiency
        """
        super().__init__()

        self.use_checkpoint = use_checkpoint and CHECKPOINT_AVAILABLE
        self.width_mult = width_mult
        self.depth_mult = depth_mult

        # Calculate total blocks AFTER applying depth_mult (FIX)
        total_blocks = sum([
            self._round_repeats(config[2], depth_mult)
            for config in self.BASE_BLOCK_CONFIGS
        ])

        # Avoid division by zero
        if total_blocks == 0:
            total_blocks = 1

        # Stem
        out_channels = self._round_channels(32, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
            nn.SiLU(inplace=False)
        )

        # Build MBConv blocks with dynamic stage tracking
        self.blocks = nn.ModuleList([])
        self.stage_indices = []  # Track stage boundaries (FIX)
        in_channels = out_channels
        block_idx = 0

        for stage_idx, (expand_ratio, channels, repeats, stride, kernel_size) in enumerate(self.BASE_BLOCK_CONFIGS):
            out_channels = self._round_channels(channels, width_mult)
            num_repeats = self._round_repeats(repeats, depth_mult)

            for i in range(num_repeats):
                # Calculate drop connect rate (linearly increase)
                drop_rate = drop_connect_rate * block_idx / total_blocks

                # First block of each stage may have stride > 1
                block_stride = stride if i == 0 else 1

                self.blocks.append(
                    MBConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=block_stride,
                        expand_ratio=expand_ratio,
                        se_ratio=0.25,
                        drop_connect_rate=drop_rate
                    )
                )

                in_channels = out_channels
                block_idx += 1

            # Store stage boundary index (FIX: dynamic calculation)
            if block_idx > 0:
                self.stage_indices.append(block_idx - 1)

        # Head
        final_channels = self._round_channels(1280, width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, final_channels, 1, bias=False),
            nn.BatchNorm2d(final_channels, momentum=0.01, eps=1e-3),
            nn.SiLU(inplace=False)
        )

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # inplace=False for gradient checkpointing compatibility
        self.dropout = nn.Dropout(dropout_rate, inplace=False)
        self.fc = nn.Linear(final_channels, num_classes)

        # Store feature dimension for downstream tasks
        self.feature_dim = final_channels

        # Initialize weights
        self._initialize_weights()

    def _round_channels(self, channels: int, width_mult: float, divisor: int = 8) -> int:
        """Round number of channels to nearest divisor (for hardware efficiency)"""
        if width_mult <= 0:
            return divisor

        channels = channels * width_mult
        new_channels = max(divisor, int(channels + divisor / 2) // divisor * divisor)

        # Make sure rounding doesn't decrease by more than 10%
        if new_channels < 0.9 * channels:
            new_channels += divisor

        return int(new_channels)

    def _round_repeats(self, repeats: int, depth_mult: float) -> int:
        """Round number of block repeats based on depth multiplier"""
        if depth_mult <= 0:
            return 1
        return max(1, int(math.ceil(depth_mult * repeats)))

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_blocks(self, x: torch.Tensor, return_features: bool = False):
        """Forward through MBConv blocks with optional checkpointing"""
        features = []

        for idx, block in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                x = _checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

            # Use dynamic stage indices (FIX)
            if return_features and idx in self.stage_indices:
                features.append(x)

        return x, features

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Args:
            x: (B, 3, H, W) - input images
            return_features: if True, return intermediate features

        Returns:
            out: (B, num_classes) - class logits
            features: (optional) list of intermediate features at stage boundaries
        """
        features = []

        # Stem
        x = self.stem(x)
        if return_features:
            features.append(x)

        # MBConv blocks
        x, block_features = self._forward_blocks(x, return_features)
        if return_features:
            features.extend(block_features)

        # Head
        x = self.head(x)
        if return_features:
            features.append(x)

        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if return_features:
            return x, features
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification head (for transfer learning)"""
        x = self.stem(x)
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = _checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def get_classifier(self) -> nn.Linear:
        """Get the classifier layer"""
        return self.fc

    def reset_classifier(self, num_classes: int):
        """Reset the classifier for a new number of classes"""
        self.fc = nn.Linear(self.feature_dim, num_classes)
        # Initialize the new classifier properly
        nn.init.kaiming_uniform_(self.fc.weight, a=math.sqrt(5))
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)


# ============================================================================
# 5. EfficientNet Variants (B0 ~ B7)
# ============================================================================

def _create_efficientnet(
    variant: str,
    num_classes: int = 1000,
    pretrained: bool = False,
    use_checkpoint: bool = False,
    drop_connect_rate: float = 0.2
) -> EfficientNet:
    """
    Factory function to create EfficientNet variants

    Args:
        variant: Model variant ('b0' to 'b7')
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights (warning if True)
        use_checkpoint: Enable gradient checkpointing
        drop_connect_rate: Drop connect rate

    Returns:
        EfficientNet model with recommended_resolution attribute
    """
    if variant not in EFFICIENTNET_CONFIGS:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(EFFICIENTNET_CONFIGS.keys())}")

    config = EFFICIENTNET_CONFIGS[variant]

    model = EfficientNet(
        num_classes=num_classes,
        width_mult=config['width'],
        depth_mult=config['depth'],
        dropout_rate=config['dropout'],
        drop_connect_rate=drop_connect_rate,
        use_checkpoint=use_checkpoint
    )

    # Attach recommended resolution to model
    model.recommended_resolution = config['resolution']
    model.variant = variant

    if pretrained:
        warnings.warn(
            f"Pretrained weights not available for from-scratch implementation. "
            f"Use torchvision.models.efficientnet_{variant}(weights='IMAGENET1K_V1') instead.",
            UserWarning
        )

    return model


def efficientnet_b0(num_classes: int = 1000, pretrained: bool = False, **kwargs) -> EfficientNet:
    """EfficientNet-B0: Resolution 224x224, ~5.3M params"""
    return _create_efficientnet('b0', num_classes, pretrained, **kwargs)


def efficientnet_b1(num_classes: int = 1000, pretrained: bool = False, **kwargs) -> EfficientNet:
    """EfficientNet-B1: Resolution 240x240, ~7.8M params"""
    return _create_efficientnet('b1', num_classes, pretrained, **kwargs)


def efficientnet_b2(num_classes: int = 1000, pretrained: bool = False, **kwargs) -> EfficientNet:
    """EfficientNet-B2: Resolution 260x260, ~9.2M params"""
    return _create_efficientnet('b2', num_classes, pretrained, **kwargs)


def efficientnet_b3(num_classes: int = 1000, pretrained: bool = False, **kwargs) -> EfficientNet:
    """EfficientNet-B3: Resolution 300x300, ~12M params"""
    return _create_efficientnet('b3', num_classes, pretrained, **kwargs)


def efficientnet_b4(num_classes: int = 1000, pretrained: bool = False, **kwargs) -> EfficientNet:
    """EfficientNet-B4: Resolution 380x380, ~19M params"""
    return _create_efficientnet('b4', num_classes, pretrained, **kwargs)


def efficientnet_b5(num_classes: int = 1000, pretrained: bool = False, **kwargs) -> EfficientNet:
    """EfficientNet-B5: Resolution 456x456, ~30M params"""
    return _create_efficientnet('b5', num_classes, pretrained, **kwargs)


def efficientnet_b6(num_classes: int = 1000, pretrained: bool = False, **kwargs) -> EfficientNet:
    """EfficientNet-B6: Resolution 528x528, ~43M params"""
    return _create_efficientnet('b6', num_classes, pretrained, **kwargs)


def efficientnet_b7(num_classes: int = 1000, pretrained: bool = False, **kwargs) -> EfficientNet:
    """EfficientNet-B7: Resolution 600x600, ~66M params"""
    return _create_efficientnet('b7', num_classes, pretrained, **kwargs)


# ============================================================================
# 6. Utility Functions
# ============================================================================

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_info(model: EfficientNet) -> Dict:
    """Get model information"""
    return {
        'variant': getattr(model, 'variant', 'unknown'),
        'resolution': getattr(model, 'recommended_resolution', 224),
        'parameters': count_parameters(model),
        'parameters_m': count_parameters(model) / 1e6,
        'feature_dim': model.feature_dim,
        'num_blocks': len(model.blocks),
        'stage_indices': list(model.stage_indices),  # Convert to list for serialization
        'use_checkpoint': model.use_checkpoint,
        'torch_version': torch.__version__,
        'supports_use_reentrant': SUPPORTS_USE_REENTRANT
    }


# ============================================================================
# 7. Testing
# ============================================================================

def test_model():
    """Test EfficientNet implementation"""
    print("=" * 80)
    print("Testing EfficientNet Optimized Implementation")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Supports use_reentrant: {SUPPORTS_USE_REENTRANT}")
    print("=" * 80)

    variants = ['b0', 'b1', 'b2', 'b3', 'b4']

    for variant in variants:
        print(f"\n{'='*60}")
        print(f"Testing EfficientNet-{variant.upper()}")
        print(f"{'='*60}")

        # Create model with gradient checkpointing
        model = _create_efficientnet(variant, num_classes=1000, use_checkpoint=True)
        model.eval()

        # Get info
        info = get_model_info(model)
        resolution = info['resolution']

        # Create dummy input
        batch_size = 2
        x = torch.randn(batch_size, 3, resolution, resolution)

        # Forward pass
        with torch.no_grad():
            output = model(x)
            output_with_features, features = model(x, return_features=True)

        # Verify
        print(f"  Input shape:  {tuple(x.shape)}")
        print(f"  Output shape: {tuple(output.shape)}")
        print(f"  Parameters:   {info['parameters_m']:.2f}M")
        print(f"  Resolution:   {resolution}x{resolution}")
        print(f"  Num blocks:   {info['num_blocks']}")
        print(f"  Stage indices: {info['stage_indices']}")
        print(f"  Feature maps: {len(features)} (at stage boundaries)")

        assert output.shape == (batch_size, 1000), f"Output shape mismatch!"

    # Test reset_classifier
    print(f"\n{'='*60}")
    print("Testing reset_classifier")
    print(f"{'='*60}")
    model = efficientnet_b0(num_classes=1000)
    model.reset_classifier(10)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, 10), "reset_classifier failed!"
    print("  reset_classifier: OK")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)


if __name__ == '__main__':
    test_model()

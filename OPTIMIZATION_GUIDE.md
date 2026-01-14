# Deepfake Detection 최적화 작업 지시서

## 환경
- **서버**: RunPod
- **RAM**: 36GB
- **GPU**: CUDA 지원 GPU (VRAM 크기에 따라 설정 조정)

---

## 파일 구조

```
Effi_ViT_DeepFakeImageClassifier/
├── efficientnet_from_scratch.py      # 원본 (결함 있음)
├── efficientnet_optimized.py         # 최적화 버전 (NEW)
├── train_deepfake_detection.py       # 원본 (결함 있음)
├── train_deepfake_optimized.py       # 최적화 버전 (NEW)
└── OPTIMIZATION_GUIDE.md             # 이 문서
```

---

## 1단계: 환경 설정

### 1.1 필수 패키지 설치

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install datasets transformers
pip install scikit-learn
pip install matplotlib tqdm pillow
```

### 1.2 GPU 확인

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

## 2단계: VRAM에 따른 최적 설정

### 2.1 설정 가이드

| VRAM | batch_size | accumulation_steps | effective_batch | use_checkpoint |
|------|------------|-------------------|-----------------|----------------|
| 8GB  | 4          | 8                 | 32              | True           |
| 12GB | 8          | 4                 | 32              | True           |
| 16GB | 12         | 3                 | 36              | True           |
| 24GB | 16         | 2                 | 32              | True (권장)    |
| 40GB | 24         | 2                 | 48              | False          |
| 80GB | 32         | 2                 | 64              | False          |

### 2.2 RunPod 36GB RAM 권장 설정

```python
config = {
    'num_epochs': 30,
    'batch_size': 16,
    'accumulation_steps': 2,  # Effective batch = 32
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'image_size': 380,
    'save_dir': 'checkpoints_deepfake',
    'patience': 7,
    'use_checkpoint': True,
    'seed': 42
}
```

---

## 3단계: 학습 실행

### 3.1 기본 실행

```bash
python train_deepfake_optimized.py
```

### 3.2 커스텀 설정으로 실행

```python
from train_deepfake_optimized import train_deepfake_detector

model, history = train_deepfake_detector(
    num_epochs=30,
    batch_size=16,
    accumulation_steps=2,
    learning_rate=1e-4,
    weight_decay=1e-5,
    image_size=380,
    save_dir='my_checkpoints',
    patience=7,
    use_checkpoint=True,
    seed=42
)
```

### 3.3 백그라운드 실행 (SSH 연결 끊어져도 계속)

```bash
nohup python train_deepfake_optimized.py > training.log 2>&1 &
tail -f training.log  # 로그 확인
```

---

## 4단계: 추론 (Inference)

### 4.1 단일 이미지 예측

```python
import torch
from train_deepfake_optimized import predict_image, efficientnet_b4

# 모델 로드
device = torch.device('cuda')
model = efficientnet_b4(num_classes=2)
checkpoint = torch.load('checkpoints_deepfake/.../best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# 예측
prediction, confidence = predict_image(model, 'test_image.jpg', device)
label = 'Fake' if prediction == 0 else 'Real'
print(f"{label}: {confidence:.2%}")
```

### 4.2 배치 예측

```python
from train_deepfake_optimized import predict_batch

image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg', ...]
results = predict_batch(model, image_paths, device, batch_size=16)

for path, pred, conf in results:
    label = 'Fake' if pred == 0 else 'Real'
    print(f"{path}: {label} ({conf:.2%})")
```

---

## 5단계: 결과 확인

### 5.1 저장되는 파일들

```
checkpoints_deepfake/efficientnet_b4_YYYYMMDD_HHMMSS/
├── best_model.pth          # 최고 성능 모델
├── checkpoint_epoch_5.pth  # 5 에폭마다 체크포인트
├── checkpoint_epoch_10.pth
├── config.json             # 학습 설정
├── history.json            # 학습 히스토리
├── results.json            # 최종 결과
└── training_curves.png     # 학습 그래프
```

### 5.2 결과 분석

```python
import json
import matplotlib.pyplot as plt

# 히스토리 로드
with open('checkpoints_deepfake/.../history.json') as f:
    history = json.load(f)

# 결과 로드
with open('checkpoints_deepfake/.../results.json') as f:
    results = json.load(f)

print(f"Best Epoch: {results['best_epoch']}")
print(f"Best Val Acc: {results['best_val_acc']:.4f}")
print(f"Test Acc: {results['test_acc']:.4f}")
```

---

## 적용된 최적화 목록

### EfficientNet (`efficientnet_optimized.py`)

| # | 결함 | 수정 내용 |
|---|------|----------|
| 1 | total_blocks 계산 오류 | `depth_mult` 적용된 실제 블록 수 계산 |
| 2 | feature index 하드코딩 | `stage_indices` 동적 계산 |
| 3 | SE reduction 기준 오류 | `base_channels` 파라미터로 input channels 기준 |
| 4 | pretrained 미사용 | 경고 메시지 추가 |
| 5 | resolution 미적용 | `recommended_resolution` 속성 추가 |
| + | 메모리 최적화 | `nn.SiLU(inplace=True)`, gradient checkpointing |

### Deepfake Detection (`train_deepfake_optimized.py`)

| # | 결함 | 수정 내용 |
|---|------|----------|
| 1 | 레이블 불일치 | docstring 및 출력 수정 (0=Fake, 1=Real) |
| 2 | Test set balancing | 원본 분포 유지 |
| 3 | Early stopping 없음 | patience counter 추가 |
| 4 | sklearn import 위치 | 파일 상단으로 이동 |
| 5 | seed 불완전 | `set_seed()` 함수로 전체 seed 고정 |
| 6 | Mixed precision 없음 | `autocast` + `GradScaler` 추가 |
| 7 | drop_last 미설정 | `drop_last=True` 추가 |
| 8 | Augmentation 부족 | JPEG, Blur, Downscale, Erasing 추가 |
| + | Gradient Accumulation | 메모리 제한 시 effective batch 증가 |
| + | Gradient Clipping | 학습 안정성 향상 |
| + | Label Smoothing | 과적합 방지 |
| + | Warmup + Cosine LR | 수렴 개선 |

---

## 예상 성능

### 학습 시간 (A100 80GB 기준)
- 에폭당: ~15분
- 전체 (30 에폭, 조기종료 고려): ~4-6시간

### 예상 정확도
- Validation: 92-96%
- Test: 90-94%

### 메모리 사용량
- GPU VRAM: ~12-16GB (batch=16, checkpoint=True)
- System RAM: ~8-12GB (DataLoader)

---

## 트러블슈팅

### CUDA Out of Memory

```python
# 1. batch_size 감소
config['batch_size'] = 8

# 2. accumulation_steps 증가
config['accumulation_steps'] = 4

# 3. 이미지 크기 감소 (성능 저하 주의)
config['image_size'] = 300
```

### DataLoader Worker 오류 (Windows)

```python
# num_workers=0으로 설정
train_loader = DataLoader(..., num_workers=0)
```

### 학습이 멈춤

```bash
# 로그 확인
tail -100 training.log

# GPU 상태 확인
nvidia-smi
```

---

## 참고사항

- 레이블 규칙: **0 = Fake (딥페이크)**, **1 = Real (진짜)**
- Test set은 원본 분포를 유지하여 실제 환경 성능을 정확히 측정
- Early stopping으로 과적합 방지 (patience=7)
- 모든 random seed 고정으로 재현 가능

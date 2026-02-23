"""
FaceswapDetection - visualize.py

학습된 모델로 샘플 이미지를 분석하고 경계 탐지 과정을 시각화.
각 이미지에 대해 얼굴 검출 -> 경계 추출 -> 예측 결과를 단계별로 표시.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from utils.face_detection import FaceDetection
from utils.boundary_extract import BoundaryExtractor

print("="*60)
print("FaceswapDetection - Visualization")
print("="*60)

# ========================================
# 1. 모델 로드
# ========================================
class FacialBoundaryDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(pretrained=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 2)
    
    def forward(self, x):
        return self.resnet(x)

print("\nLoading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FacialBoundaryDetector()
model.load_state_dict(torch.load('models/facial_boundary_detector.pth', map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

face_detector = FaceDetection()
boundary_extractor = BoundaryExtractor(boundary_width=20)

print("Ready.")

# ========================================
# 2. 이미지 분석
# ========================================
def analyze_image(img_path):
    """단일 이미지에 대해 탐지 파이프라인 실행 및 시각화 데이터 반환"""
    
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 얼굴 검출 (70% 축소)
    face_mask, face_bbox = face_detector.detect_face(str(img_path), shrink_ratio=0.7)
    
    if face_mask is None:
        return None
    
    # 경계 추출
    boundary_mask, boundary_pixels = boundary_extractor.extract(face_mask)
    
    if boundary_mask is None:
        return None
    
    # 경계 영역 crop
    ys, xs = np.where(boundary_mask > 0)
    margin = 15
    y_min = max(0, ys.min() - margin)
    y_max = min(img_rgb.shape[0], ys.max() + margin)
    x_min = max(0, xs.min() - margin)
    x_max = min(img_rgb.shape[1], xs.max() + margin)
    
    boundary_crop = img_rgb[y_min:y_max, x_min:x_max].copy()
    boundary_mask_crop = boundary_mask[y_min:y_max, x_min:x_max]
    boundary_crop[boundary_mask_crop == 0] = 0
    
    # CNN 예측
    boundary_img = Image.fromarray(boundary_crop)
    img_tensor = transform(boundary_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]
    
    real_prob = probs[0].item() * 100
    fake_prob = probs[1].item() * 100
    prediction = 'FAKE' if fake_prob > 50 else 'REAL'
    
    # 시각화용 이미지 생성
    img_with_box = img_rgb.copy()
    x, y, w, h = face_bbox
    cv2.rectangle(img_with_box, (x, y), (x+w, y+h), (0, 255, 0), 3)
    
    face_mask_rgb = np.zeros_like(img_rgb)
    face_mask_rgb[face_mask > 0] = [255, 255, 255]
    
    boundary_mask_rgb = np.zeros_like(img_rgb)
    boundary_mask_rgb[boundary_mask > 0] = [255, 0, 0]
    
    boundary_overlay = img_rgb.copy()
    boundary_overlay[boundary_mask > 0] = [255, 0, 0]
    boundary_vis = cv2.addWeighted(img_rgb, 0.7, boundary_overlay, 0.3, 0)
    
    return {
        'original': img_rgb,
        'with_box': img_with_box,
        'face_mask': face_mask_rgb,
        'boundary_mask': boundary_mask_rgb,
        'boundary_overlay': boundary_vis,
        'boundary_crop': boundary_crop,
        'prediction': prediction,
        'real_prob': real_prob,
        'fake_prob': fake_prob,
        'bbox': face_bbox
    }

# ========================================
# 3. 테스트 이미지 선택
# ========================================
print("\nLoading test images...")

REAL_PATH = Path(r"C:\DeepHunter\detection\data\deepfakevsrealclassification_real")
FAKE_PATH = Path(r"C:\DeepHunter\detection\data\deepfakevsrealclassification_deepfake")

real_images = list(REAL_PATH.glob("*.jpg")) + list(REAL_PATH.glob("*.png"))
if len(real_images) == 0:
    real_images = list(REAL_PATH.rglob("*.jpg")) + list(REAL_PATH.rglob("*.png"))

fake_images = list(FAKE_PATH.glob("*.jpg"))

# 학습에 사용되지 않은 이미지 사용
real_samples = real_images[500:503]
fake_samples = fake_images[500:503]

print(f"  Real: {len(real_samples)}")
print(f"  Fake: {len(fake_samples)}")

# ========================================
# 4. 분석
# ========================================
print("\nAnalyzing images...")
real_results = []
fake_results = []
real_errors = []
fake_errors = []

for img_path in tqdm(real_samples, desc="Real"):
    result = analyze_image(img_path)
    if result:
        result['filename'] = img_path.name
        real_results.append(result)
    else:
        real_errors.append(img_path.name)

for img_path in tqdm(fake_samples, desc="Fake"):
    result = analyze_image(img_path)
    if result:
        result['filename'] = img_path.name
        fake_results.append(result)
    else:
        fake_errors.append(img_path.name)

print(f"\nResults:")
print(f"  Real: {len(real_results)}/{len(real_samples)} processed")
if real_errors:
    print(f"  Failed: {real_errors}")
print(f"  Fake: {len(fake_results)}/{len(fake_samples)} processed")
if fake_errors:
    print(f"  Failed: {fake_errors}")

# ========================================
# 5. 시각화
# ========================================
total_rows = len(real_results) + len(fake_results)
print(f"\nGenerating visualization for {total_rows} images...")

fig = plt.figure(figsize=(20, total_rows * 2.5))

for idx, (result, label) in enumerate([(r, 'REAL') for r in real_results] + 
                                       [(r, 'FAKE') for r in fake_results]):
    row = idx
    
    ax1 = plt.subplot(total_rows, 6, row*6 + 1)
    ax1.imshow(result['original'])
    emoji = "OK" if result['prediction'] == label else "X"
    ax1.set_title(f"{emoji} {result['filename']}\nActual: {label}", fontsize=8)
    ax1.axis('off')
    
    ax2 = plt.subplot(total_rows, 6, row*6 + 2)
    ax2.imshow(result['with_box'])
    ax2.set_title(f"Face Detection\n(70% shrink)", fontsize=8)
    ax2.axis('off')
    
    ax3 = plt.subplot(total_rows, 6, row*6 + 3)
    ax3.imshow(result['face_mask'])
    ax3.set_title(f"Face Region", fontsize=8)
    ax3.axis('off')
    
    ax4 = plt.subplot(total_rows, 6, row*6 + 4)
    ax4.imshow(result['boundary_mask'])
    ax4.set_title(f"Boundary Extract", fontsize=8)
    ax4.axis('off')
    
    ax5 = plt.subplot(total_rows, 6, row*6 + 5)
    ax5.imshow(result['boundary_overlay'])
    ax5.set_title(f"Boundary Overlay", fontsize=8)
    ax5.axis('off')
    
    ax6 = plt.subplot(total_rows, 6, row*6 + 6)
    ax6.imshow(result['boundary_crop'])
    
    is_correct = result['prediction'] == label
    bg_color = 'lightgreen' if is_correct else 'lightcoral'
    text_color = 'darkgreen' if is_correct else 'darkred'
    
    prob_text = f"Real: {result['real_prob']:.1f}%\nFake: {result['fake_prob']:.1f}%"
    ax6.text(0.88, 0.12, prob_text,
             transform=ax6.transAxes, ha='right', va='top',
             fontsize=8, color='white', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.7, edgecolor='white'))
    
    ax6.text(0.5, -0.15, f"-> {result['prediction']} <-", 
             transform=ax6.transAxes, ha='center', va='top',
             fontsize=14, fontweight='bold', color=text_color,
             bbox=dict(boxstyle='round,pad=0.5', facecolor=bg_color, alpha=0.8))
    
    ax6.axis('off')

plt.suptitle("Facial Boundary Detection Results\n"
            f"(Top 3 rows: Real images, Bottom 3 rows: Fake images)", 
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0.02, 1, 0.99])

output_path = Path("data/outputs/facial_boundary_visualization.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
print(f"\nSaved: {output_path}")

plt.show()

# ========================================
# 6. 통계 요약
# ========================================
print("\n" + "="*60)
print("Results Summary")
print("="*60)

all_results = real_results + fake_results
correct = sum(1 for r, label in [(r, 'REAL') for r in real_results] + 
              [(r, 'FAKE') for r in fake_results] if r['prediction'] == label)

print(f"\nAccuracy: {correct}/{len(all_results)} = {correct/len(all_results)*100:.1f}%")

print("\nReal images:")
for r in real_results:
    status_symbol = "[O]" if r['prediction'] == 'REAL' else "[X]"
    print(f"  {status_symbol} {r['filename']:<25s} Predicted: {r['prediction']:<5s} "
          f"(Real: {r['real_prob']:>5.1f}%, Fake: {r['fake_prob']:>5.1f}%)")

print("\nFake images:")
for r in fake_results:
    status_symbol = "[O]" if r['prediction'] == 'FAKE' else "[X]"
    print(f"  {status_symbol} {r['filename']:<25s} Predicted: {r['prediction']:<5s} "
          f"(Real: {r['real_prob']:>5.1f}%, Fake: {r['fake_prob']:>5.1f}%)")

print("\nDone.")
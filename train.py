"""
FaceswapDetection - train.py

얼굴 이목구비 경계 영역을 기반으로 딥페이크를 탐지하는 모델 학습 스크립트.

Pipeline:
    Haar Cascade 얼굴 검출 (shrink_ratio=0.7)
    -> 축소된 얼굴 영역 경계 추출
    -> ResNet50 분류
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2

sys.path.append(str(Path(__file__).parent))
from utils.face_detection import FaceDetection
from utils.boundary_extract import BoundaryExtractor

print("="*60)
print("FaceswapDetection - Boundary-based Classifier")
print("="*60)

# ========================================
# 1. 전처리
# ========================================
def preprocess_and_extract_facial_boundaries(image_paths, labels, output_dir):
    """
    이미지에서 이목구비 중심 경계 영역 추출 후 저장.
    경계 외부 픽셀은 검은색으로 마스킹.
    """
    print("\n[Preprocessing] Extracting facial boundary regions...")
    print("-"*60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading face detector...")
    face_detector = FaceDetection()
    
    processed_paths = []
    processed_labels = []
    skipped = 0
    
    print("\nProcessing images...")
    for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths)):
        try:
            # 얼굴 검출 - shrink_ratio=0.7로 이목구비 중심 영역 추출
            face_mask, face_bbox = face_detector.detect_face(str(img_path), shrink_ratio=0.7)
            
            if face_mask is None or np.sum(face_mask > 0) < 1000:
                skipped += 1
                continue
            
            # 축소된 얼굴 영역의 경계 추출
            extractor = BoundaryExtractor(boundary_width=20)
            boundary_mask, boundary_pixels = extractor.extract(face_mask)
            
            if boundary_mask is None or len(boundary_pixels) < 100:
                skipped += 1
                continue
            
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 경계 영역 bounding box crop
            ys, xs = np.where(boundary_mask > 0)
            
            margin = 15
            y_min = max(0, ys.min() - margin)
            y_max = min(img_rgb.shape[0], ys.max() + margin)
            x_min = max(0, xs.min() - margin)
            x_max = min(img_rgb.shape[1], xs.max() + margin)
            
            boundary_crop = img_rgb[y_min:y_max, x_min:x_max].copy()
            boundary_mask_crop = boundary_mask[y_min:y_max, x_min:x_max]
            
            # 경계 외부 마스킹
            boundary_crop[boundary_mask_crop == 0] = 0
            
            if boundary_crop.shape[0] < 50 or boundary_crop.shape[1] < 50:
                skipped += 1
                continue
            
            label_str = "fake" if label == 1 else "real"
            output_path = output_dir / f"{label_str}_{img_path.stem}.jpg"
            
            boundary_img = Image.fromarray(boundary_crop)
            boundary_img.save(output_path, quality=95)
            
            processed_paths.append(output_path)
            processed_labels.append(label)
            
        except Exception as e:
            skipped += 1
            continue
    
    print(f"\nPreprocessing complete: {len(processed_paths):,} images")
    print(f"  Real  : {sum(1 for l in processed_labels if l == 0):,}")
    print(f"  Fake  : {sum(1 for l in processed_labels if l == 1):,}")
    print(f"  Skipped: {skipped:,}")
    
    return processed_paths, processed_labels

# ========================================
# 2. Dataset
# ========================================
class BoundaryDataset(Dataset):
    """경계 이미지 Dataset"""
    
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            img_tensor = self.transform(img)
            return img_tensor, self.labels[idx]
        except:
            return torch.zeros(3, 224, 224), self.labels[idx]

# ========================================
# 3. 모델
# ========================================
class FacialBoundaryDetector(nn.Module):
    """ResNet50 기반 경계 artifact 분류 모델"""
    
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 2)
    
    def forward(self, x):
        return self.resnet(x)

# ========================================
# 4. 학습
# ========================================
def train_model(train_loader, val_loader, device, epochs=5):
    """모델 학습 및 best checkpoint 저장"""
    
    model = FacialBoundaryDetector().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print('='*60)
        
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        val_acc = 100. * val_correct / val_total
        auc = roc_auc_score(all_labels, all_probs)
        
        print(f"\nTrain Acc: {train_acc:.2f}%")
        print(f"Val Acc:   {val_acc:.2f}%")
        print(f"Val AUC:   {auc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path('models').mkdir(exist_ok=True)
            torch.save(model.state_dict(), 'models/facial_boundary_detector.pth')
            print(f"Best model saved. (val_acc: {best_val_acc:.2f}%)")
    
    return model

# ========================================
# 5. 메인
# ========================================
if __name__ == '__main__':
    print("\nLoading dataset...")
    
    REAL_PATH = Path(r"C:\DeepHunter\detection\data\ffhq_real")
    FAKE_PATH = Path(r"C:\DeepHunter\detection\data\flickr_deepfake")
    
    real_images = list(REAL_PATH.glob("*.jpg")) + list(REAL_PATH.glob("*.png"))
    fake_images = list(FAKE_PATH.glob("*.jpg"))
    
    if len(real_images) == 0:
        real_images = list(REAL_PATH.rglob("*.jpg")) + list(REAL_PATH.rglob("*.png"))
    
    print(f"  Real: {len(real_images):,}")
    print(f"  Fake: {len(fake_images):,}")
    
    n_samples = int(input("\nSamples per class (recommended: 300-500): ").strip() or "300")
    
    real_samples = real_images[:n_samples]
    fake_samples = fake_images[:n_samples]
    
    all_paths = real_samples + fake_samples
    all_labels = [0] * len(real_samples) + [1] * len(fake_samples)
    
    boundary_dir = Path("data/facial_boundary_crops")
    
    print(f"\n{'='*60}")
    print("Extracting facial boundary regions...")
    print("  Haar Cascade -> 70% shrink -> boundary extraction")
    print('='*60)
    
    if not (boundary_dir / "done.txt").exists():
        processed_paths, processed_labels = preprocess_and_extract_facial_boundaries(
            all_paths, all_labels, boundary_dir
        )
        
        if len(processed_paths) > 0:
            with open(boundary_dir / "done.txt", "w") as f:
                f.write("done")
    else:
        print("Using cached preprocessed data.")
        processed_paths = list(boundary_dir.glob("*.jpg"))
        processed_labels = [1 if 'fake' in p.stem else 0 for p in processed_paths]
    
    print(f"\nBoundary images: {len(processed_paths):,}")
    
    if len(processed_paths) < 100:
        print("\nNot enough images to train.")
        print("  -> Face detection may have failed on most images.")
        print("  -> Try different images or increase sample count.")
        sys.exit(1)
    
    # Train/Val/Test split
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        processed_paths, processed_labels, test_size=0.15, random_state=42, stratify=processed_labels
    )
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=0.176, random_state=42, stratify=train_val_labels
    )
    
    print(f"\nTrain: {len(train_paths):,}")
    print(f"Val:   {len(val_paths):,}")
    print(f"Test:  {len(test_paths):,}")
    
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = BoundaryDataset(train_paths, train_labels, transform_train)
    val_dataset = BoundaryDataset(val_paths, val_labels, transform_test)
    test_dataset = BoundaryDataset(test_paths, test_labels, transform_test)
    
    batch_size = int(input("\nBatch size (recommended: 32): ").strip() or "32")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    epochs = int(input("\nEpochs (recommended: 3-5): ").strip() or "3")
    
    print("\nStarting training...")
    model = train_model(train_loader, val_loader, device, epochs)
    
    # 최종 평가
    print("\nRunning final test...")
    model.load_state_dict(torch.load('models/facial_boundary_detector.pth'))
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = 100. * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    auc = roc_auc_score(all_labels, all_probs)
    
    print("\n" + "="*60)
    print("Test Results")
    print("="*60)
    print(f"Accuracy : {accuracy:.2f}%")
    print(f"AUC      : {auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))
    
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(f"              Pred Real  Pred Fake")
    print(f"Actual Real   {cm[0,0]:>8d}  {cm[0,1]:>10d}")
    print(f"Actual Fake   {cm[1,0]:>8d}  {cm[1,1]:>10d}")
    
    print("\n" + "="*60)
    print("Comparison")
    print("="*60)
    print(f"Face-background boundary (prev) : ~60%")
    print(f"Facial feature boundary (current): {accuracy:.2f}%")
    
    if accuracy > 85:
        print("\nHigh accuracy achieved with facial feature boundaries.")
        print("  -> Fine-grained inconsistencies in deepfake regions detected.")
    elif accuracy > 70:
        print("\nImproved over baseline, but further gains possible.")
        print("  -> Consider adding color distribution or frequency features.")
    else:
        print("\nFacial boundary alone insufficient.")
        print("  -> Alternative approach needed.")
    
    print("\nDone.")
# FaceswapDetection

딥페이크 얼굴의 이목구비 경계에서 합성 흔적을 탐지합니다.

---

## 파일 구조

```
FaceswapDetection/
├── demo.py          # Gradio 웹 데모
├── train.py         # 학습 메인
├── visualize.py     # 결과 시각화
│
└── utils/
    ├── face_detection.py    # 얼굴 검출 (Haar Cascade)
    └── boundary_extract.py  # 경계선 추출
```

---

## 개발 히스토리

### 1차 시도 (기각)
- **방식**: DeepLabV3 Segmentation → 얼굴-배경 경계 추출 → CNN
- **문제**: 배경 텍스처 노이즈가 커서 신호가 묻힘
- **정확도**: ~60%

### 2차 시도 (현재)
- **방식**: Haar Cascade 얼굴 검출 → 70% 축소 → 이목구비 경계 추출 → ResNet50
- **핵심 개선**: 얼굴-배경 경계 대신 얼굴 내부(눈/코/입 주변) 경계에 집중
- **정확도**: 98.84% (FFHQ), 76.20% (외부 데이터셋)

---

## 파이프라인

```
이미지
  └─► FaceDetection (shrink_ratio=0.7)   # 얼굴 박스 70% 축소
        └─► BoundaryExtractor            # 이목구비 주변 경계 추출
              └─► ResNet50               # 진짜/가짜 분류
```

---

## 사용법

### 학습
```bash
python train.py
```

### 결과 시각화
```bash
python visualize.py
```

### 웹 데모
```bash
python demo.py
```
<img width="1440" height="701" alt="image" src="https://github.com/user-attachments/assets/c87e5c0b-8466-4847-b468-ab5144d7fe53" />

---

## 데이터셋
- **Real**: FFHQ (Flickr-Faces-HQ)
- **Fake**: FFHQ 기반 Face swap 이미지 (111K)

---

## 성능

| 데이터셋 | Accuracy | AUC |
|---|---|---|
| FFHQ (학습 분포) | 98.84% | 0.9543 |
| 외부 데이터셋 | 76.20% | - |

> 외부 데이터셋 정확도 차이는 일반화 문제로, 추가 데이터 다양성 확보 필요

---

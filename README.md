# Faceswap Detection via Facial Feature Boundary Analysis

Face swap 이미지에서 눈·코·입 주변 경계에 남은 합성 흔적을 찾아 딥페이크를 판별합니다.

<img width="800" alt="Facial Boundary Detection Results" src="https://github.com/user-attachments/assets/8e368df9-cc9d-4470-95c1-6f1a315d8159" />

> 요즘 딥페이크는 얼굴 전체를 새로 만들지 않고 일부만 바꿔 자연스럽게 섞습니다. 그래서 이미지 전체를 봐서는 티가 거의 나지 않습니다.
> 이 프로젝트는 "진짜처럼 보이는가"보다 **"어디를 건드렸는가"**를 보는 데서 출발했습니다.

---

## 성능

| 데이터셋 | 정확도 | Real 검출률 | Fake 검출률 |
|---|---|---|---|
| FFHQ (학습) | **98.84%** | - | - |
| Validation | 76.20% | 99.47% | 49.08% |
| 20K (Wild) | 65.36% | 62.50% | 68.92% |

**AUC**: 0.9543

- 고화질 이미지에서는 경계에 남은 합성 흔적이 뚜렷해서 정확도가 높습니다.
- 압축된 wild 이미지에서는 흔적이 뭉개져 정확도가 떨어집니다.

---

## 개발 과정

| | Phase 1. VAE 재구성 | v1. 얼굴–배경 경계 | **v2. 이목구비 경계 (현재)** |
|---|---|---|---|
| 아이디어 | 정상 얼굴을 학습한 VAE로 재구성해 보고, 재구성이 잘 안 되는 이미지를 가짜로 판단 | 얼굴과 배경이 맞닿은 외곽에서 흔적을 찾음 | 얼굴 안쪽, 눈·코·입 주변 경계에서 흔적을 찾음 |
| 방법 | β-VAE (β=4.0) | Segmentation → CNN | Haar Cascade → 70% 축소 → 경계 추출 → ResNet50 |
| 결과 | KL을 57% 줄였지만 고화질 딥페이크는 잘 걸러내지 못함 | 약 60%. 배경 무늬가 복잡해 흔적이 묻힘 | **98.84%** |

VAE 실험 이후로는 이미지를 다시 그려 보는 방식을 버리고, 합성한 자리에 남은 흔적을 직접 찾는 쪽으로 방향을 바꿨습니다. 처음에는 얼굴 외곽을 봤는데 배경이 복잡해서 신호가 약했습니다. 범위를 얼굴 안쪽으로 좁히자 눈·코·입 경계에서 흔적이 또렷하게 드러났습니다.

### 파이프라인

```
Input Image
  └─► FaceDetection (Haar Cascade, shrink_ratio=0.7)   # 얼굴 박스를 70%로 줄여 배경을 덜어냄
        └─► BoundaryExtractor                          # 눈·코·입 주변 경계만 잘라냄
              └─► ResNet50                             # 진짜/가짜 분류
```

---

## 데이터셋

- **Real**: FFHQ (Flickr-Faces-HQ)
- **Fake**: FFHQ로 만든 Face swap 이미지 111K장
- 인터넷에서 돌아다니는, 화질과 압축 정도가 제각각인 이미지도 포함했습니다.

<!-- 데이터셋 예시 (원본 2장 + 합성 결과) 이미지 -->
<img width="1205" height="410" alt="image" src="https://github.com/user-attachments/assets/cd8e8f2d-b898-42c9-84d5-140cf2f59b4f" />

*왼쪽 두 장이 원본이고, 오른쪽이 두 얼굴을 합성한 결과입니다. 눈으로는 합성한 티가 거의 나지 않습니다.*

---

## 실행 방법

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

<!-- Gradio 데모 스크린샷 -->
<img width="1440" height="1201" alt="image" src="https://github.com/user-attachments/assets/f1141f10-1fef-42b0-a98f-9a68a674c059" />


데모에서는 슬라이더로 threshold를 바꿀 수 있습니다.

| Threshold | 모드 | 이럴 때 쓰세요 |
|---|---|---|
| 30–40% | Aggressive | 가짜를 하나라도 더 잡아야 할 때 |
| 50% | Default | 기본값. 진짜 이미지를 가짜로 잘못 판정하지 않는 쪽 |
| 60–70% | Conservative | 진짜를 가짜로 판정하는 일을 최대한 줄여야 할 때 |

> 어디에 쓰느냐에 따라 감수할 수 있는 오판이 다르기 때문에, 기준을 하나로 고정하지 않았습니다.

---

## 폴더 구조

```
FaceswapDetection/
├── demo.py                  # Gradio 웹 데모 (threshold 조절)
├── train.py                 # 학습
├── visualize.py             # 결과 시각화
└── utils/
    ├── face_detection.py    # 얼굴 검출 및 축소 (Haar Cascade)
    └── boundary_extract.py  # 이목구비 경계 추출
```

---

## 한계

- **압축에 약함**: wild 이미지에서는 흔적이 뭉개져 정확도가 65.36%까지 떨어집니다.
- **Real 쪽으로 치우침**: Validation에서 Real은 99.47%를 맞혔지만 Fake는 49.08%만 잡았습니다. 기본 설정이 진짜 이미지를 보호하는 쪽이라, 가짜를 더 많이 잡아야 한다면 threshold를 낮춰 쓰는 게 좋습니다.
- **앞으로 할 일**: 압축 augmentation을 넣고, 다른 face swap 기법으로 만든 데이터도 학습해서 일반화 성능을 높일 계획입니다.

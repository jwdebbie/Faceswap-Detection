"""
FaceswapDetection - demo.py

Gradio 기반 웹 인터페이스. 단일 이미지 및 배치 처리 지원.
"""

import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import sys
import pandas as pd
from datetime import datetime

sys.path.append(str(Path(__file__).parent))
from utils.face_detection import FaceDetection
from utils.boundary_extract import BoundaryExtractor

class FacialBoundaryDetector(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.resnet = models.resnet50(weights=None)
        except TypeError:
            self.resnet = models.resnet50(pretrained=False)
        
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 2)
    
    def forward(self, x):
        return self.resnet(x)

print("Loading model...")
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

def analyze_single_image(image, filename="unknown", threshold=50.0):
    """
    단일 이미지 분석.
    
    Args:
        threshold: 가짜 판정 임계값 (기본 50%)
    """
    
    if image is None:
        return None, None, "분석불가", 0.0, 0.0, "이미지 없음"
    
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image
    
    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = img
    
    temp_path = Path(f"temp_{filename}.jpg")
    cv2.imwrite(str(temp_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    
    try:
        face_mask, face_bbox = face_detector.detect_face(str(temp_path), shrink_ratio=0.7)
        
        if face_mask is None:
            return img_rgb, None, "검출실패", 0.0, 0.0, "얼굴 검출 실패"
        
        boundary_mask, _ = boundary_extractor.extract(face_mask)
        
        if boundary_mask is None:
            return img_rgb, None, "검출실패", 0.0, 0.0, "경계선 추출 실패"
        
        ys, xs = np.where(boundary_mask > 0)
        if len(ys) < 100:
            return img_rgb, None, "검출실패", 0.0, 0.0, "경계선 영역 부족"
        
        margin = 15
        y_min = max(0, ys.min() - margin)
        y_max = min(img_rgb.shape[0], ys.max() + margin)
        x_min = max(0, xs.min() - margin)
        x_max = min(img_rgb.shape[1], xs.max() + margin)
        
        boundary_crop = img_rgb[y_min:y_max, x_min:x_max].copy()
        boundary_mask_crop = boundary_mask[y_min:y_max, x_min:x_max]
        boundary_crop[boundary_mask_crop == 0] = 0
        
        if boundary_crop.shape[0] < 50 or boundary_crop.shape[1] < 50:
            return img_rgb, None, "검출실패", 0.0, 0.0, "분석 영역 너무 작음"
        
        boundary_img = Image.fromarray(boundary_crop)
        img_tensor = transform(boundary_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0]
        
        real_prob = float(probs[0].item() * 100)
        fake_prob = float(probs[1].item() * 100)
        
        if fake_prob > threshold:
            prediction = "FAKE"
            confidence = fake_prob
        else:
            prediction = "REAL"
            confidence = real_prob
        
        boundary_overlay = img_rgb.copy()
        boundary_overlay[boundary_mask > 0] = [255, 0, 0]
        boundary_vis = cv2.addWeighted(img_rgb, 0.7, boundary_overlay, 0.3, 0)
        
        x, y, w, h = face_bbox
        cv2.rectangle(boundary_vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        return boundary_vis, boundary_crop, prediction, real_prob, fake_prob, "분석 완료"
        
    except Exception as e:
        return img_rgb, None, "분석불가", 0.0, 0.0, f"오류: {str(e)}"
        
    finally:
        if temp_path.exists():
            temp_path.unlink()

def analyze_batch(files, threshold, progress=gr.Progress()):
    """
    여러 이미지 배치 분석.
    
    Args:
        threshold: 가짜 판정 임계값
    """
    
    if files is None or len(files) == 0:
        return None, "이미지를 업로드해주세요", None
    
    results = []
    gallery_images = []
    
    progress(0, desc="분석 시작...")
    
    for idx, file in enumerate(files):
        progress((idx + 1) / len(files), desc=f"분석 중... ({idx + 1}/{len(files)})")
        
        if isinstance(file, str):
            filename = Path(file).name
            image = Image.open(file)
        else:
            filename = file.name if hasattr(file, 'name') else f"image_{idx}"
            image = Image.open(file) if hasattr(file, 'read') else Image.fromarray(file)
        
        img_array = np.array(image)
        
        vis_img, boundary_img, prediction, real_prob, fake_prob, status = \
            analyze_single_image(img_array, filename, threshold)
        
        if prediction == "REAL":
            result_kr = "진짜"
            result_emoji = "[R]"
        elif prediction == "FAKE":
            result_kr = "가짜"
            result_emoji = "[F]"
        elif prediction == "검출실패":
            result_kr = "검출실패"
            result_emoji = "[-]"
        else:
            result_kr = "분석불가"
            result_emoji = "[?]"
        
        results.append({
            "파일명": filename,
            "판정": result_kr,
            "진짜 확률": f"{real_prob:.1f}%",
            "가짜 확률": f"{fake_prob:.1f}%",
            "상세": status
        })
        
        if vis_img is not None:
            vis_img_copy = vis_img.copy()
            h, w = vis_img_copy.shape[:2]
            
            if prediction == "FAKE":
                bg_color = (220, 38, 38)
                text_main = "FAKE"
                text_sub = f"{fake_prob:.1f}%"
            elif prediction == "REAL":
                bg_color = (22, 163, 74)
                text_main = "REAL"
                text_sub = f"{real_prob:.1f}%"
            else:
                bg_color = (107, 114, 128)
                text_main = result_kr
                text_sub = ""
            
            overlay = vis_img_copy.copy()
            cv2.rectangle(overlay, (0, 0), (w, 100), bg_color, -1)
            cv2.addWeighted(overlay, 0.85, vis_img_copy, 0.15, 0, vis_img_copy)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(vis_img_copy, text_main, (20, 55), 
                       font, 1.8, (255, 255, 255), 5)
            
            if text_sub:
                cv2.putText(vis_img_copy, f"Confidence: {text_sub}", (20, 85), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            gallery_images.append((vis_img_copy, f"{result_emoji} {filename}"))
    
    df = pd.DataFrame(results)
    
    total = len(results)
    real_count = sum(1 for r in results if r["판정"] == "진짜")
    fake_count = sum(1 for r in results if r["판정"] == "가짜")
    fail_count = sum(1 for r in results if r["판정"] in ["검출실패", "분석불가"])
    
    summary = f"""
Analysis Complete
{'='*40}

Threshold : {threshold:.1f}%
Total     : {total}
{'─' * 40}
Real      : {real_count:3d}  ({real_count/total*100:5.1f}%)
Fake      : {fake_count:3d}  ({fake_count/total*100:5.1f}%)
Failed    : {fail_count:3d}  ({fail_count/total*100:5.1f}%)
{'='*40}
Fake if fake_prob > {threshold:.1f}%
Real if fake_prob <= {threshold:.1f}%
    """
    
    return df, summary, gallery_images

def create_interface():
    """Gradio 인터페이스 생성"""
    
    with gr.Blocks(title="FaceswapDetection") as demo:
        gr.Markdown("""
        # FaceswapDetection
        ### Deepfake detection based on facial boundary artifact analysis
        """)
        
        with gr.Tabs():
            with gr.Tab("Single Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        single_input = gr.Image(label="Upload Image", type="numpy")
                        
                        single_threshold = gr.Slider(
                            minimum=30,
                            maximum=70,
                            value=50,
                            step=5,
                            label="Threshold (%)",
                            info="Lower = more aggressive detection"
                        )
                        
                        single_btn = gr.Button("Analyze", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        with gr.Row():
                            single_output_vis = gr.Image(label="Boundary Analysis")
                            single_output_boundary = gr.Image(label="Extracted Boundary")
                        
                        with gr.Row():
                            single_output_text = gr.Textbox(
                                label="",
                                lines=5,
                                show_label=False,
                                container=True
                            )
                            single_output_conf = gr.Label(
                                label="Confidence", 
                                num_top_classes=2
                            )
                
                def single_analyze_wrapper(image, threshold):
                    if image is None:
                        return None, None, "No image provided.", {"Real": 0, "Fake": 0}
                    
                    vis_img, boundary_img, prediction, real_prob, fake_prob, status = \
                        analyze_single_image(image, threshold=threshold)
                    
                    if prediction == "FAKE":
                        result_text = f"""
Result      : FAKE (Deepfake)
Confidence  : {fake_prob:.2f}%
Threshold   : {threshold:.1f}%
"""
                    elif prediction == "REAL":
                        result_text = f"""
Result      : REAL
Confidence  : {real_prob:.2f}%
Threshold   : {threshold:.1f}%
"""
                    elif prediction == "검출실패":
                        result_text = f"""
Result      : Detection Failed
Detail      : {status}
"""
                    else:
                        result_text = f"""
Result      : Analysis Failed
Detail      : {status}
"""
                    
                    confidence = {
                        "Real": real_prob,
                        "Fake": fake_prob
                    }
                    
                    return vis_img, boundary_img, result_text, confidence
                
                single_btn.click(
                    fn=single_analyze_wrapper,
                    inputs=[single_input, single_threshold],
                    outputs=[single_output_vis, single_output_boundary, 
                            single_output_text, single_output_conf]
                )
            
            with gr.Tab("Batch"):
                gr.Markdown("""
                ### Analyze multiple images at once
                - Ctrl+click or drag & drop to select multiple files
                """)
                
                with gr.Row():
                    batch_input = gr.File(
                        label="Select image files",
                        file_count="multiple",
                        file_types=["image"]
                    )
                
                batch_threshold = gr.Slider(
                    minimum=30,
                    maximum=70,
                    value=50,
                    step=5,
                    label="Threshold (%)",
                    info="30%: aggressive | 50%: default | 70%: conservative"
                )
                
                batch_btn = gr.Button("Run Batch Analysis", variant="primary", size="lg")
                
                gr.Markdown("### Results")
                
                with gr.Row():
                    batch_summary = gr.Textbox(
                        label="Summary", 
                        lines=14,
                        show_label=True
                    )
                
                batch_table = gr.Dataframe(
                    label="Detail",
                    wrap=True
                )
                
                batch_gallery = gr.Gallery(
                    label="Visualization", 
                    columns=3, 
                    height="auto",
                    object_fit="contain"
                )
                
                batch_btn.click(
                    fn=analyze_batch,
                    inputs=[batch_input, batch_threshold],
                    outputs=[batch_table, batch_summary, batch_gallery]
                )
        
        gr.Markdown("""
        ---
        ### Notes
        - Best results with frontal, clearly visible faces
        - Small or angled faces may fail detection
        - When multiple faces are present, the largest is analyzed
        - Changing the threshold will affect results on the same image
        
        ### Threshold Guide
        - **30-40%**: Aggressive (more false positives)
        - **50%**: Default (balanced)
        - **60-70%**: Conservative (more false negatives)
        
        ### Model Performance
        - FFHQ dataset: 98.84% accuracy
        - External dataset: 76.20% accuracy
        - AUC: 0.9543
        """)
    
    return demo

if __name__ == "__main__":
    print("\n" + "="*60)
    print("FaceswapDetection - Web Demo")
    print("="*60)
    print(f"Device: {device}")
    
    demo = create_interface()
    demo.launch(
        share=False,
        inbrowser=True,
        server_port=None
    )
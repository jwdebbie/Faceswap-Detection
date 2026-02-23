"""
FaceswapDetection - utils/face_detection.py

Haar Cascade 기반 얼굴 검출. shrink_ratio로 검출 박스를 축소해
이목구비 중심 영역을 반환한다.
"""

import cv2
import numpy as np

class FaceDetection:
    """얼굴 영역 검출 (축소)"""
    
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
    
    def detect_face(self, image_path, shrink_ratio=0.8):
        """
        얼굴 영역 검출 후 마스크 반환.

        Args:
            image_path  : 입력 이미지 경로
            shrink_ratio: 검출 박스 축소 비율 (0.7 = 이목구비 중심)

        Returns:
            face_mask: 얼굴 영역 마스크 (numpy array)
            bbox     : (x, y, w, h)
        """
        img = cv2.imread(str(image_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None, None
        
        # 가장 큰 얼굴 선택
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # 박스 축소 (중심 기준)
        center_x = x + w // 2
        center_y = y + h // 2
        
        new_w = int(w * shrink_ratio)
        new_h = int(h * shrink_ratio)
        
        new_x = max(0, center_x - new_w // 2)
        new_y = max(0, center_y - new_h // 2)
        
        face_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        face_mask[new_y:new_y+new_h, new_x:new_x+new_w] = 255
        
        return face_mask, (new_x, new_y, new_w, new_h)


if __name__ == "__main__":
    detector = FaceDetection()
    print("Usage:")
    print("  face_mask, bbox = detector.detect_face('image.jpg')")
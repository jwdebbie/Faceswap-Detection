"""
FaceswapDetection - utils/boundary_extract.py

얼굴 마스크에서 경계 영역을 추출한다.
안쪽/바깥쪽 경계를 모두 포함하며, boundary_width로 두께를 조절한다.
"""

import cv2
import numpy as np

class BoundaryExtractor:
    """경계선 추출"""
    
    def __init__(self, boundary_width=20):
        """
        Args:
            boundary_width: 경계 양쪽으로 추출할 픽셀 수
        """
        self.boundary_width = boundary_width
    
    def extract(self, mask):
        """
        마스크에서 경계 영역 추출.

        Args:
            mask: 얼굴 영역 마스크 (0 또는 255)

        Returns:
            boundary_mask  : 경계 영역 마스크
            boundary_pixels: 경계 픽셀 좌표 배열
        """
        contours, _ = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            print("No contours found.")
            return None, []
        
        # 가장 큰 윤곽선 선택
        main_contour = max(contours, key=cv2.contourArea)
        
        boundary_mask = np.zeros_like(mask)
        
        # 안쪽 경계
        cv2.drawContours(
            boundary_mask, 
            [main_contour], 
            -1, 
            255, 
            self.boundary_width
        )
        
        # 바깥쪽 경계 포함
        dilated = cv2.dilate(mask, None, iterations=self.boundary_width//2)
        eroded = cv2.erode(mask, None, iterations=self.boundary_width//2)
        boundary_mask = cv2.bitwise_or(
            boundary_mask,
            cv2.bitwise_xor(dilated, eroded)
        )
        
        boundary_pixels = np.column_stack(np.where(boundary_mask > 0))
        
        return boundary_mask, boundary_pixels
    
    def visualize(self, image, boundary_mask):
        """
        경계를 원본 이미지에 반투명 오버레이.

        Args:
            image        : 원본 이미지
            boundary_mask: 경계 마스크

        Returns:
            result: 경계가 표시된 이미지
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # 경계를 빨간색으로 표시
        overlay = image.copy()
        overlay[boundary_mask > 0] = [255, 0, 0]
        
        result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        
        return result


if __name__ == "__main__":
    extractor = BoundaryExtractor(boundary_width=20)
    print("Usage:")
    print("  boundary_mask, pixels = extractor.extract(mask)")
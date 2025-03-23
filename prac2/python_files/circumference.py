import cv2
import numpy as np

def calculate_circumference(mask):
    mask = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return 0
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    if len(largest_contour) < 5:
        return 0
    
    ellipse = cv2.fitEllipse(largest_contour)
    (center, axes, angle) = ellipse
    a, b = axes[0] / 2, axes[1] / 2
    circumference = np.pi * (3*(a + b) - np.sqrt((3*a + b) * (a + 3*b)))
    
    return circumference

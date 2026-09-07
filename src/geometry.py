import cv2
import numpy as np

def extract_contours(binary_mask):
    """Extracts external topological boundaries from a binary mask."""
    # Ensure mask is uint8
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
        
    contours, _ = cv2.findContours(
        binary_mask, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    return contours

def simplify_polygons(contours, epsilon_factor=0.01):
    """Applies Douglas-Peucker boundary sharpening."""
    simplified = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        simplified.append(approx)
    return simplified

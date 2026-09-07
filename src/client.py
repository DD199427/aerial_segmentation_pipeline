import requests
import cv2
import numpy as np

def send_for_inference(image_array, endpoint_url, api_token):
    """Sends preprocessed image to the segmentation endpoint."""
    _, encoded_img = cv2.imencode('.png', image_array)
    
    headers = {"Authorization": f"Bearer {api_token}"}
    files = {"image": ("aerial_patch.png", encoded_img.tobytes(), "image/png")}
    
    response = requests.post(endpoint_url, headers=headers, files=files)
    response.raise_for_status()
    
    return response.json()

def resize_mask(mask, target_shape):
    """Aligns output masks using nearest-neighbor interpolation."""
    return cv2.resize(mask, target_shape, interpolation=cv2.INTER_NEAREST)

import rasterio
import numpy as np

def load_geotiff(filepath):
    """Loads a GeoTIFF and returns the image array and profile."""
    with rasterio.open(filepath) as src:
        image = src.read()
        profile = src.profile
    return image, profile

def normalize_raster(image):
    """Applies min-max normalization and converts float32 to uint8."""
    img_min = np.nanmin(image)
    img_max = np.nanmax(image)
    
    # Epsilon guard to prevent division by zero
    eps = 1e-6
    normalized = (image - img_min) / (img_max - img_min + eps)
    
    # Scale to 0-255 and convert to uint8
    uint8_img = (normalized * 255).astype(np.uint8)
    
    # Convert from CHW to HWC for standard CV processing
    return np.transpose(uint8_img, (1, 2, 0))

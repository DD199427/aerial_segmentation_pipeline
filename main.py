import os
import argparse
from src.io import load_geotiff, normalize_raster
from src.client import send_for_inference
from src.geometry import extract_contours, simplify_polygons

def main(args):
    print(f"Processing {args.input}...")
    
    # 1. I/O & Preprocessing
    image, profile = load_geotiff(args.input)
    norm_image = normalize_raster(image)
    
    print("Simulating inference... (Replace API client with local PyTorch UNet if needed)")
    
    # 2. API Request
    # response = send_for_inference(norm_image, os.environ.get("ENDPOINT_URL"), os.environ.get("API_TOKEN"))
    
    # Mocking a binary mask to demonstrate the geometry pipeline
    import numpy as np
    mock_mask = np.zeros(norm_image.shape[:2], dtype=np.uint8)
    mock_mask[50:150, 50:150] = 255
    
    # 3. Geometry processing
    contours = extract_contours(mock_mask)
    polygons = simplify_polygons(contours)
    
    print(f"Extracted {len(polygons)} sharpened polygons.")
    print("Pipeline execution complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aerial Segmentation Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input GeoTIFF")
    args = parser.parse_args()
    
    # Create a dummy GeoTIFF for testing if it doesn't exist
    if not os.path.exists(args.input):
        import rasterio
        from rasterio.transform import from_origin
        import numpy as np
        
        print(f"Creating dummy GeoTIFF at {args.input} for testing...")
        os.makedirs(os.path.dirname(args.input), exist_ok=True)
        transform = from_origin(0, 0, 1, 1)
        dummy_data = np.random.rand(3, 256, 256).astype(np.float32)
        with rasterio.open(
            args.input, 'w', driver='GTiff', 
            height=256, width=256, count=3, dtype='float32',
            crs='+proj=latlong', transform=transform
        ) as dst:
            dst.write(dummy_data)
            
    main(args)

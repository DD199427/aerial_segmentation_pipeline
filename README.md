# Aerial Segmentation Pipeline

A robust computer vision pipeline designed for processing high-dynamic-range aerial imagery, running semantic segmentation inference, and extracting sharpened vector boundaries. 

This repository is structured to cleanly separate geospatial I/O, network orchestration, and geometric computer vision tasks, making it highly extensible. It serves as a strong foundation for portfolio demonstrations involving spatial loss mapping and UNet architecture integrations.

## Architecture

* **`src/io.py`:** Handles multi-band GeoTIFF ingestion via `rasterio`. Includes NaN-safe min-max normalization, converting float32 DSM/RGB to standardized uint8 tensors.
* **`src/client.py`:** Orchestrates payload transmission. Can be easily swapped out for a local PyTorch `UNet` model if moving away from an external API endpoint.
* **`src/geometry.py`:** Converts pixel masks into clean polygon geometries using topological contour extraction and Douglas-Peucker approximation via `OpenCV`.
* **`main.py`:** The CLI entrypoint that stitches the pipeline together.

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the pipeline on a raw GeoTIFF:

```bash
python main.py --input data/raw/sample.tif
```

*(Note: If the input file is not found, the script will automatically generate a randomized dummy GeoTIFF raster to demonstrate the pipeline's end-to-end execution.)*

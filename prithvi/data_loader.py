import numpy as np
import rasterio
from pathlib import Path

BAND_KEYS = ["B02", "B03", "B04", "B08", "B11", "B12"]

def load_temporal_stack(base_dir, folders, patch_size=224):
    spectral_frames, ndvi_frames = [], []

    for month in folders:
        month_path = Path(base_dir) / month
        bands = []

        for b_key in BAND_KEYS:
            file = next(month_path.glob(f"*{b_key}*.tiff"))
            with rasterio.open(file) as src:
                arr = src.read(1).astype(np.float32)
                # normalize per-band to [0,1] regardless of source scaling
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)
                bands.append(arr[:patch_size, :patch_size])

        spectral_frames.append(np.stack(bands))

        ndvi_file = next(month_path.glob("*NDVI*.tiff"))
        with rasterio.open(ndvi_file) as src:
            ndvi_arr = src.read(1).astype(np.float32)
            ndvi_arr = (ndvi_arr - ndvi_arr.min()) / (ndvi_arr.max() - ndvi_arr.min() + 1e-10)
            ndvi_frames.append(ndvi_arr[:patch_size, :patch_size])

        print(f"  {month} ✓")

    return np.stack(spectral_frames), np.stack(ndvi_frames)

def calculate_area_stats(mask, resolution=10):
    """
    Calculates the total area in hectares from a binary mask.
    1 hectare = 10,000 m^2
    """
    pixel_count = np.sum(mask > 0.5)  # Count pixels flagged as double crop
    
    # Area in square meters
    area_m2 = pixel_count * (resolution ** 2)
    
    # Convert to hectares
    area_ha = area_m2 / 10000.0
    
    return area_ha, int(pixel_count)
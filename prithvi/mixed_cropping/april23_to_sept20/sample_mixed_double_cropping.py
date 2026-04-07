import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy.signal import savgol_filter, find_peaks
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

# --- CONFIG & PATHS ---
p = os.path.dirname(os.path.abspath(__file__))
dates = ["march", "june", "august"]
real_dates = [0, 92, 152]  # Days from start (approx) for better interpolation
tile_size = 64

def split_into_tiles(img, size):
    h, w = img.shape
    tiles = [img[i:i+size, j:j+size] for i in range(0, h, size) for j in range(0, w, size)]
    positions = [(i, j) for i in range(0, h, size) for j in range(0, w, size)]
    # Filter only full-size tiles
    valid = [t for t in tiles if t.shape == (size, size)]
    valid_pos = [p for idx, p in enumerate(positions) if tiles[idx].shape == (size, size)]
    return valid, valid_pos

all_tile_series = [] 

# --- STEP 1: LOAD & CALCULATE ---
for folder in dates:
    f_list = os.listdir(os.path.join(p, folder))
    
    # Get Red (B04) and NIR (B08)
    r_path = [os.path.join(p, folder, f) for f in f_list if "B04" in f][0]
    n_path = [os.path.join(p, folder, f) for f in f_list if "B08" in f][0]

    with rasterio.open(r_path) as src_r, rasterio.open(n_path) as src_n:
        r = src_r.read(1).astype(float)
        n = src_n.read(1).astype(float)
        
        # NDVI Calculation
        ndvi = (n - r) / (n + r + 1e-10)
        
        # --- CLOUD FILTERING ---
        # If NDVI is suspiciously low in June (like in your image), we flag it
        # In a professional setup, you'd use the SCL/QA band here instead.
        ndvi[ndvi < 0.05] = np.nan 
        
        tiles, pos = split_into_tiles(ndvi, tile_size)
        all_tile_series.append([np.nanmean(t) for t in tiles])
        tile_positions = pos

# --- STEP 2: INTERPOLATION (The Multi-Crop Fix) ---
# Convert to array: Rows = Tiles, Columns = Months
data = np.array(all_tile_series).T 

# Linear interpolation to fill the "June Cloud Gap"
df = pd.DataFrame(data)
df = df.interpolate(axis=1, limit_direction='both')
filled_data = df.to_numpy()

# --- STEP 3: PEAK DETECTION & CLUSTERING ---
peak_counts = []
for series in filled_data:
    # Smooth to avoid "false peaks" from noise
    smoothed = savgol_filter(series, 3, 2) if not np.isnan(series).any() else series
    
    # Find humps (Multi-cropping detection)
    # We lower height because June might still be recovering from the cloud dip
    peaks, _ = find_peaks(smoothed, height=0.15, distance=1) 
    peak_counts.append(len(peaks))

km = KMeans(n_clusters=4, n_init=10).fit(filled_data)

# --- STEP 4: VISUALIZE ---
max_i, max_j = max(p[0] for p in tile_positions)+tile_size, max(p[1] for p in tile_positions)+tile_size
res_map = np.zeros((max_i, max_j))
pk_map = np.zeros((max_i, max_j))

for idx, (i, j) in enumerate(tile_positions):
    res_map[i:i+tile_size, j:j+tile_size] = km.labels_[idx]
    pk_map[i:i+tile_size, j:j+tile_size] = peak_counts[idx]

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(res_map, cmap='tab10'); ax[0].set_title("Clusters")
ax[1].imshow(pk_map, cmap='viridis'); ax[1].set_title("Multi-Crop Peak Count")
plt.show()
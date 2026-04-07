from qgis.core import QgsProject, QgsRasterLayer
import processing
import os
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt

#config
BASE_PATH = r'Y:\14_Zuckerrübe\Sentinel_2\wue'
FIELD_LAYER_NAME = 'single_field — sb_fields_wue_2024'
TARGET_PATCH = '32UPA_0_4'  

# load field
field_layer = QgsProject.instance().mapLayersByName(FIELD_LAYER_NAME)[0]

chronological_dates = []
ndvi_health_scores = []

print(f"Starting Temporal for Patch {TARGET_PATCH}...")

# loop
for date_folder in sorted(os.listdir(BASE_PATH)):
    date_path = os.path.join(BASE_PATH, date_folder)
    if not os.path.isdir(date_path): continue

    patch_path = os.path.join(date_path, TARGET_PATCH)
    bands_path = os.path.join(patch_path, 'bands.tif')
    scl_path   = os.path.join(patch_path, 'scl.tif')

    if not os.path.exists(bands_path) or not os.path.exists(scl_path): continue

    
    scl_temp = os.path.join(patch_path, 'scl_clip_temp.tif')
    processing.run("gdal:cliprasterbymasklayer", {
        'INPUT': scl_path, 'MASK': field_layer, 'OUTPUT': scl_temp, 'NODATA': -1
    })
    
    ds_scl = gdal.Open(scl_temp)
    scl_arr = ds_scl.GetRasterBand(1).ReadAsArray()
    ds_scl = None
    
    #4 or 5?
    valid_mask = (scl_arr == 4) | (scl_arr == 5)
    clear_ratio = np.sum(valid_mask) / np.sum(scl_arr != -1)
    os.remove(scl_temp) # cleanup immediately

    if clear_ratio < 0.5: # if less than 50% is clear, skip
        print(f" Skipping {date_folder}: {((1-clear_ratio)*100):.0f}% obscured")
        continue

    print(f"Analyzing {date_folder} ({clear_ratio*100:.0f}% Clear)...")

    #ndvi
    ndvi_temp = os.path.join(patch_path, 'v_temp_ndvi.tif')
    clip_temp = os.path.join(patch_path, 'v_temp_clip.tif')

    try:
        processing.run("gdal:rastercalculator", {
            'INPUT_A': bands_path, 'BAND_A': 4,
            'INPUT_B': bands_path, 'BAND_B': 3,
            'FORMULA': '(A.astype(float) - B) / (A + B + 1e-6)',
            'OUTPUT': ndvi_temp
        })

        processing.run("gdal:cliprasterbymasklayer", {
            'INPUT': ndvi_temp, 'MASK': field_layer, 'NODATA': -9999, 'OUTPUT': clip_temp
        })

        ds = gdal.Open(clip_temp)
        arr = ds.GetRasterBand(1).ReadAsArray().astype(float)
        ds = None 

        valid_pixels = arr[(arr != -9999) & (arr > 0) & (arr < 1)]

        if valid_pixels.size > 0:
            mean_ndvi = np.mean(valid_pixels)
            chronological_dates.append(date_folder)
            ndvi_health_scores.append(mean_ndvi)

    finally:
        for f in [ndvi_temp, clip_temp]:
            if os.path.exists(f): os.remove(f)

# plot
plt.figure(figsize=(12, 6))
plt.plot(chronological_dates, ndvi_health_scores, color='#2ecc71', marker='o', linewidth=2)
plt.title(f"Field Growth Curve| Patch {TARGET_PATCH}", fontsize=14, fontweight='bold')
plt.xlabel("Observation Date")
plt.ylabel("NDVI")
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 0.8) 
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
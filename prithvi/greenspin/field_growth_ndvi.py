from qgis.core import QgsProject
import processing
import os
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt

#config
BASE_PATH        = r'Y:\14_Zuckerrübe\Sentinel_2\wue'
FIELD_LAYER_NAME = 'intra_field — sb_fields_wue_2024'
TARGET_PATCH     = '32UPA_0_4'

# SCLcodes 4 and 5 : vegetation

MIN_CLEAR_RATIO  = 0.5

#load field boundary layer
field_layer = QgsProject.instance().mapLayersByName(FIELD_LAYER_NAME)[0]

dates       = []   # e.g. "2024-07-14"
ndvi_means  = []   # average greenness across the field on that date
ndvi_stdevs = []   # spatial spread — high = patchy, low = uniform

print(f"Starting temporal analysis for patch {TARGET_PATCH}...")

#main loop
for date_folder in sorted(os.listdir(BASE_PATH)):

    date_path  = os.path.join(BASE_PATH, date_folder)
    bands_path = os.path.join(date_path, TARGET_PATCH, 'bands.tif')
    scl_path   = os.path.join(date_path, TARGET_PATCH, 'scl.tif')

    
    if not os.path.isdir(date_path):                          continue
    if not os.path.exists(bands_path):                        continue
    if not os.path.exists(scl_path):                          continue

    #cloud check
    scl_clipped = os.path.join(date_path, TARGET_PATCH, '_scl_temp.tif')
    processing.run("gdal:cliprasterbymasklayer", {
        'INPUT':  scl_path,
        'MASK':   field_layer,
        'NODATA': -1,
        'OUTPUT': scl_clipped,
    })

    ds_scl  = gdal.Open(scl_clipped)
    scl_arr = ds_scl.GetRasterBand(1).ReadAsArray()
    ds_scl  = None
    os.remove(scl_clipped)

    #4 = vegetation, 5 = bare soil
    clear_pixels = np.sum((scl_arr == 4) | (scl_arr == 5))
    valid_pixels = np.sum(scl_arr != -1)
    clear_ratio  = clear_pixels / valid_pixels if valid_pixels > 0 else 0

    if clear_ratio < MIN_CLEAR_RATIO:
        obscured_pct = (1 - clear_ratio) * 100
        print(f"  Skipping {date_folder} — {obscured_pct:.0f}% Hidden by cloud/shadow")
        continue

    print(f"  Analysing {date_folder}  ({clear_ratio * 100:.0f}% clear)...")

    #ndvi compute
    ndvi_raw    = os.path.join(date_path, TARGET_PATCH, '_ndvi_temp.tif')
    ndvi_clipped = os.path.join(date_path, TARGET_PATCH, '_ndvi_clip_temp.tif')

    try:
        #band 4 = NIR,  band 3 = Red
        processing.run("gdal:rastercalculator", {
            'INPUT_A': bands_path, 'BAND_A': 4,
            'INPUT_B': bands_path, 'BAND_B': 3,
            'FORMULA': '(A.astype(float) - B) / (A + B + 1e-6)',
            'OUTPUT':  ndvi_raw,
        })
        processing.run("gdal:cliprasterbymasklayer", {
            'INPUT':  ndvi_raw,
            'MASK':   field_layer,
            'NODATA': -9999,
            'OUTPUT': ndvi_clipped,
        })

        ds  = gdal.Open(ndvi_clipped)
        arr = ds.GetRasterBand(1).ReadAsArray().astype(float)
        ds  = None

        #discard nodata, cloud
        good = arr[(arr != -9999) & (arr > 0) & (arr < 1)]

        if good.size > 0:
            dates.append(date_folder)
            ndvi_means.append(np.mean(good))
            ndvi_stdevs.append(np.std(good))

    finally:
        #temp file cleanup
        for tmp in [ndvi_raw, ndvi_clipped]:
            if os.path.exists(tmp):
                os.remove(tmp)

#plot:mean NDVI and StDev
if not dates:
    print("No usable dates found — nothing to plot.")
else:
    fig, ax_ndvi = plt.subplots(figsize=(13, 6))

    #left axis : mean NDVI 
    ax_ndvi.plot(dates, ndvi_means,
                 color='#2ecc71', marker='o', linewidth=2, label='Mean NDVI (crop health)')
    ax_ndvi.set_ylabel('Mean NDVI', color='#2ecc71', fontweight='bold')
    ax_ndvi.set_ylim(0, 1.0)
    ax_ndvi.tick_params(axis='x', rotation=45)
    ax_ndvi.set_xlabel('Observation date')
    ax_ndvi.grid(axis='y', linestyle='--', alpha=0.4)

    #right axis : StDev bars
    ax_var = ax_ndvi.twinx()
    ax_var.bar(dates, ndvi_stdevs,
               color='#e74c3c', alpha=0.25, label='NDVI StDev (intra-field variation)')
    ax_var.set_ylabel('NDVI StDev  (intra-field variation)', color='#e74c3c', fontweight='bold')
    ax_var.set_ylim(0, 0.3)

    #combine both legends into one box
    lines,  labels  = ax_ndvi.get_legend_handles_labels()
    bars,   blabels = ax_var.get_legend_handles_labels()
    ax_ndvi.legend(lines + bars, labels + blabels, loc='upper left')

    plt.title(f'Field growth curve vs. spatial variation  |  Patch {TARGET_PATCH}',
              fontsize=13, fontweight='bold')
    fig.tight_layout()
    plt.show()

    print(f"\nDone. {len(dates)} plotted.")
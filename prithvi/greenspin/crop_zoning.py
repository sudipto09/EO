from qgis.core import QgsProject, QgsRasterLayer
import processing
import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

#config
BASE_PATH = r'Y:\14_Zuckerrübe\Sentinel_2\wue'
DATE      = '2024-07-14'
PATCH_ID  = '32UPA_0_4'

def p(name): return os.path.join(BASE_PATH, DATE, PATCH_ID, name)

bands_path       = p('bands.tif')
scl_path         = p('scl.tif')
ndvi_path        = p('ndvi_qgis.tif')
ndvi_masked_path = p('ndvi_masked.tif')
zones_path       = p('zones_qgis.tif')

#ndvi
processing.run("gdal:rastercalculator", {
    'INPUT_A': bands_path, 'BAND_A': 4,
    'INPUT_B': bands_path, 'BAND_B': 3,
    'FORMULA': '(A.astype(float) - B) / (A + B + 1e-6)',
    'OUTPUT':  ndvi_path
})

#cloud mask 
processing.run("gdal:rastercalculator", {
    'INPUT_A': scl_path,  'BAND_A': 1,
    'INPUT_B': ndvi_path, 'BAND_B': 1,
    'FORMULA': 'where((A == 4) | (A == 5), B, nan)',
    'OUTPUT':  ndvi_masked_path
})

#zone classification
# 2 = High   (NDVI >= 0.6)
# 1 = Medium (NDVI 0.3–0.6)
# 0 = Low    (NDVI < 0.3)
processing.run("gdal:rastercalculator", {
    'INPUT_A': ndvi_masked_path, 'BAND_A': 1,
    'FORMULA': 'where(isnan(A), -1, where(A >= 0.6, 2, where(A >= 0.3, 1, 0)))',
    'OUTPUT':  zones_path
})

#load layers
for path, name in [
    (ndvi_path,        "NDVI"),
    (ndvi_masked_path, "NDVI Masked"),
    (zones_path,       "Zones"),
]:
    lyr = QgsRasterLayer(path, name)
    assert lyr.isValid(), f"Failed to load: {path}"
    QgsProject.instance().addMapLayer(lyr)

#scores
z_layer = QgsRasterLayer(zones_path, "Zones_Data")
assert z_layer.isValid(), f"Failed to load zones: {zones_path}"

block = z_layer.dataProvider().block(1, z_layer.extent(), z_layer.width(), z_layer.height())

data = np.array([
    block.value(r, c)
    for r in range(z_layer.height())
    for c in range(z_layer.width())
])
data = data[data != -1]  #drop masked pixels

total = len(data)
if total == 0:
    print("No valid pixels found.")
else:
    for zone, label in [(2, "High Vegetation"), (1, "Medium Growth"), (0, "Bare Soil/Low")]:
        print(f"{label:20s} (Zone {zone}): {(np.sum(data == zone) / total * 100):.1f}%")
    print(f"Total valid pixels : {total}")

    #pie chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    values = [np.sum(data == z) for z in [2, 1, 0]]
    axes[0].pie(values, labels=["High", "Medium", "Low"],
                autopct='%1.1f%%', colors=['#2d6a4f', '#95d5b2', '#d4a373'])
    axes[0].set_title(f"Vegetation Zone Distribution\n{DATE} | {PATCH_ID}")

    #ndvimap
    ds   = gdal.Open(ndvi_masked_path)
    ndvi = ds.GetRasterBand(1).ReadAsArray().astype(float)
    ndvi[ndvi == ds.GetRasterBand(1).GetNoDataValue()] = np.nan
    ds   = None  # close file

    im = axes[1].imshow(ndvi, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[1], label="NDVI")
    axes[1].set_title(f"NDVI Map (Masked)\n{DATE} | {PATCH_ID}")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    #ndvi histogram
    plt.figure(figsize=(7, 4))
    valid_ndvi = ndvi[~np.isnan(ndvi)].flatten()
    plt.hist(valid_ndvi, bins=50, color='#52b788', edgecolor='white')
    plt.axvline(0.3, color='orange', linestyle='--', label='Low/Med threshold')
    plt.axvline(0.6, color='red',    linestyle='--', label='Med/High threshold')
    plt.title(f"NDVI Distribution | {DATE}")
    plt.xlabel("NDVI Value")
    plt.ylabel("Pixel Count")
    plt.legend()
    plt.tight_layout()
    plt.show()



#field-specific 
FIELD_LAYER_NAME = 'single_field — sb_fields_wue_2024'
matches = QgsProject.instance().mapLayersByName(FIELD_LAYER_NAME)
assert len(matches) > 0, f"Layer not found: '{FIELD_LAYER_NAME}'\nLoaded: {[l.name() for l in QgsProject.instance().mapLayers().values()]}"
field_layer = matches[0]

#clipped path
ndvi_field_path        = p('ndvi_field.tif')
ndvi_masked_field_path = p('ndvi_masked_field.tif')
zones_field_path       = p('zones_field.tif')

for src, dst in [
    (ndvi_path,        ndvi_field_path),
    (ndvi_masked_path, ndvi_masked_field_path),
    (zones_path,       zones_field_path),
]:
    processing.run("gdal:cliprasterbymasklayer", {
        'INPUT':           src,
        'MASK':            field_layer,
        'CROP_TO_CUTLINE': True,
        'KEEP_RESOLUTION': True,
        'NODATA':          -9999,
        'OUTPUT':          dst
    })

# load 
for path, name in [
    (ndvi_field_path,        "NDVI (Field)"),
    (ndvi_masked_field_path, "NDVI Masked (Field)"),
    (zones_field_path,       "Zones (Field)"),
]:
    lyr = QgsRasterLayer(path, name)
    assert lyr.isValid(), f"Failed to load: {path}"
    QgsProject.instance().addMapLayer(lyr)

#field score
z_field = QgsRasterLayer(zones_field_path, "Zones_Field")
assert z_field.isValid()

block_f = z_field.dataProvider().block(1, z_field.extent(), z_field.width(), z_field.height())
data_f  = np.array([
    block_f.value(r, c)
    for r in range(z_field.height())
    for c in range(z_field.width())
])
data_f = data_f[(data_f != -1) & (data_f != -9999)]

total_f = len(data_f)
if total_f == 0:
    print("No valid field pixels found.")
else:
    print(f"\nField: {DATE} | {PATCH_ID}")
    for zone, label in [(2, "High Vegetation"), (1, "Medium Growth"), (0, "Bare Soil/Low")]:
        print(f"{label:20s} (Zone {zone}): {(np.sum(data_f == zone) / total_f * 100):.1f}%")
    print(f"Total valid pixels : {total_f}")

    #pie chart and ndvi map
    ds_f   = gdal.Open(ndvi_masked_field_path)
    ndvi_f = ds_f.GetRasterBand(1).ReadAsArray().astype(float)
    nodata = ds_f.GetRasterBand(1).GetNoDataValue()
    if nodata is not None:
        ndvi_f[ndvi_f == nodata] = np.nan
    ds_f   = None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Field Analysis | {DATE} | {PATCH_ID}", fontsize=13, fontweight='bold')

    values_f = [np.sum(data_f == z) for z in [2, 1, 0]]
    axes[0].pie(values_f, labels=["High", "Medium", "Low"],
                autopct='%1.1f%%', colors=['#2d6a4f', '#95d5b2', '#d4a373'])
    axes[0].set_title("Vegetation Zone Distribution (Field)")

    im_f = axes[1].imshow(ndvi_f, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(im_f, ax=axes[1], label="NDVI")
    axes[1].set_title("NDVI Map (Field, Masked)")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    #ndvi histogram
    valid_f = ndvi_f[~np.isnan(ndvi_f)].flatten()

    plt.figure(figsize=(7, 4))
    plt.hist(valid_f, bins=50, color='#52b788', edgecolor='white')
    plt.axvline(0.3, color='orange', linestyle='--', label='Low/Med threshold')
    plt.axvline(0.6, color='red',    linestyle='--', label='Med/High threshold')
    plt.title(f"NDVI Distribution (Field) | {DATE}")
    plt.xlabel("NDVI Value")
    plt.ylabel("Pixel Count")
    plt.legend()
    plt.tight_layout()
    plt.show()


FIELD_LAYER_NAME = 'single_field — sb_fields_wue_2024'
matches = QgsProject.instance().mapLayersByName(FIELD_LAYER_NAME)
assert len(matches) > 0, f"Layer not found: '{FIELD_LAYER_NAME}'\nLoaded: {[l.name() for l in QgsProject.instance().mapLayers().values()]}"
field_layer = matches[0]


from qgis.core import QgsProject, QgsRasterLayer
import processing
import os

#config
BASE_PATH = r'Y:\14_Zuckerrübe\Sentinel_2\wue'
DATE      = '2024-07-14'
PATCH_ID  = '32UPA_0_4'

def p(name): return os.path.join(BASE_PATH, DATE, PATCH_ID, name)

ndvi_path        = p('ndvi_qgis.tif')
ndvi_masked_path = p('ndvi_masked.tif')
zones_path       = p('zones_qgis.tif')

#my field
FIELD_LAYER_NAME = 'single_field — sb_fields_wue_2024'
matches = QgsProject.instance().mapLayersByName(FIELD_LAYER_NAME)
assert len(matches) > 0, f"Layer not found: '{FIELD_LAYER_NAME}'\nLoaded layers: {[l.name() for l in QgsProject.instance().mapLayers().values()]}"
field_layer = matches[0]

#clip fn
def clip_raster(input_raster, output_path):
    processing.run("gdal:cliprasterbymasklayer", {
        'INPUT':           input_raster,
        'MASK':            field_layer,
        'CROP_TO_CUTLINE': True,
        'KEEP_RESOLUTION': True,
        'NODATA':          -9999,
        'OUTPUT':          output_path
    })

# clipping
ndvi_clip_path        = p('ndvi_clipped.tif')
ndvi_masked_clip_path = p('ndvi_masked_clipped.tif')
zones_clip_path       = p('zones_clipped.tif')

clip_raster(ndvi_path,        ndvi_clip_path)
clip_raster(ndvi_masked_path, ndvi_masked_clip_path)
clip_raster(zones_path,       zones_clip_path)

# result loading
for path, name in [
    (ndvi_clip_path,        "NDVI (Field)"),
    (ndvi_masked_clip_path, "NDVI Masked (Field)"),
    (zones_clip_path,       "Zones (Field)"),
]:
    lyr = QgsRasterLayer(path, name)
    assert lyr.isValid(), f"Failed to load: {path}"
    QgsProject.instance().addMapLayer(lyr)

print("Field-specific rasters created")
from qgis.core import (
    QgsProject, QgsRasterLayer,
    QgsCoordinateReferenceSystem, QgsCoordinateTransform
)
import os, math


LAYER_NAME = 'single_field — sb_fields_wue_2024'
ROOT_PATH  = r'Y:\14_Zuckerrübe\Sentinel_2\wue'
TARGET_CRS = QgsCoordinateReferenceSystem("EPSG:32632")

SCL_LABELS = {
    3: "SHADOW  : BLIND",
    4: "VEGETATION : CLEAR",
    5: "BARE SOIL  : CLEAR",
    8: "CLOUD MED  : BLIND",
    9: "CLOUD HIGH : BLIND",
   10: "THIN CIRRUS: BLIND",
}

#get field
v_layer  = QgsProject.instance().mapLayersByName(LAYER_NAME)[0]
xform    = QgsCoordinateTransform(v_layer.crs(), TARGET_CRS, QgsProject.instance())
center   = xform.transform(v_layer.extent().center())  

print(f"{'Date':<15} | {'Patch':<15} | {'SCL':<4} | Status")
print("-" * 60)

#loop
for date_folder in sorted(os.listdir(ROOT_PATH)):
    date_dir = os.path.join(ROOT_PATH, date_folder)
    if not os.path.isdir(date_dir):
        continue

    for patch_id in sorted(os.listdir(date_dir)):
        scl_path = os.path.join(date_dir, patch_id, 'scl.tif')
        if not os.path.exists(scl_path):
            continue

        r_layer = QgsRasterLayer(scl_path, "temp_scl")
        if not r_layer.isValid():
            continue

        
        raster_xform = QgsCoordinateTransform(TARGET_CRS, r_layer.crs(), QgsProject.instance())
        sample_point = raster_xform.transform(center)

        #sample band 1 at field center
        val, ok = r_layer.dataProvider().sample(sample_point, 1)

        if not ok or math.isnan(val):
            continue  #center falls outside this raster tile

        scl_val = int(val)
        status  = SCL_LABELS.get(scl_val, f"OTHER SCL={scl_val}")
        print(f"{date_folder:<15} | {patch_id:<15} | {scl_val:<4} | {status}")
        break  
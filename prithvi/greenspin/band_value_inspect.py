from qgis.core import QgsProject, QgsRasterLayer
import os


target_date = '2024-07-14'
patch_id = '32UPA_0_4'
bands_path = os.path.join(r'Y:\14_Zuckerrübe\Sentinel_2\wue', target_date, patch_id, 'bands.tif')

r_layer = QgsRasterLayer(bands_path, "target_bands")

if r_layer.isValid():
    
    #samples first four (B2, B3, B4, B8)
    for i in range(1, 5):
        val, res = r_layer.dataProvider().sample(sample_point, i)
        print(f"Band {i} Value: {val}")
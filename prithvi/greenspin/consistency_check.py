from qgis.core import QgsProject, QgsCoordinateTransform, QgsCoordinateReferenceSystem
import numpy as np
from osgeo import gdal

#layer selection
LAYER_NAME = 'sb_fields_wue_2024'
layer = QgsProject.instance().mapLayersByName(LAYER_NAME)[0]
selected = layer.selectedFeatures()

if not selected:
    print(f"Nothing selected on '{LAYER_NAME}'. ")
          
else:
    #load the patch and pull nir and red bands 
    RASTER_PATH = r'Y:\14_Zuckerrübe\Sentinel_2\wue\2024-07-14\32UPA_0_4\bands.tif'
    ds  = gdal.Open(RASTER_PATH)
    gt  = ds.GetGeoTransform()   # maps pixel coords to real world coords

    nir = ds.GetRasterBand(4).ReadAsArray().astype(float)   # nir
    red = ds.GetRasterBand(3).ReadAsArray().astype(float)

    #ndvi
    ndvi = (nir - red) / (nir + red + 1e-6)   # tiny epsilon avoids divide-by-zero

    #reproject the selected field into raster's CRS 
    raster_crs  = QgsCoordinateReferenceSystem("EPSG:32632")   # UTM zone 32N
    crs_convert = QgsCoordinateTransform(layer.crs(), raster_crs, QgsProject.instance())

    field   = selected[0]
    geom    = field.geometry()
    geom.transform(crs_convert)

    # translate the field's bounding box into pixel row/col indices 
    bbox = geom.boundingBox()

    # gt[0], gt[3] = top-left corner of the raster in real-world coords
    # gt[1] = pixel width  (+)
    # gt[5]= pixel height (-, because rows run top to bottom)
    col_left  = int((bbox.xMinimum() - gt[0]) / gt[1])
    col_right = int((bbox.xMaximum() - gt[0]) / gt[1])
    row_top   = int((bbox.yMaximum() - gt[3]) / gt[5])
    row_bot   = int((bbox.yMinimum() - gt[3]) / gt[5])

    #slice out the pixels that fall inside the field's bounding box
    window = ndvi[row_top:row_bot, col_left:col_right]

    #drops clouds, shadows, and bare soil edges
    vegetation_pixels = window[(window > 0.1) & (window < 1.0)]

    #spatial variability
    if vegetation_pixels.size == 0:
        print("No valid pixels found. "
              "Check that the selected field overlaps the 32UPA_0_4 patch.")
    else:
        stdev = np.std(vegetation_pixels)
        print(f"Field ID {field.id()}  |  NDVI StDev = {stdev:.3f}")

        if stdev > 0.12:
            
            print("PATCHY: Strong intra-field variation detected. "
                  "Good candidate for a Prithvi analysis.")
        else:
            
            print("UNIFORM: Field is consistent.")
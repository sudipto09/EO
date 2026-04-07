from qgis.core import QgsProject
from qgis.PyQt.QtCore import QTimer
from osgeo import gdal, ogr, osr
import numpy as np
import os, csv

#Config
BASE_PATH        = r'Y:\14_Zuckerrübe\Sentinel_2\wue'
FIELD_LAYER_NAME = 'sb_fields_wue_2024'
OUTPUT_DIR       = r'C:\Users\Sudipto\internship\EO\prithvi\greenspin\multi_crop_output'
CLEAR_THRESHOLD  = 0.5
STDEV_THRESHOLD  = 0.12
MIN_VALID_DATES  = 3
os.makedirs(OUTPUT_DIR, exist_ok=True)
gdal.UseExceptions()

date_folders = sorted([d for d in os.listdir(BASE_PATH)
                       if os.path.isdir(os.path.join(BASE_PATH, d))])
all_patches = sorted({
    item
    for df in date_folders
    for item in os.listdir(os.path.join(BASE_PATH, df))
    if os.path.isdir(os.path.join(BASE_PATH, df, item))
    and os.path.exists(os.path.join(BASE_PATH, df, item, 'bands.tif'))
})

#build field masks for a patch in 1 rasterization 
def make_all_masks(field_layer, patch_id):
    ref_path = next((os.path.join(BASE_PATH, df, patch_id, 'scl.tif')
                     for df in date_folders
                     if os.path.exists(os.path.join(BASE_PATH, df, patch_id, 'scl.tif'))),
                    None)
    if not ref_path:
        return {}, None

    ref_ds = gdal.Open(ref_path)
    gt, proj = ref_ds.GetGeoTransform(), ref_ds.GetProjection()
    W, H     = ref_ds.RasterXSize, ref_ds.RasterYSize
    ref_ds   = None

    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(proj)
    raster_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    field_srs = osr.SpatialReference()
    field_srs.ImportFromEPSG(4326)
    field_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(field_srs, raster_srs)

    x_min = gt[0];           x_max = gt[0] + W * gt[1]
    y_min = gt[3] + H * gt[5]; y_max = gt[3]

    #single OGR layer with all fields, burn value = sequential int ID 
    tmp_ds  = ogr.GetDriverByName('Memory').CreateDataSource('tmp')
    tmp_lyr = tmp_ds.CreateLayer('f', srs=raster_srs)
    tmp_lyr.CreateField(ogr.FieldDefn('burn_id', ogr.OFTInteger))

    field_meta = {} 
    burn_id = 1

    for feat in field_layer.getFeatures():
        ogr_geom = ogr.CreateGeometryFromWkt(feat.geometry().asWkt())
        ogr_geom.Transform(transform)
        env = ogr_geom.GetEnvelope()  # xmin, xmax, ymin, ymax
        if env[1] < x_min or env[0] > x_max or env[3] < y_min or env[2] > y_max:
            continue

        ogr_feat = ogr.Feature(tmp_lyr.GetLayerDefn())
        ogr_feat.SetGeometry(ogr_geom)
        ogr_feat.SetField('burn_id', burn_id)
        tmp_lyr.CreateFeature(ogr_feat)

        field_meta[burn_id] = {
            'fid':       feat.id(),
            'fieldusno': feat['FIELDUSNO'],
            'acrename':  feat['ACRENAME'],
            'ha':        feat['ha_calc'],
        }
        burn_id += 1

    if not field_meta:
        return {}, (gt, proj, W, H)

    # 1 rasterization for all fields in this patch
    mem = gdal.GetDriverByName('MEM').Create('', W, H, 1, gdal.GDT_Int32)
    mem.SetGeoTransform(gt)
    mem.SetProjection(proj)
    mem.GetRasterBand(1).Fill(0)
    gdal.RasterizeLayer(mem, [1], tmp_lyr,
                        options=['ATTRIBUTE=burn_id'])
    id_raster = mem.GetRasterBand(1).ReadAsArray()  # shape (H, W), value = burn_id
    mem = None

    #slice per field 
    masks = {}
    for bid, meta in field_meta.items():
        m = (id_raster == bid)
        if np.sum(m) >= 5:
            meta['mask']   = m
            meta['pixels'] = int(np.sum(m))
            masks[bid]     = meta

    print(f"  → {len(masks)} fields in patch (1 rasterization)")
    return masks, (gt, proj, W, H)


#Scanner
class DoubleCropScanner:
    def __init__(self):
        self.field_layer = QgsProject.instance().mapLayersByName(FIELD_LAYER_NAME)[0]
        self.patch_queue = list(all_patches)
        self.patch_idx   = 0
        self.all_results = []
        self.timer       = QTimer()
        self.timer.setInterval(0)
        self.timer.timeout.connect(self._tick)
        print(f"Patches: {len(all_patches)}  Dates: {len(date_folders)}")
        print("Startings...\n")
        self.timer.start()

    def _tick(self):
        if self.patch_idx >= len(self.patch_queue):
            self.timer.stop()
            self._finish()
            return

        patch_id = self.patch_queue[self.patch_idx]
        self.patch_idx += 1
        pct = self.patch_idx / len(self.patch_queue) * 100
        print(f"[{self.patch_idx}/{len(self.patch_queue)} {pct:.0f}%] {patch_id}")

        masks, meta = make_all_masks(self.field_layer, patch_id)
        if not masks:
            print(f"  → no overlapping fields")
            return

        gt, proj, W, H = meta

        for date_folder in date_folders:
            scl_path   = os.path.join(BASE_PATH, date_folder, patch_id, 'scl.tif')
            bands_path = os.path.join(BASE_PATH, date_folder, patch_id, 'bands.tif')
            if not os.path.exists(scl_path) or not os.path.exists(bands_path):
                continue

            try:
                scl_ds  = gdal.Open(scl_path)
                scl_arr = scl_ds.GetRasterBand(1).ReadAsArray()
                scl_ds  = None
                if np.all(scl_arr == 0):
                    continue

                ds  = gdal.Open(bands_path)
                nir = ds.GetRasterBand(4).ReadAsArray().astype(np.float32)
                red = ds.GetRasterBand(3).ReadAsArray().astype(np.float32)
                ds  = None
                ndvi = (nir - red) / (nir + red + 1e-6)

                for bid, fdata in masks.items():
                    mask = fdata['mask']
                    clear_ratio = (np.sum(mask & ((scl_arr==4)|(scl_arr==5)))
                                   / np.sum(mask))
                    if clear_ratio < CLEAR_THRESHOLD:
                        continue

                    valid = ndvi[mask & (ndvi > -1) & (ndvi < 1)]
                    if valid.size < 5:
                        continue

                    mean_ndvi  = float(np.mean(valid))
                    stdev_ndvi = float(np.std(valid))
                    is_double  = stdev_ndvi > STDEV_THRESHOLD and mean_ndvi > 0.15

                    self.all_results.append({
                        'patch_id':   patch_id,
                        'fid':        fdata['fid'],
                        'fieldusno':  fdata['fieldusno'],
                        'acrename':   fdata['acrename'],
                        'ha':         fdata['ha'],
                        'pixels':     fdata['pixels'],
                        'date':       date_folder,
                        'mean_ndvi':  round(mean_ndvi, 4),
                        'stdev_ndvi': round(stdev_ndvi, 4),
                        'is_double':  is_double,
                    })

                    if is_double:
                        print(f"  {fdata['acrename']}  "
                              f"date={date_folder}  "
                              f"mean={mean_ndvi:.3f}  stdev={stdev_ndvi:.3f}")

            except Exception as e:
                print(f"  error {date_folder}: {e}")
                continue

    def _finish(self):
        flagged  = [r for r in self.all_results if r['is_double']]
        full_csv = os.path.join(OUTPUT_DIR, 'all_fields_ndvi.csv')
        flag_csv = os.path.join(OUTPUT_DIR, 'double_crop_candidates.csv')

        if self.all_results:
            with open(full_csv, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=self.all_results[0].keys())
                w.writeheader(); w.writerows(self.all_results)

        if flagged:
            with open(flag_csv, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=flagged[0].keys())
                w.writeheader(); w.writerows(flagged)

        print(f"\n Done")
        print(f"   Observations : {len(self.all_results)}")
        print(f"   Double-crop  : {len(flagged)}")
        print(f"   Full CSV  → {full_csv}")
        print(f"   Flags CSV → {flag_csv}")

scanner = DoubleCropScanner()

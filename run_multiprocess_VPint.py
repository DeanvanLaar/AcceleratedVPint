# Imports
import time

import numpy as np
import multiprocessing

import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from pyproj import Transformer

from VPint.WP_MRP import WP_SMRP

import csv
import os
import sys

# Data
scenes = []
# Scenes
target_europe_urban_madrid = "S2A_MSIL2A_20200801T105631_N0214_R094_T30TVK_20200801T121533"
m1_europe_urban_madrid = "S2B_MSIL2A_20200710T110619_N0214_R137_T30TVK_20200710T140319"
mask_europe_urban_madrid = "S2A_MSIL2A_20200930T105851_N0214_R094_T30TVK_20200930T135525"
scenes.append(["europe_urban_madrid", target_europe_urban_madrid, m1_europe_urban_madrid, mask_europe_urban_madrid])

# target = 1w    and   feature = 6m
target_europe_cropland_hungary = "S2A_MSIL2A_20211028T093111_N0301_R136_T34TES_20211028T122324"
m1_europe_cropland_hungary = "S2A_MSIL2A_20210511T093031_N0300_R136_T34TES_20210511T120358"
mask_europe_cropland_hungary = "S2A_MSIL2A_20220429T094041_N0400_R036_T34TES_20220429T144214"
scenes.append(["europe_cropland_hungary", target_europe_cropland_hungary, m1_europe_cropland_hungary,
               mask_europe_cropland_hungary])

target_europe_cropland_ukraine = "S2A_MSIL2A_20220220T084001_N0400_R064_T36UYU_20220220T110621"
m1_europe_cropland_ukraine = "S2A_MSIL2A_20220108T083331_N0301_R021_T37TCN_20220108T104906"
mask_europe_cropland_ukraine = "S2A_MSIL2A_20220421T083611_N0400_R064_T37UCP_20220421T124655"
scenes.append(["europe_cropland_ukraine", target_europe_cropland_ukraine, m1_europe_cropland_ukraine,
               mask_europe_cropland_ukraine])

target_africa_cropland_nile = "S2A_MSIL2A_20200905T082611_N0214_R021_T36RUV_20200905T111042"
m1_africa_cropland_nile = "S2A_MSIL2A_20200806T082611_N0214_R021_T36RUV_20200806T112556"
mask_africa_cropland_nile = "S2A_MSIL2A_20201104T083131_N0214_R021_T36RUV_20201104T110428"
scenes.append(["africa_cropland_nile", target_africa_cropland_nile, m1_africa_cropland_nile, mask_africa_cropland_nile])

target_america_shrubs_mexico = "S2B_MSIL2A_20210331T172859_N0300_R055_T13REK_20210406T192806"
m1_america_shrubs_mexico = "S2A_MSIL2A_20210214T173421_N0214_R055_T13REK_20210214T215327"
mask_america_shrubs_mexico = "S2B_MSIL2A_20210927T173009_N0301_R055_T13REK_20210927T220131"
scenes.append(
    ["america_shrubs_mexico", target_america_shrubs_mexico, m1_america_shrubs_mexico, mask_america_shrubs_mexico])

target_asia_herbaceous_mongoliaeast = "S2B_MSIL2A_20210922T031539_N0301_R118_T49TFK_20210922T063832"
m1_asia_herbaceous_mongoliaeast = "S2A_MSIL2A_20210831T032541_N0301_R018_T49TFK_20210831T071138"
mask_asia_herbaceous_mongoliaeast = "S2B_MSIL2A_20220321T031539_N0400_R118_T49TFK_20220321T055132"
scenes.append(["asia_herbaceous_mongoliaeast", target_asia_herbaceous_mongoliaeast, m1_asia_herbaceous_mongoliaeast,
               mask_asia_herbaceous_mongoliaeast])

target_asia_urban_beijing = "S2A_MSIL2A_20211024T030811_N0301_R075_T50TMK_20211024T062031"
m1_asia_urban_beijing = "S2A_MSIL2A_20211001T025551_N0301_R032_T50TMK_20211001T060617"
mask_asia_urban_beijing = "S2A_MSIL2A_20220422T030551_N0400_R075_T50TMK_20220422T053906"
scenes.append(["asia_urban_beijing", target_asia_urban_beijing, m1_asia_urban_beijing, mask_asia_urban_beijing])

# target = 1w
target_africa_herbaceous_southafrica = "S2B_MSIL2A_20220501T074609_N0400_R135_T35JNF_20220501T110359"
m1_africa_herbaceous_southafrica = "S2A_MSIL2A_20220406T074611_N0400_R135_T35JNF_20220406T102554"
mask_africa_herbaceous_southafrica = "S2A_MSIL2A_20221102T075101_N0400_R135_T35HNE_20221102T104758"
scenes.append(["africa_herbaceous_southafrica", target_africa_herbaceous_southafrica, m1_africa_herbaceous_southafrica,
               mask_africa_herbaceous_southafrica])

target_australia_shrubs_south = "S2A_MSIL2A_20220507T005711_N0400_R002_T53JMG_20220507T044010"
m1_australia_shrubs_south = "S2B_MSIL2A_20220412T005709_N0400_R002_T53JMG_20220412T024411"
mask_australia_shrubs_south = "S2A_MSIL2A_20221103T005711_N0400_R002_T53JMG_20221103T052901"
scenes.append(
    ["australia_shrubs_south", target_australia_shrubs_south, m1_australia_shrubs_south, mask_australia_shrubs_south])

# feature = 6m
target_asia_cropland_india = "S2A_MSIL2A_20220514T052651_N0400_R105_T43REM_20220514T100220"
m1_asia_cropland_india = "S2A_MSIL2A_20211215T053231_N0301_R105_T43REM_20211215T072135"
mask_asia_cropland_india = "S2A_MSIL2A_20221110T053041_N0400_R105_T43REM_20221110T082053"
scenes.append(["asia_cropland_india", target_asia_cropland_india, m1_asia_cropland_india, mask_asia_cropland_india])

target_america_cropland_iowa = "S2A_MSIL2A_20220623T170901_N0400_R112_T15TVJ_20220623T231619"
m1_america_cropland_iowa = "S2A_MSIL2A_20220603T170901_N0400_R112_T15TVJ_20220603T234510"
mask_america_cropland_iowa = "S2A_MSIL2A_20220822T170901_N0400_R112_T15TVJ_20220823T005402"
scenes.append(
    ["america_cropland_iowa", target_america_cropland_iowa, m1_america_cropland_iowa, mask_america_cropland_iowa])

target_asia_herbaceous_kazakhstan = "S2B_MSIL2A_20220829T060629_N0400_R134_T42UYU_20220829T074151"
m1_asia_herbaceous_kazakhstan = "S2A_MSIL2A_20220725T060641_N0400_R134_T42UYU_20220725T095259"
mask_asia_herbaceous_kazakhstan = "S2B_MSIL2A_20230225T060829_N0509_R134_T42UYU_20230225T073455"
scenes.append(["asia_herbaceous_kazakhstan", target_asia_herbaceous_kazakhstan, m1_asia_herbaceous_kazakhstan,
               mask_asia_herbaceous_kazakhstan])

target_america_herbaceous_peru = "S2B_MSIL2A_20220825T150719_N0400_R082_T18LXJ_20220829T173637"
m1_america_herbaceous_peru = "S2A_MSIL2A_20220731T150731_N0400_R082_T18LXJ_20220731T214607"
mask_america_herbaceous_peru = "S2B_MSIL2A_20230221T150719_N0509_R082_T18LXJ_20230221T190953"
scenes.append(["america_herbaceous_peru", target_america_herbaceous_peru, m1_america_herbaceous_peru,
               mask_america_herbaceous_peru])

target_asia_shrubs_indiapakistan = "S2A_MSIL2A_20230115T055201_N0509_R048_T42RYS_20230115T091703"
m1_asia_shrubs_indiapakistan = "S2B_MSIL2A_20221221T055239_N0509_R048_T42RYS_20221221T072130"
mask_asia_shrubs_indiapakistan = "S2A_MSIL2A_20230316T054641_N0509_R048_T42RYS_20230316T092553"
scenes.append(["asia_shrubs_indiapakistan", target_asia_shrubs_indiapakistan, m1_asia_shrubs_indiapakistan,
               mask_asia_shrubs_indiapakistan])

# target = 1w
target_australia_shrubs_west = "S2A_MSIL2A_20200518T020451_N0214_R017_T50JNP_20200518T040042"
m1_australia_shrubs_west = "S2B_MSIL2A_20200503T020439_N0214_R017_T50JNP_20200503T070250"
mask_australia_shrubs_west = "S2B_MSIL2A_20201119T020449_N0214_R017_T50JPP_20201119T043552"
scenes.append(
    ["australia_shrubs_west", target_australia_shrubs_west, m1_australia_shrubs_west, mask_australia_shrubs_west])

# feature = 6m
target_america_urban_atlanta = "S2B_MSIL2A_20201210T162659_N0214_R040_T16SGC_20201210T190951"
m1_america_urban_atlanta = "S2B_MSIL2A_20200620T160829_N0214_R140_T16SGC_20200620T201455"
mask_america_urban_atlanta = "S2B_MSIL2A_20210208T162429_N0214_R040_T17SKT_20210210T165841"
scenes.append(
    ["america_urban_atlanta", target_america_urban_atlanta, m1_america_urban_atlanta, mask_america_urban_atlanta])

# target = 1w
target_asia_cropland_china = "S2B_MSIL2A_20211218T031129_N0301_R075_T50SKA_20211218T054114"
m1_asia_cropland_china = "S2B_MSIL2A_20211205T030109_N0301_R032_T50SKA_20211205T061129"
mask_asia_cropland_china = "S2B_MSIL2A_20220226T030659_N0400_R075_T50SKB_20220226T055944"
scenes.append(["asia_cropland_china", target_asia_cropland_china, m1_asia_cropland_china, mask_asia_cropland_china])

target_america_forest_mississippi = "S2B_MSIL2A_20220517T162839_N0400_R083_T16SEC_20220517T194438"
m1_america_forest_mississippi = "S2B_MSIL2A_20220427T162829_N0400_R083_T16SEC_20220427T205356"
mask_america_forest_mississippi = "S2B_MSIL2A_20221113T163529_N0400_R083_T16SEC_20221113T194832"
scenes.append(["america_forest_mississippi", target_america_forest_mississippi, m1_america_forest_mississippi,
               mask_america_forest_mississippi])

target_africa_forest_angola = "S2B_MSIL2A_20220906T084559_N0400_R107_T33LYE_20220906T114544"
m1_africa_forest_angola = "S2B_MSIL2A_20220817T084559_N0400_R107_T33LYE_20220817T120435"
mask_africa_forest_angola = "S2B_MSIL2A_20230305T084749_N0509_R107_T33LYE_20230305T123021"
scenes.append(["africa_forest_angola", target_africa_forest_angola, m1_africa_forest_angola, mask_africa_forest_angola])



# Main functions

def multi_VPint_interpolation(pred_dict, grids, band, use_IP=True, use_EB=True):
    MRP = WP_SMRP(grids[0], grids[1])
    pred_dict[band] = MRP.run(method='exact',
                                 auto_adapt=True, auto_adaptation_verbose=False,
                                 auto_adaptation_epochs=25, auto_adaptation_max_iter=100,
                                 auto_adaptation_strategy='random', auto_adaptation_proportion=0.8,
                                 resistance=use_EB, prioritise_identity=use_IP)


def load_product_windowed(path, y_size, x_size, y_offset, x_offset,
                          keep_bands=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"],
                          bands_10m={"B2": 1, "B3": 2, "B4": 3, "B8": 7},
                          bands_20m={"B5": 4, "B6": 5, "B7": 6, "B8A": 8, "B11": 11, "B12": 12, "CLD": 13},
                          bands_60m={"B1": 0, "B9": 9, "B10": 10},
                          return_bounds=False):
    grid = None
    size_y = -1
    size_x = -1

    scales = [bands_10m, bands_20m, bands_60m, {}]  # For bands that have multiple resolutions

    with rasterio.open(path) as raw_product:
        product = raw_product.subdatasets

    # Initialise grid
    with rasterio.open(product[1]) as bandset:
        size_y = bandset.profile['height']
        size_x = bandset.profile['width']
        # y_size, x_size is patch (given through arguments), size_y, size_x are scene dimensions from product
        grid = np.zeros((y_size, x_size, len(keep_bands))).astype(np.uint16)

    # Iterate over band sets (resolutions)
    resolution_index = 0
    for bs in product:
        with rasterio.open(bs, dtype="uint16") as bandset:
            desc = bandset.descriptions
            size_y_local = bandset.profile['height']
            size_x_local = bandset.profile['width']

            band_index = 1
            # Iterate over bands
            for d in desc:
                band_name = d.split(",")[0]

                if (band_name in keep_bands and band_name in scales[resolution_index]):

                    if (band_name in bands_10m):
                        b = bands_10m[band_name]
                        upscale_factor = (1 / 2)
                    elif (band_name in bands_20m):
                        b = bands_20m[band_name]
                        upscale_factor = 1
                    elif (band_name in bands_60m):
                        b = bands_60m[band_name]
                        upscale_factor = 3

                    # Output window using target resolution
                    window = Window(x_offset, y_offset, x_size, y_size)

                    # Second window for reading in local resolution
                    res_window = Window(x_offset * x_size / upscale_factor, y_offset * y_size / upscale_factor,
                                        window.width / upscale_factor, window.height / upscale_factor)

                    if (return_bounds and band_name in bands_20m):
                        # Compute bounds for data fusion, needlessly computing for multiple bands but shouldn't be a big deal
                        # First indices, then points for xy, then extract coordinates from xy
                        # BL, TR --> (minx, miny), (maxx, maxy)
                        # Take special care with y axis; with xy indices, 0 should be top (coords 0/min is bottom)
                        left = x_offset * x_size / upscale_factor
                        top = y_offset * y_size / upscale_factor
                        right = left + x_size / upscale_factor
                        bottom = top + y_size / upscale_factor
                        tr = rasterio.transform.xy(bandset.transform, left, bottom)
                        bl = rasterio.transform.xy(bandset.transform, right, top)

                        transformer = Transformer.from_crs(bandset.crs, 4326)
                        bl = transformer.transform(bl[0], bl[1])
                        tr = transformer.transform(tr[0], tr[1])

                        left = bl[0]
                        bottom = bl[1]
                        right = tr[0]
                        top = tr[1]
                        bounds = (left, bottom, right, top, bandset.transform, bandset.crs)

                    band_values = bandset.read(band_index,
                                               out_shape=(
                                                   window.height,
                                                   window.width
                                               ),
                                               resampling=Resampling.bilinear,
                                               # masked=True,
                                               window=res_window,
                                               )
                    grid[:, :, b] = band_values
                band_index += 1
        resolution_index += 1

    if (return_bounds):
        return (grid, bounds)
    else:
        return (grid)


def compute_MAE_and_MSE(true, pred, mask_2d):
    # true = np.moveaxis(true,0,2)
    mask = np.zeros((pred.shape[0], pred.shape[1], pred.shape[2]))
    for b in range(0, true.shape[2]):
        mask[:, :, b] = mask_2d.copy()

    diff_mae = np.absolute(true - pred)
    diff_mse = (true-pred)**2

    flattened_diff_mae = diff_mae.reshape((diff_mae.shape[0] * diff_mae.shape[1] * diff_mae.shape[2]))
    flattened_mask_mae = mask.copy().reshape((mask.shape[0] * mask.shape[1] * mask.shape[2]))
    flattened_diff_mse = diff_mse.reshape((diff_mse.shape[0]*diff_mse.shape[1]*diff_mse.shape[2]))
    flattened_mask_mse = mask.copy().reshape((mask.shape[0]*mask.shape[1]*mask.shape[2]))

    flattened_diff_mae = flattened_diff_mae[flattened_mask_mae > 20]
    flattened_diff_mse = flattened_diff_mse[flattened_mask_mse > 20]

    mae = np.nanmean(flattened_diff_mae)
    mse = np.nanmean(flattened_diff_mse)


    return mae, mse



def run_patch(base_path, target_n, mask_n, m1_n, y_size, x_size, y_offset, x_offset, buffer_mask=False,
              cloud_threshold=20, method="exact", mask_buffer_size=5):
    target_path = base_path + "/" + target_n + ".zip"
    feature_path = base_path + "/" + mask_n + ".zip"
    mask_path = base_path + "/" + m1_n + ".zip"

    # Load target and mask first, just return target if no cloudy pixels in mask
    target = load_product_windowed(target_path, y_size, x_size, y_offset, x_offset).astype(float)
    if (not (np.any(target > 0))):
        # Black stuff for non-filling scenes?
        print("All zeros for scene: ", scene, y_offset, x_offset)
        print("\n")
        return_list = [np.zeros(target.shape), "no clouds", "no clouds", "no clouds", "no clouds", "no clouds"]
        return return_list
    mask = load_product_windowed(mask_path, y_size, x_size, y_offset, x_offset, keep_bands=["CLD"],
                                 bands_20m={"CLD": 0}).astype(float)[:, :, 0]

    if (not (np.any(mask > cloud_threshold))):
        return_list = [target, "no clouds", "no clouds", "no clouds", "no clouds", "no clouds"]
        return return_list

    # If there are any cloudy pixels, load features and run algorithms
    features = load_product_windowed(feature_path, y_size, x_size, y_offset, x_offset).astype(float)

    if (buffer_mask):
        mask_grid = mask_buffer(mask, mask_buffer_size)

    target_cloudy = target.copy()
    for i in range(0, target_cloudy.shape[0]):
        for j in range(0, target_cloudy.shape[1]):
            if (mask[i, j] > cloud_threshold):
                a = np.ones(target_cloudy.shape[2]) * np.nan
                target_cloudy[i, j, :] = a

    # Create dictionary to which the different bands are added after VPint
    manager = multiprocessing.Manager()
    pred_dict = manager.dict()

    grid_combos = []
    bands = []
    procs = len(range(0, target_cloudy.shape[2]))

    # Create lists containing the bands in order
    for b in range(0, target_cloudy.shape[2]):
        targetc = target_cloudy[:, :, b]
        feature = features[:, :, b]
        band0 = b
        grid_combos.append([targetc, feature])
        bands.append(band0)

    # Start the processes
    vpint_start = time.time()
    jobs = []
    for i in range(0, procs):
        process = multiprocessing.Process(target=multi_VPint_interpolation,
                                          args=(pred_dict, grid_combos[i], bands[i]))
        jobs.append(process)
    for j in jobs:
        j.start()

    for j in jobs:
        j.join()

        # Ensure all of the processes have finished
        if j.is_alive():
            print("Job is not finished!")
    vpint_end = time.time()

    #Sort the dictionary after running VPint on the bands
    start_dict = time.time()
    sorted_dict = dict(sorted(pred_dict.items()))
    pred0 = np.array([*sorted_dict.values()])
    pred1 = pred0.swapaxes(0, 2)
    pred_grid_final = pred1.swapaxes(0, 1)
    end_dict = time.time()
    time_dict_sort = end_dict - start_dict


    start_metrics = time.time()
    if (pred_grid_final.shape != target.shape):
        print("Mismatched shapes for ", ", True shape:", target.shape, ", pred shape:", pred_grid_final.shape)
        return_list = ["error", "error", "error", "error", "error", "error", "error", "error"]
        return return_list
    else:
        # Remove edge case nans; should only happen on true (faulty pixels), but
        # do both just in case.
        pred_mean = np.nanmean(pred_grid_final)
        true_mean = np.nanmean(target)
        pred_vals = np.nan_to_num(pred_grid_final, nan=pred_mean)
        true_vals = np.nan_to_num(target, nan=true_mean)

        mae, mse = compute_MAE_and_MSE(true_vals, pred_vals, mask)
    end_metrics = time.time()
    metrics_time = end_metrics - start_metrics

    vpint_time = vpint_end - vpint_start

    return_list = [pred_grid_final, vpint_time, metrics_time, time_dict_sort, mae, mse]

    return return_list



def mask_buffer(mask, passes=1):
    for p in range(0, passes):
        new_mask = mask.copy()
        for i in range(0, mask.shape[0]):
            for j in range(0, mask.shape[1]):
                if (np.isnan(mask[i, j])):
                    if (i > 0):
                        new_mask[i - 1, j] = np.nan
                    if (i < mask.shape[0] - 1):
                        new_mask[i + 1, j] = np.nan
                    if (j > 0):
                        new_mask[i, j - 1] = np.nan
                    if (j < mask.shape[1] - 1):
                        new_mask[i, j + 1] = np.nan
        mask = new_mask
    return (mask)

if __name__ == '__main__':
    size_y = 256
    size_x = 256

    log = rasterio.logging.getLogger()
    log.setLevel(rasterio.logging.FATAL)

    # Run

    base_path = sys.argv[1]
    path_results = sys.argv[2]

    # scene[0] = name, scene[1] = target, scene[2] = 1m, scene[3] = mask

    # Iterate scenes and patches
    for scene in scenes:
        scene_dir = path_results + "/" + scene[0]
        if (not (os.path.exists(scene_dir))):
            try:
                os.mkdir(scene_dir)
            except:
                pass

        scene_start = time.time()

        scene_height = -1
        scene_width = -1
        ref_product_path = base_path + "/" + scene[1] + ".zip"
        with rasterio.open(ref_product_path) as raw_product:
            product = raw_product.subdatasets
        with rasterio.open(product[1]) as fp:
            scene_height = fp.height
            scene_width = fp.width

        max_row = int(str(scene_height / size_y).split(".")[0])
        max_col = int(str(scene_width / size_x).split(".")[0])

        # Shuffle indices to allow multiple tasks to run
        row_list = np.arange(max_row)
        col_list = np.arange(max_col)
        np.random.shuffle(row_list)
        np.random.shuffle(col_list)

        end_pre_proc = time.time()

        scene_metrics = [["patch", "patch runtime", "VPint runtime", "Metrics runtime", "Dictionary sort runtime", "MAE", "MSE"]]
        # Iterate
        for y_offset in row_list:
            for x_offset in col_list:
                patch_name = "r" + str(y_offset) + "_c" + str(x_offset)

                patch_start = time.time()
                outp = run_patch(base_path, scene[1], scene[2], scene[3], size_y, size_x, y_offset, x_offset, buffer_mask=False)
                patch_end = time.time()
                scene_metrics.append([patch_name, patch_end - patch_start, outp[1], outp[2], outp[3], outp[4], outp[5]])

        scene_end = time.time()
        scene_metrics.append(["Entire scene", scene_end - scene_start, "nan", "nan", "nan", "nan", "nan"])
        scene_name = scene_dir + "/" + scene[0] + "_results_multi.csv"

        with open(scene_name, 'w', newline='') as file:
            writer = csv.writer(file, dialect='excel', delimiter=',')
            writer.writerows(scene_metrics)

        print("Terminated successfully")



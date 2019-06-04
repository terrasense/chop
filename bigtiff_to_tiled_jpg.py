import os, multiprocessing, re, shutil
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
import rasterio
from rasterio import windows
import numpy as np
import cv2

def cropTifToMaskCoordinates(tif_path, out_folder, masks_folder, mask_name, out_format):
    try:
        bounds = os.path.splitext(mask_name)[0]
        mask_format = os.path.splitext(mask_name)[-1]
        mask_path = os.path.sep.join([masks_folder, mask_name])
        with Image.open(mask_path) as mask:
            out_width, out_height = mask.size
    except Exception as e:
        print(e)
        return
    if (mask_format not in ['.png', '.jpg', '.jpeg']):
        print('File format not supported: ', mask_name)
        return
    bounds = os.path.splitext(mask_name)[0]
    new_img = os.path.sep.join([out_folder, bounds + '.' + out_format])
    coords = []
    for bound in bounds.split('__'):
        coords.append(float(bound.replace('_','.')))
    coords_left, coords_bottom, coords_right, coords_top = coords
    try:
        tif = rasterio.open(tif_path)
    except Exception as e:
        print(e)
        return
    tif_left, tif_bottom, tif_right, tif_top = tif.bounds
    window_maker = windows.WindowMethodsMixin()
    window_maker.transform = tif.transform
    window_maker.height = tif.height
    window_maker.width = tif.width
    out_img_window = window_maker.window(*coords)
    tif_window = window_maker.window(*tif.bounds)
    if (windows.intersect([out_img_window, tif_window])):
        intersection_window = out_img_window.intersection(tif_window) 
        img_arr = np.stack([np.zeros((int(out_img_window.width),
                                    int(out_img_window.height))) for i in range(3)],
                           axis=2)
        temp_arr = np.stack([tif.read(4-i, window=intersection_window)
                             for i in range(1,4)], axis=2)
        arr_col_off = int(intersection_window.col_off - out_img_window.col_off)
        arr_row_off = int(intersection_window.row_off - out_img_window.row_off)
        img_arr[arr_row_off:arr_row_off+temp_arr.shape[0],
                arr_col_off:arr_col_off+temp_arr.shape[1]] = temp_arr
        img = Image.fromarray(img_arr)
        img = img.resize((out_width, out_height), Image.ANTIALIAS)
        img.save(new_img)
        #cv2.imwrite(new_img, img_arr.astype('uint8'))
        temp_arr = None
        intersection_window = None
        out_img_window = None
        tif_window = None
        tif.close()
        shutil.move(mask_path, new_img.replace('.'+out_format, mask_format))
        return 1
    tif.close()

def tifToTiles(tif_path, masks_folder, out_format, mask_folder_id, **kwargs):
    mask_files = os.listdir(masks_folder)
    masks_object = re.sub(mask_folder_id, '', os.path.basename(masks_folder))
    out_folder = tif_path.replace('.tif', str(r'_img_tiles')) + '_' + masks_object
    try:
        os.mkdir(out_folder)
    except Exception as e:
        #print('Files seem to already exist. Remove them if new ones are needed.', out_folder)
        shutil.rmtree(out_folder) 
        os.mkdir(out_folder)
    print('Starting to convert tif to tiled images of ' + masks_object, tif_path)
    num_cores = max(1, multiprocessing.cpu_count()-1)
    results = Parallel(n_jobs=num_cores)(delayed(cropTifToMaskCoordinates)
                                         (tif_path, out_folder, masks_folder,
                                          mask_name, out_format) 
                                         for mask_name in tqdm(mask_files))
    print('length '+tif_path +' '+masks_object, np.count_nonzero(results))
    print('Done tiling TIFF: ', tif_path)

parser = ArgumentParser(description='Transforms TIFF to images of the same resolution as their relative binary masks, bounded by the same coordinate.')
parser.add_argument('-t', '--tifpath', dest='tif_path', required=True, help='The path to the TIFF.', metavar='FILE')
parser.add_argument('-m', '--masksfolder', dest='masks_folder', type=str, required=True, help='Path of the folder containing the masks.')
parser.add_argument('-f', '--format', dest='out_format', type=str, default='jpg', help='File type of the output image.')
parser.add_argument('-mfi', '--mask_folder_id', dest='mask_folder_id', type=str, default='_binary_imgs', help='The identification string that the binary mask folders have.')
parser.set_defaults(func=tifToTiles)
args = parser.parse_args()
args.func(**vars(args))

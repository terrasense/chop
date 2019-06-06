import os, multiprocessing, re, shutil
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
import rasterio
from rasterio import windows
import numpy as np

def getMaskCoordsAndOutDim(masks_folder, mask_name):
    bounds = os.path.splitext(mask_name)[0]
    mask_path = os.path.sep.join([masks_folder, mask_name])
    with Image.open(mask_path) as mask:
        out_width, out_height = mask.size
    coords = []
    for bound in bounds.split('__'):
        coords.append(float(bound.replace('_','.')))
    del bounds
    return coords, out_width, out_height

def makeTifAndImageWindow(transform, width, height, bounds, coords):
    window_maker = windows.WindowMethodsMixin()
    window_maker.transform = transform
    window_maker.height = height
    window_maker.width = width
    out_img_window = window_maker.window(*coords)
    tif_window = window_maker.window(*bounds)
    del window_maker
    return tif_window, out_img_window

def cropTifToMaskCoordinates(tif_path, out_folder, masks_folder, mask_name,
                             out_format, tif_left, tif_bottom, tif_right,
                             tif_top):
    found_overlap = 0
    coords,out_width,out_height=getMaskCoordsAndOutDim(masks_folder,mask_name)
    mask_format = os.path.splitext(mask_name)[-1]
    if mask_format not in ['.png', '.jpg', '.jpeg']:
        return found_overlap
    new_img_path = os.path.sep.join([out_folder, mask_name.replace(mask_format,
                                                                   '.'+out_format)])
    mask_path = os.path.sep.join([masks_folder, mask_name])
    with rasterio.open(tif_path) as tif:
        tif_window, out_img_window = makeTifAndImageWindow(tif.transform,
                                                           tif.width,
                                                           tif.height,
                                                          tif.bounds,
                                                          coords)
        if (windows.intersect([out_img_window, tif_window])):
            intersection_window = out_img_window.intersection(tif_window) 
            img_arr = np.stack([np.zeros((int(out_img_window.width),
                                        int(out_img_window.height))) for i in range(3)],
                               axis=2)
            try:
                temp_arr = np.stack([tif.read(4-i, window=intersection_window)
                                 for i in range(1,4)], axis=2)
                arr_col_off = int(intersection_window.col_off - out_img_window.col_off)
                arr_row_off = int(intersection_window.row_off - out_img_window.row_off)
                img_arr[arr_row_off:arr_row_off+temp_arr.shape[0],
                        arr_col_off:arr_col_off+temp_arr.shape[1]] = temp_arr
                img = Image.fromarray(img_arr.astype('uint8'))
                del temp_arr, img_arr
                img = img.resize((out_width, out_height), Image.ANTIALIAS)
                if (img.convert("L").getextrema() != (0,0)):
                    mask_img = Image.open(mask_path)
                    mask_img = mask_img.convert('RGB')
                    images = [img, mask_img]
                    widths, heights = zip(*(i.size for i in images))
                    total_width = sum(widths)
                    max_height = max(heights)
                    new_img = Image.new('RGB', (total_width, max_height), 'black')
                    x_offset = 0
                    for i in images:
                        new_img.paste(i, (x_offset, 0))
                        x_offset += i.size[0]
                    new_img.save(new_img_path)
                    images = []
                    new_img.close()
                    mask_img.close()
                    found_overlap = 1
                img.close()
            except Exception as e:
                print(e)
        del out_img_window, tif_window, intersection_window
    return found_overlap

def getOverlapping(tif_left, tif_bottom, tif_right, tif_top, mask_name):
    bounds = os.path.splitext(mask_name)[0].split('__')
    msk_left, msk_bottom, msk_right, msk_top = [float(bound.replace('_','.'))
                                                for bound in bounds] 
    left = max(msk_left, tif_left)
    bottom = max(msk_bottom, tif_bottom)
    right = min(msk_right, tif_right)
    top = min(msk_top, tif_top)
    if (left < right and bottom < top):
        return mask_name
    else:
        return None

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
    with rasterio.open(tif_path) as tif:
        tif_left, tif_bottom, tif_right, tif_top = tif.bounds
    print('Starting to convert tif to tiled images of ' + masks_object, tif_path)
    num_cores = max(1, multiprocessing.cpu_count()-1)
    intersecting = Parallel(n_jobs=num_cores)(delayed(getOverlapping)
                                              (tif_left, tif_bottom,
                                               tif_right, tif_top,
                                               mask_name)
                                              for mask_name in
                                              tqdm(mask_files))
    intersecting = [entry for entry in intersecting if entry is not None]
    print('Number of preliminary intersections: ', len(intersecting))
    for mask_name in tqdm(intersecting):
        cropTifToMaskCoordinates(tif_path, out_folder, masks_folder,
                                 mask_name, out_format, tif_left,
                                 tif_bottom, tif_right, tif_top) 
#    results = Parallel(n_jobs=num_cores)(delayed(cropTifToMaskCoordinates)
#                                         (tif_path, out_folder, masks_folder,
#                                          mask_name, out_format, tif_left,
#                                          tif_bottom, tif_right, tif_top) 
#                                         for mask_name in tqdm(intersecting))
   # print('Total image overlaps found between '+tif_path+' and '+masks_object,
   #       np.count_nonzero(results))
    print('Done tiling TIFF: ', tif_path)

parser = ArgumentParser(description='Transforms TIFF to images of the same resolution as their relative binary masks, bounded by the same coordinate.')
parser.add_argument('-t', '--tifpath', dest='tif_path', required=True, help='The path to the TIFF.', metavar='FILE')
parser.add_argument('-m', '--masksfolder', dest='masks_folder', type=str, required=True, help='Path of the folder containing the masks.')
parser.add_argument('-f', '--format', dest='out_format', type=str, default='jpg', help='File type of the output image.')
parser.add_argument('-mfi', '--mask_folder_id', dest='mask_folder_id', type=str, default='_binary_imgs', help='The identification string that the binary mask folders have.')
parser.set_defaults(func=tifToTiles)
args = parser.parse_args()
args.func(**vars(args))

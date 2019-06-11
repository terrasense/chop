import os, re, shutil
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
import rasterio
from rasterio import windows
import numpy as np
import multiprocessing as mp

def getCoordsAndSize(masks_folder, mask_name):
    bounds = os.path.splitext(mask_name)[0]
    mask_path = os.path.sep.join([masks_folder, mask_name])
    with Image.open(mask_path) as mask:
        out_width, out_height = mask.size
    coords = []
    for bound in bounds.split('__'):
        coords.append(float(bound.replace('_','.')))
    del bounds
    return mask_name, coords, (out_width, out_height)

def makeWindow(transform, width, height, bounds, mask_name=None, mask_size=None):
    window_maker = windows.WindowMethodsMixin()
    window_maker.transform = transform
    window_maker.height = height
    window_maker.width = width
    out_window = window_maker.window(*bounds)
    del window_maker
    return mask_name, out_window, mask_size

def getIntersection(tif_window, mask_window, mask_name, mask_size):
    if windows.intersect(mask_window, tif_window):
        return (mask_name,
                mask_window, mask_size, mask_window.intersection(tif_window))
    return None

def getOverlapping(tif_bounds, mask_name):
    bounds = os.path.splitext(mask_name)[0].split('__')
    msk_left, msk_bottom, msk_right, msk_top = [float(bound.replace('_','.'))
                                                for bound in bounds] 
    left = max(msk_left, tif_bounds[0])
    bottom = max(msk_bottom, tif_bounds[1])
    right = min(msk_right, tif_bounds[2])
    top = min(msk_top, tif_bounds[3])
    if (left < right and bottom < top):
        return mask_name
    else:
        return None


def makeImage(temp_arr, inter_window, mask_window, mask_size, mask_name,
              masks_folder, out_folder, out_format):
    found_overlap = 0
    out_width, out_height= mask_size
    mask_format = os.path.splitext(mask_name)[-1]
    if mask_format not in ['.png', '.jpg', '.jpeg']:
        return found_overlap
    new_img_path = os.path.join(out_folder, mask_name.replace(mask_format,
                                                                   '.'+out_format))
    mask_path = os.path.join(masks_folder, mask_name)
    img_arr = np.zeros(temp_arr.shape)
    arr_col_off = int(inter_window.col_off-mask_window.col_off)
    arr_row_off = int(inter_window.row_off-mask_window.row_off)
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
        del new_img, mask_img, images
        found_overlap = 1
    img.close()
    del img, mask_window, inter_window
    return found_overlap

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
    with rasterio.open(tif_path, 'r') as tif:
        tif_transform = tif.transform
        tif_bounds = tif.bounds
        tif_width = tif.width
        tif_height = tif.height
    num_cores = max(1, mp.cpu_count())
    pool = mp.Pool(num_cores)
    intersecting = pool.starmap_async(getOverlapping, [(tif_bounds, mask_name)
                                              for mask_name in
                                              tqdm(mask_files)], chunksize=10000)
    intersecting = [entry for entry in intersecting.get() if entry is not None]
    print('Getting masks\' coordinates and sizes.') 
    mask_info = pool.starmap_async(getCoordsAndSize, [(masks_folder, mask_name) for
                                                  mask_name in
                                            tqdm(intersecting)], chunksize=10000)
    del intersecting
    mask_info = mask_info.get()
    print('Making mask windows.')
    tif_window = makeWindow(tif_transform, tif_width, tif_height, tif_bounds)
    mask_windows = pool.starmap_async(makeWindow, [(tif_transform, tif_width, tif_height,
                                          bounds, mask_name, size) for mask_name,
                                         bounds, size in
                                         tqdm(mask_info)])
    del mask_info
    mask_windows = mask_windows.get()
    print('Getting intersection windows.')
    inter_windows = pool.starmap_async(getIntersection, [(tif_window[1], mask_window,
                                               mask_name, mask_size) for
                                              mask_name, mask_window, mask_size
                                              in tqdm(mask_windows)])
    del mask_windows
    inter_windows = inter_windows.get()
    with rasterio.Env():
        with rasterio.open(tif_path) as tif:
            batch_size = num_cores*5
            b_range = range(0, len(inter_windows), batch_size)
            for index, step in enumerate(b_range):
                saved_images = pool.starmap_async(makeImage,
                                        [(np.moveaxis(tif.read(
                                            window=inter_window),0,-1),
                                            inter_window, mask_window,
                                            mask_size, mask_name, masks_folder,
                                            out_folder, out_format) for mask_name,
                                            mask_window, mask_size,
                                            inter_window in
                                            tqdm(inter_windows[step:
                                                               b_range[index+1]])])
                saved_images.get()
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

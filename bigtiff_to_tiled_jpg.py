import gdal, os, multiprocessing, re
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser

def cropTifToMaskCoordinates(tif_path, out_folder, masks_folder, file, out_format):
    try:
        bounds = os.path.splitext(file)[0]
        format = os.path.splitext(file)[1]
        mask = Image.open(masks_folder + '\\' + file)
        out_width, out_height = mask.size
        mask.close()
    except Exception as e:
        print(e)
        return
    if (format not in ['.png', '.jpg', '.jpeg']):
        print('File format not supported. ', file)
        return
    bounds = os.path.splitext(file)[0]
    new_img = out_folder + '\\\\' + bounds + '.' + out_format
    projWin = []
    coords = []
    for bound in bounds.split('__'):
        coords.append(float(bound.replace('_','.')))
    x0, y0, x1, y1 = coords
    projWin += [x0, y1, x1, y0]
    tif = gdal.Open(tif_path)
    transform = tif.GetGeoTransform()
    pixelWidth = transform[1]
    pixelHeight = transform[5]
    cols = tif.RasterXSize
    rows = tif.RasterYSize
    xLeft = transform[0]
    yTop = transform[3]
    xRight = xLeft+cols*pixelWidth
    yBottom = yTop+rows*pixelHeight
    x_start = max(x0, xLeft)
    x_end = min (x1, xRight)
    y_start = max(y0, yBottom)
    y_end = min(y1, yTop)
    if (x_start < x_end and y_start < y_end):
        gdal.Translate(new_img, tif, projWin=projWin, width=out_width, height=out_height)
    del tif

def tifToTiles(tif_path, masks_folder, out_format, mask_folder_id, **kwargs):
    mask_files = os.listdir(masks_folder)
    masks_object = re.sub(mask_folder_id, '', masks_folder.split('\\')[-1])
    out_folder = tif_path.replace('.tif', str(r'_img_tiles')) + '_' + masks_object
    try:
        os.mkdir(out_folder)
    except Exception as e:
        print('Files seem to already exist. Remove them if new ones are needed.', out_folder)
        return
    print('Starting to convert tif to tiled images.', tif_path)
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(cropTifToMaskCoordinates)(tif_path, out_folder, masks_folder, file, out_format) for file in tqdm(mask_files))
    print('Done tiling TIFF: ', tif_path)

parser = ArgumentParser(description='Transforms TIFF to images of the same resolution as their relative binary masks, bounded by the same coordinate.')
parser.add_argument('-t', '--tifpath', dest='tif_path', required=True, help='The path to the TIFF.', metavar='FILE')
parser.add_argument('-m', '--masksfolder', dest='masks_folder', type=str, required=True, help='Path of the folder containing the masks.')
parser.add_argument('-f', '--format', dest='out_format', type=str, default='jpg', help='File type of the output image.')
parser.add_argument('-mfi', '--mask_folder_id', dest='mask_folder_id', type=str, default='_binary_imgs', help='The identification string that the binary mask folders have.')
parser.set_defaults(func=tifToTiles)
args = parser.parse_args()
args.func(**vars(args))
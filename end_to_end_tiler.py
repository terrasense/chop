import os, multiprocessing, subprocess
from joblib import Parallel, delayed
import timeit
from argparse import ArgumentParser
import subprocess

def tileShp(shp_path, pixel_res, extra_shps, stride, out_width, out_height):
    if (shp_path.endswith('.shp')):
        subprocess.call(['python', 'shp_to_binary.py', '-shp', shp_path,
                         '--pixel_res', str(pixel_res), '--extra_shapefiles',
                        extra_shps[0], '-s', str(stride), '--width',
                         str(out_width),
                         '--height', str(out_height)])
        if extra_shps == []:
            imgs_path = shp_path.replace('.shp', '_binary_imgs')
        else:
            imgs_path = shp_path.replace('.shp', '_and_other_binary_imgs')
        return imgs_path

def tileTIFF(tif_path, masks, out_tif_folder):
    if (tif_path.endswith('.tif')):
        subprocess.call(['python', 'bigtiff_to_tiled_jpg.py', '-t', tif_path,
                         '-m', masks, '-o', out_tif_folder])

def tileEverything(shps_folder, tiffs_folder, out_width, out_height,
                   stride, pixel_res, masks, extra_shps, out_tif_folder):
    print("Master Tiler starting.")
    num_cores = max(1, multiprocessing.cpu_count())
    if shps_folder != None:
        masks_folders = Parallel(n_jobs=num_cores)(delayed(tileShp)(shp.path,
                                                                    pixel_res,
                                                                    extra_shps,
                                                                   stride,
                                                                    out_width,
                                                                    out_height)
                                                   for shp in
                                                   os.scandir(shps_folder))
    else:
        masks_folders = [masks]
    for masks in masks_folders:
        if (masks != None):
            Parallel(n_jobs=num_cores)(delayed(tileTIFF)(tif.path, masks,
                                                         out_tif_folder) for tif in os.scandir(tiffs_folder))

def timer(shps_folder, tiffs_folder, out_width, out_height,
          stride, pixel_res, extra_shps, masks, out_tif_folder, **kwargs):
    t = timeit.Timer("tileEverything({0!a}, {1!a}, {2}, {3}, {4}, {5}, {6}, {7}, {8!a})"
                     .format(shps_folder, tiffs_folder, out_width,
                             out_height, stride, pixel_res, masks, extra_shps,
                             out_tif_folder),
                     setup="from __main__ import tileEverything")
    print( "\t{:.2f}s\n".format(t.timeit(1)))

parser = ArgumentParser(description='Transforms shapefile to binary masks and TIFF to images of the same resolution as their relative binary masks, bounded by the same coordinate.')
parser.add_argument('-s', '--shps_folder', dest='shps_folder', type=str,
                    default=None, help='Path of the folder containing the masks.')
parser.add_argument('-t', '--tiffs_folder', dest='tiffs_folder', type=str,
                    required=True, help='Path to the folder containing the TIFFs.')
parser.add_argument('--width', dest='out_width', type=int, default=1000,
                    help='Width of output image, in pixels.')
parser.add_argument('--height', dest='out_height', type=int, default=1000,
                    help='Height of output image, in pixels.')
parser.add_argument('--stride', dest='stride', type=int, default=750,
                    help='Step size between tile edges. Default is 500.')
parser.add_argument('--pixel_res', dest='pixel_res', type=float, default=0.04,
                    help='Resolution of pixels of saved image. Defaults to 0.04.')
parser.add_argument('--extra_shapefiles', dest='extra_shps', nargs='+',
                    default=[],
                    help='Extra shapefiles to add in grey tones. Should be ' + 
                    'provided in the desired depth wanted.')
parser.add_argument('-m', '--masks_folders', dest='masks', type=str,
                    default=None,
                    help='Masks folder as alternative to shapefiles.')
parser.add_argument('-ot', '--out_tif_folder', dest='out_tif_folder', type=str,
                    default=None,
                    help='Output folder for tif files.')
parser.set_defaults(func=timer)
args = parser.parse_args()
args.func(**vars(args))

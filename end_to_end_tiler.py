import os, multiprocessing, subprocess
from joblib import Parallel, delayed
import timeit
from argparse import ArgumentParser
import subprocess

def tileShp(shp_path):
    if (shp_path.endswith('.shp')):
        subprocess.call(['python', 'shp_to_binary.py', '-shp', shp_path])
        return shp_path.replace('.shp', '_binary_imgs')

def tileTIFF(tif_path, masks):
    if (tif_path.endswith('.tif')):
        subprocess.call(['python', 'bigtiff_to_tiled_jpg.py', '-t', tif_path, '-m', masks])

def tileEverything(shps_folder, tiffs_folder, out_width, out_height, stride):
    print("Master Tiler starting.")
    num_cores = max(1, multiprocessing.cpu_count()-1)
    masks_folders = Parallel(n_jobs=num_cores)(delayed(tileShp)(shp.path) for shp in os.scandir(shps_folder))
    for masks in masks_folders:
        if (masks != None):
            Parallel(n_jobs=num_cores)(delayed(tileTIFF)(tif.path, masks) for tif in os.scandir(tiffs_folder))

def timer(shps_folder, tiffs_folder, out_width, out_height, stride, **kwags):
    t = timeit.Timer("tileEverything({0!a}, {1!a}, {2}, {3}, {4})".format(shps_folder, tiffs_folder, out_width, out_height, stride),
                     setup="from __main__ import tileEverything")
    print( "\t{:.2f}s\n".format(t.timeit(1)))

parser = ArgumentParser(description='Transforms shapefile to binary masks and TIFF to images of the same resolution as their relative binary masks, bounded by the same coordinate.')
parser.add_argument('-s', '--shps_folder', dest='shps_folder', type=str, required=True, help='Path of the folder containing the masks.')
parser.add_argument('-t', '--tiffs_folder', dest='tiffs_folder', type=str, required=True, help='Path to the folder containing the TIFFs.')
parser.add_argument('--width', dest='out_width', type=int, default=1000, help='Width of output image, in pixels.')
parser.add_argument('--height', dest='out_height', type=int, default=1000, help='Height of output image, in pixels.')
parser.add_argument('--stride', dest='stride', type=int, default=750, help='Step size between tile edges. Default is 500.')
parser.set_defaults(func=timer)
args = parser.parse_args()
args.func(**vars(args))

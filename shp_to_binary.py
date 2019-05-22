import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from argparse import ArgumentParser
import os, multiprocessing, gdal
from math import sqrt
from PIL import Image

'''
input: geopandas raw items (Polygons, MultiPolygons, etc)
output: list of Polygons
'''
def getPolygons(items):
    polygons = []
    if (type(items) == None):
        return polygons
    if (items.type == 'Polygon' or items.type == 'Point' or items.type == 'Line'):
        polygons.append(items)
        return polygons
    for item in items:
        if (item.type == 'MultiPolygon'):
            polygons += getPolygons(item)
    return polygons

'''
input: x and y pixel resolution, DotsPerInch, pyplot axis, x & y pix. to add
output: a pyplot axis with the desired dimensions
'''
def setAxisSize(res_x,res_y,dpi, ax, add_x_pix=0,add_y_pix=0):
    w = res_x/dpi
    h =res_y/dpi
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw+add_x_pix/dpi, figh+add_y_pix/dpi)
    return ax

def makeTile(out_path, pixel_res, res_x, res_y, add_pix_x, add_pix_y, xlist, ylist, x_index, y_index, out_format, shp_path):
    dataframe = gpd.read_file(shp_path)
    fig = plt.figure(frameon=False, dpi=1/pixel_res*39.37)
    ax = fig.add_subplot(111)
    dpi = fig.get_dpi()
    ax = setAxisSize(res_x, res_y, dpi, ax, add_pix_x, add_pix_y)
    ax.set_aspect('equal')
    fig.canvas.draw()
    ax.margins(0)
    ax.tick_params(which='both', direction='in')
    ax.axis('off')
    ax.set_axis_off()
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    dataframe.plot(ax=ax, color='white')
    plt.xlim([xlist[x_index],xlist[x_index+1]])
    plt.ylim([ylist[y_index],ylist[y_index+1]])
    plt.savefig(out_path, format=out_format, aspect='normal', facecolor='black', pad_inches=0, bbox_inches=extent, dpi='figure')
    plt.close()
    dataframe = None

def getProperDimensions(out_path, pixel_res, out_width, out_height, add_pix_x, add_pix_y, xlist, ylist, x_index, y_index, out_format, shp_path):
    makeTile(out_path, pixel_res, out_width, out_height, add_pix_x, add_pix_y, xlist, ylist, 0, 0, out_format, shp_path)
    img = Image.open(out_path, mode='r')
    trail_width, trail_height = img.size
    img.close()
    os.remove(out_path)
    if (trail_width < out_width):
        add_pix_x += 5
        getProperDimensions(out_path, pixel_res, out_width, out_height, add_pix_x, add_pix_y, xlist, ylist, 0, 0, out_format, shp_path)
    if (trail_height < out_height):
        add_pix_y += 5
        getProperDimensions(out_path, pixel_res, out_width, out_height, add_pix_x, add_pix_y, xlist, ylist, 0, 0, out_format, shp_path)
    if (trail_width > out_width):
        add_pix_x -= 1
        getProperDimensions(out_path, pixel_res, out_width, out_height, add_pix_x, add_pix_y, xlist, ylist, 0, 0, out_format, shp_path)
    if (trail_height > out_height):
        add_pix_y -= 1
        getProperDimensions(out_path, pixel_res, out_width, out_height, add_pix_x, add_pix_y, xlist, ylist, 0, 0, out_format, shp_path)
    return add_pix_x, add_pix_y
'''
input: outer bounds of square area, path to the shapefile, output folder path,
       step size for output window, x and y resolutions, DotsPerInch,
       output image format, x and y pixel additions
effect: helper, creates binary images based on the provided shapefile and
        the desired output file format
'''
def tilePolygon(bounds, shp_path, out_folder, stride, pixel_res, out_width, out_height, out_format, add_pix_x, add_pix_y, get_size_params=False):
    if (pixel_res < 1):
        stride = stride * pixel_res
    else:
        stride = stride / pixel_res
    xmin = bounds[0]
    ymin = bounds[1]
    xmax = bounds[2]
    ymax = bounds[3]
    width = xmax-xmin
    height = ymax-ymin
    xlist = np.arange(xmin, xmax, stride)
    ylist = np.arange(ymin, ymax, stride)
    if (len(xlist) == 1):
        xlist = [xmin, xmin+stride]
    if (len(ylist) == 1):
        ylist = [ymin, ymin+stride]
    x_max_iter = len(xlist)-1
    y_max_iter = len(ylist)-1
    if (get_size_params):
        out_path = out_folder + '\\' + 'trail' + '.' + out_format
        add_pix_x, add_pix_y = getProperDimensions(out_path, pixel_res, out_width, out_height, add_pix_x, add_pix_y, xlist, ylist, 0, 0, out_format, shp_path)
        return add_pix_x, add_pix_y
    for x_index, x in enumerate(xlist[:x_max_iter]):
        for y_index, y in enumerate(ylist[:y_max_iter]):
            out_path = out_folder + '\\' + str(x).replace('.','_') + '__' + str(y).replace('.','_') + '__' + str(xlist[x_index+1]).replace('.','_') + '__' + str(ylist[y_index+1]).replace('.','_') + '.' + out_format
            makeTile(out_path, pixel_res, out_width, out_height, add_pix_x, add_pix_y, xlist, ylist, x_index, y_index, out_format, shp_path)

'''
input: path to the shapefile, step size for output window,
       desired width and height of output image (in pixels),
       output image format, x and y pixel additions
effect: creates binary images based on the provided shapefile and
        the desired output file format
'''
def shpToBinaryImg(shp_path, stride, out_width, out_height, out_format, tiff_path, pixel_res, add_pix_x, add_pix_y, **kwargs):
    print('Starting to convert shapefile to binary masks.', shp_path)
    out_folder = shp_path.replace('.shp', str(r'_binary_imgs'))
    pixel_res = abs(pixel_res)
    if (tiff_path != None):
        tiff = gdal.Open(tiff_path)
        transform = tiff.GetGeoTransform()
        pixel_x_res = transform[1]
        pixel_y_res = transform[5]
        pixel_res = sqrt(abs(pixel_x_res*pixel_y_res))
        del tiff
    try:
        os.mkdir(out_folder)
    except Exception as e:
        print('Files already exist. Remove them if new ones are needed.', out_folder)
        return
    dataframe = gpd.read_file(shp_path)
    geometry = dataframe.geometry
    polygons = []
    for item in geometry.iteritems():
        if (type(item[1]) == None):
            break
        polygons += getPolygons(item[1])

    centroids = []
    polybounds = []
    for poly in polygons:
        centroids.append(poly.centroid)
        polybounds.append(poly.bounds)
    if (len(polybounds) < 1):
        print('No polygons found in shapefile.')
        return
    map_coords = dataframe.total_bounds
    out_path = out_folder + '\\' + str(map_coords[0]).replace('.','_') + '__'
    for lim in map_coords[1:]:
        out_path += '__' + str(lim).replace('.','_')
    out_path += '.' + out_format
    dataframe = None
    geometry = None
    add_pix_x, add_pix_y = tilePolygon(polybounds[1], shp_path, out_folder, stride, pixel_res, out_width, out_height, out_format, add_pix_x, add_pix_y, get_size_params=True)
    print('Pixels added to width: ', add_pix_x)
    print('Pixels added to height: ', add_pix_y)
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores,)(delayed(tilePolygon)(bounds, shp_path, out_folder, stride, pixel_res, out_width, out_height, out_format, add_pix_x, add_pix_y) for bounds in tqdm(polybounds))
    print('Done tiling and masking shapefile: ', shp_path)
parser = ArgumentParser(description='Transforms the given shapefile to various tiled binary images of the desired format and size.')
parser.add_argument('-shp', '--shapefile', dest='shp_path', required=True, help='The path to the shapefile.', metavar='FILE')
parser.add_argument('-s', '--stride', dest='stride', type=int, default=750, help='Step size between tile edges. Default is 500.')
parser.add_argument('--width', dest='out_width', type=int, default=1000, help='Width of output image, in pixels.')
parser.add_argument('--height', dest='out_height', type=int, default=1000, help='Height of output image, in pixels.')
parser.add_argument('--add_x_pixel', dest='add_pix_x', type=int, default=0, help='Fine tune output image resolution, add pixels to width.')
parser.add_argument('--add_y_pixel', dest='add_pix_y', type=int, default=0, help='Fine tune output image resolution, add pixels to height.')
parser.add_argument('-f', '--format', dest='out_format', type=str, default='png', help='File type of the output image.')
parser.add_argument('-t', '--tiff', dest='tiff_path', metavar='FILE', default=None, help='TIFF file for pixel resolution.')
parser.add_argument('--pixel_res', dest='pixel_res', type=float, default=0.04, help='Pixel resolution. Defaults to 0.04 meters.')

parser.set_defaults(func=shpToBinaryImg)
args = parser.parse_args()
args.func(**vars(args))

import geopandas as gpd
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from argparse import ArgumentParser
import shutil, os, multiprocessing, gdal, random
from math import sqrt
from PIL import Image
from shapely.geometry import mapping
'''
input: geopandas raw items (Polygons, MultiPolygons, etc)
output: list of Polygons
'''
def getPolygons(items):
    polygons = []
    if (items == None or type(items) == None):
        return None
    if (items.type == 'Polygon' or items.type == 'Point' or items.type == 'Line'):
        polygons.append(items)
        return polygons
    for item in items:
        polygons += getPolygons(item)
    return polygons

'''
input: x and y pixel resolution, DotsPerInch, pyplot axis, x & y pix. to add
output: a pyplot axis with the desired dimensions
'''
def setAxisSize(out_width,out_height,dpi, ax, add_x_pix=0,add_y_pix=0):
    w = out_width/dpi
    h = out_height/dpi
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw+add_x_pix/dpi, figh+add_y_pix/dpi)
    return ax

'''
input: output file name and path, the resolution of one pixel,
        output width and height, additional pixels needed,
        list of x and y 
effect: saves an image of the desired format with the dimensions provided
'''
def makeTile(out_path, pixel_res, out_width, out_height, add_pix_x, add_pix_y,
             xmin, ymin, xmax, ymax, out_format, shp_path,
             extra_shp_files=[]):
    dataframe = gpd.read_file(shp_path)
    fig = plt.figure(frameon=False, dpi=1/pixel_res*39.37, facecolor='black')
    ax = fig.add_subplot(111)
    dpi = fig.get_dpi()
    ax = setAxisSize(out_width, out_height, dpi, ax, add_pix_x, add_pix_y)
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    fig.canvas.draw()
    ax.margins(0)
    ax.tick_params(which='both', direction='in')
    ax.axis('off')
    ax.set_axis_off()
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    patch = matplotlib.patches.Rectangle((xmin,ymin),xmax-xmin, ymax-ymin,
                                 color='black', zorder=0) 
    ax.add_patch(patch)
    extra_dfs = []
    if (extra_shp_files != []):
        for shp in extra_shp_files:
            extra_dfs.append(gpd.read_file(shp))
        gray_shades = 1/(len(extra_dfs)+1)
        for df in extra_dfs:
            df.plot(ax=ax, facecolor='black', color=str(gray_shades))
            gray_shades += gray_shades
    dataframe.plot(ax=ax, facecolor='black', color='white')
    fig.set_facecolor('black')
    fig.Color = 'black'
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])
    plt.savefig(out_path, aspect='normal',
                facecolor='r', pad_inches=0, bbox_inches=extent,
                dpi='figure', edgecolor='none', transparent=False)
    plt.close()
    img = Image.open(out_path)
    if (img.convert("L").getextrema() == (0,0)):
        img.close()
        os.remove(out_path)
    img.close()
    for df in extra_dfs:
        df = None
    extra_dfs = None
    dataframe = None

'''
input: everything needed to make a tile
output: additional pixels in the x and y dimensions that are needed to make the output image have the desired dimensions
'''
def getProperDimensions(out_path, pixel_res, out_width, out_height, add_pix_x, add_pix_y, xmin, ymin, xmax, ymax, out_format, shp_path):
    makeTile(out_path, pixel_res, out_width, out_height, add_pix_x, add_pix_y, xmin, ymin, xmax, ymax, out_format, shp_path)
    try:
        img = Image.open(out_path, mode='r')
        trail_width, trail_height = img.size
        img.close()
        os.remove(out_path)
    except Exception as e:
        return e
    if (trail_width < out_width):
        add_pix_x += 5
        getProperDimensions(out_path, pixel_res, out_width, out_height, add_pix_x, add_pix_y, xmin, ymin, xmax, ymax, out_format, shp_path)
    if (trail_height < out_height):
        add_pix_y += 5
        getProperDimensions(out_path, pixel_res, out_width, out_height, add_pix_x, add_pix_y, xmin, ymin, xmax, ymax, out_format, shp_path)
    if (trail_width > out_width):
        add_pix_x -= 2
        getProperDimensions(out_path, pixel_res, out_width, out_height, add_pix_x, add_pix_y, xmin, ymin, xmax, ymax, out_format, shp_path)
    if (trail_height > out_height):
        add_pix_y -= 2
        getProperDimensions(out_path, pixel_res, out_width, out_height, add_pix_x, add_pix_y, xmin, ymin, xmax, ymax, out_format, shp_path)
    return add_pix_x, add_pix_y

'''
input: outer bounds of square area, path to the shapefile, output folder path,
       step size for output window, x and y resolutions, DotsPerInch,
       output image format, x and y pixel additions
effect: helper, creates binary images based on the provided shapefile and
        the desired output file format
'''
def tilePolygon(bounds, shp_path, out_folder, stride, pixel_res,
                out_width, out_height, out_format, add_pix_x,
                add_pix_y, x_size, y_size, get_size_params=False,
                extra_shp_files=[]):
    xmin = bounds[0]
    ymin = bounds[1]
    xmax = bounds[2]
    ymax = bounds[3]
    width = xmax-xmin
    height = ymax-ymin
    xlist = np.arange(xmin-x_size/2, xmax, stride)
    ylist = np.arange(ymin-y_size/2, ymax, stride)
    paths = []
    if (len(xlist) <= 1):
        xlist = [xmin-x_size/2, xmin, random.uniform(xmin-x_size/2, xmax-x_size), xmax-x_size]
    if (len(ylist) <= 1):
        ylist = [ymin-y_size/2, ymin, random.uniform(ymin-y_size/2, ymax-y_size),ymax-y_size]
    if (get_size_params):
        out_path = os.path.sep.join([out_folder, 'trail' + '.' + out_format])
        add_pix_x, add_pix_y = getProperDimensions(out_path, pixel_res, out_width, out_height, add_pix_x, add_pix_y, xlist[0], ylist[0], x_size, y_size, out_format, shp_path)
        return add_pix_x, add_pix_y
    for x in xlist:
        for y in ylist:
            out_path = os.path.sep.join([out_folder, str(x).replace('.','_')
                                         + '__' + str(y).replace('.','_') + '__'
                                         + str(x+x_size).replace('.','_') + '__'
                                         + str(y+y_size).replace('.','_') + '.'
                                         + out_format])
            makeTile(out_path, pixel_res, out_width, out_height,
                     add_pix_x, add_pix_y, x, y, x+x_size, y+y_size,
                     out_format, shp_path, extra_shp_files=extra_shp_files)

'''
input: path to the shapefile, step size for output window,
       desired width and height of output image (in pixels),
       output image format, x and y pixel additions
effect: creates binary images based on the provided shapefile and
        the desired output file format
'''
def shpToBinaryImg(shp_path, stride, out_width, out_height, out_format,
                   tiff_path, pixel_res, add_pix_x, add_pix_y,
                   extra_shps, **kwargs):
    if extra_shps == []:
        out_folder = shp_path.replace('.shp', str(r'_binary_imgs'))
    else:
        out_folder = shp_path.replace('.shp', str(r'_and_other_binary_imgs'))
    try:
        os.mkdir(out_folder)
    except Exception as e:
#         print('Files already exist. Remove them if new ones are needed.', out_folder)
        shutil.rmtree(out_folder)
        os.mkdir(out_folder)
    print('Starting to convert shapefile to binary masks.', shp_path)
    pixel_res = abs(pixel_res)
    if (tiff_path != None):
        tiff = gdal.Open(tiff_path)
        transform = tiff.GetGeoTransform()
        pixel_x_res = transform[1]
        pixel_y_res = transform[5]
        pixel_res = sqrt(abs(pixel_x_res*pixel_y_res))
        del tiff
    try:
        dataframe = gpd.read_file(shp_path)
        dataframe.to_file(shp_path)
    except Exception as e:
        dataframe = None
        print('Dataframe error.')
        print(e)
        return
    dataframe = dataframe.dropna()
    map_coords = dataframe.total_bounds
    geometry = dataframe.geometry
    geometries = [geometry]
    if extra_shps != []:
        for shp in extra_shps:
            df = gpd.read_file(shp)
            df.to_file(shp)
            geom = df.geometry
            geometries.append(geom)
            df = None
            geom = None
    polygons = []
    for geometry in geometries:
        for item in geometry.iteritems():
            if (type(item[1]) == None):
                break
            poly = getPolygons(item[1])
            if (poly != None):
                polygons += poly
    polybounds = []
    for poly in polygons:
        polybounds.append(poly.bounds)
    if (len(polybounds) < 1):
        print('No polygons found in shapefile.')
        return
    dataframe = None
    geometry = None
    stride = stride * pixel_res
    x_size = out_width*pixel_res
    y_size = out_height*pixel_res
    add_pix_x, add_pix_y = tilePolygon(polybounds[0], shp_path, out_folder, stride, pixel_res, out_width, out_height, out_format, add_pix_x, add_pix_y, x_size, y_size, get_size_params=True)
    print('Pixels added to width: ', add_pix_x)
    print('Pixels added to height: ', add_pix_y)
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(tilePolygon)(bounds, shp_path,out_folder,
                                                    stride, pixel_res,
                                                    out_width, out_height,
                                                    out_format, add_pix_x,
                                                    add_pix_y, x_size,
                                                    y_size,
                                                    extra_shp_files=extra_shps)
                               for bounds in tqdm(polybounds))
    print('Done tiling and masking shapefile: ', shp_path)

parser = ArgumentParser(description='Transforms the given shapefile to various tiled binary images of the desired format and size.')
parser.add_argument('-shp', '--shapefile', dest='shp_path', required=True, help='The path to the shapefile.', metavar='FILE')
parser.add_argument('-s', '--stride', dest='stride', type=int, default=500, help='Step size between tile edges. Default is 500.')
parser.add_argument('--width', dest='out_width', type=int, default=1000, help='Width of output image, in pixels.')
parser.add_argument('--height', dest='out_height', type=int, default=1000, help='Height of output image, in pixels.')
parser.add_argument('--add_x_pixel', dest='add_pix_x', type=int, default=0, help='Fine tune output image resolution, add pixels to width.')
parser.add_argument('--add_y_pixel', dest='add_pix_y', type=int, default=0, help='Fine tune output image resolution, add pixels to height.')
parser.add_argument('-f', '--format', dest='out_format', type=str, default='png', help='File type of the output image.')
parser.add_argument('-t', '--tiff', dest='tiff_path', metavar='FILE', default=None, help='TIFF file for pixel resolution.')
parser.add_argument('--pixel_res', dest='pixel_res', type=float,
                    default=0.04, help='Pixel resolution. Defaults to 0.04 meters.')
parser.add_argument('--extra_shapefiles', dest='extra_shps', default=[],
                    nargs='*',
                    help='A list of paths of shapefiles that will be tiled together.')
parser.set_defaults(func=shpToBinaryImg)
args = parser.parse_args()
args.func(**vars(args))

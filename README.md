# chop

Non-exhaustive list of modules to install prior to use (these will install other dependencies that are also needed):
rasterio
pillow
tqdm
geopandas
matplotlib

Script Information:
General Information:
These scripts were developed to tile TIFF images so they are easily consumed by most Neural Networks, while creating respective grayscale masks for each tile
Various other uses can come from this
To see the parameters of each script use terminal and call:
python script_name.py -h
If available the scripts will use several CPUs to speed up the processing
 
shp_to_binary.py
Takes in ‘.shp’ files and processes them into various tiled gray scale images
The gray scale masks are saved with the coordinates of the corners of the tile, in the coordinate system used by the shapefile
The order of input matters!
The ‘-shp’ parameter is the shapefile that will be placed in the front-most layer
Any shapefiles inputted through ‘- -extrashapefiles’  will be placed in layers that correspond to their input order
‘-shp rectangle.shp --extra_shapefiles circle.shp triangle.shp’

![Mask Image]
(https://github.com/terrasense/chop/blob/master/mask_example.PNG)

i.e. shapes in the rectangle.shp file will occlude those in the extra shapefiles

bigtiff_to_tiled_img.py 
Transforms a TIFF image into tiled images of a provided format (compatible with pillow) corresponding to the given shapefile masks coordinates, which should be specified in the name. 
The names of the masks should be xmin__ymin__xmax__ymax.FORMAT
An example name is:
3781939_83104__58811030_481949__3841939_15104__59811030_411449.jpg 
which would correspond to the bounding box vertices in:
(3781939.83104, 58811030.481949), (3841939.15104, 59811030.411449)
Important: the coordinates of the masks must overlap those of the TIFF image and they must both be in the same coordinate system.

end_to_end_tiler.py
Uses both bigtiff_to_tiled_img.py and shp_to_binary.py to create masks and corresponding tiles from shapefiles and TIFF images.
Takes in folders of shapefiles and TIFFs and cross checks for overlapping coordinates to tile them.

img_augmentor.py
Takes in images that hold a mask attached to its corresponding image.
Splits the image in half (horizontally) and flips both and rotates them at the provided angle saving an image pair every time.

from PIL import Image, ImageOps
import os, shutil, multiprocessing
from argparse import ArgumentParser
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = 1000000000

def rotatePair(pair, angle, out_dir, sub_dirs, img_name,
               img_format, count, width, height, label): 
    new_img = Image.new('RGB', (int(width/2), height))
    new_label = Image.new('RGB', (int(width/2), height))
    new_img.paste(pair[0].rotate(angle), (0,0))
    new_label.paste(pair[1].rotate(angle), (0,0))
    folder = np.random.choice([0,2], p=[.8, .2])
    new_img.save(os.path.join(out_dir,sub_dirs[folder],
                              img_name+'_'+ label + '_' + str(count)+img_format))
    new_label.save(os.path.join(out_dir,sub_dirs[folder+1],
                              img_name+'_'+ label + '_' + str(count)+img_format))
    new_img.close()
    new_label.close()
    del new_img, new_label

def imageMultiplier(img_path, out_dir, sub_dirs, angle):
    img = Image.open(img_path)
    width, height = img.size
    img_A = img.crop((0,0,width/2,height))
    img_B = img.crop((width/2,0, width,height))
    imgs = [img_A, img_B]
    img.close()
    del img_A
    del img_B
    del img
    mirrors = []
    for i in imgs:
        mirrors.append(ImageOps.mirror(i))
    pairs = {'original':imgs, 'mirrored':mirrors}
    del imgs
    del mirrors
    img_name, img_format = os.path.splitext(os.path.basename(img_path))
    num_cores = multiprocessing.cpu_count()
    for label, pair in pairs.items():
        Parallel(n_jobs=num_cores)(delayed(rotatePair)(pair, angle, out_dir,
                                                       sub_dirs, img_name,
                                                       img_format, count,
                                                       width, height, label)
                                   for count, angle in enumerate(range(0, 360,
                                                                       angle)))

def masterDataAugmentor(in_dir, out_dir, make_subs, angle, **kwargs):
    img_paths = os.listdir(in_dir)
    if (out_dir == ''):
        out_dir = in_dir
    if (make_subs):
        sub_dirs=['train_img', 'train_label', 'test_img', 'test_label']
        for name in sub_dirs:
            subdir = os.path.join(out_dir, name)
            if (not os.path.exists(subdir)):
                os.mkdir(subdir)
                print('Created new dir: ', subdir)
            else:
                print('Directory already exists.', subdir)
    else:
        sub_dirs = []
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(imageMultiplier)
                               (os.path.join(in_dir, img_path),
                                out_dir, sub_dirs, angle)
                               for img_path in tqdm(img_paths))
    print('Total train images in folder: ',
          len(os.listdir(os.path.join(out_dir,sub_dirs[0]))))
    print('Total test images in folder: ',
          len(os.listdir(os.path.join(out_dir,sub_dirs[2]))))


parser = ArgumentParser(description='Splits image pairs, flips, rotates and '
                        + 'pastes them together.')
parser.add_argument('-i', '--in_dir', dest='in_dir', required=True,
                    help='Path to the original images.')
parser.add_argument('-o', '--out_dir', dest='out_dir', default='',
                    help='Path to where the images should be saved.')
parser.add_argument('--create_subfolders', dest='make_subs', default=True,
                    help='Whether to create a train, val and test subfolder.')
parser.add_argument('--angle', dest='angle', default=90, type=int,
                    help='Angle used to rotate image with.')
parser.set_defaults(func=masterDataAugmentor)
args = parser.parse_args()
args.func(**vars(args))

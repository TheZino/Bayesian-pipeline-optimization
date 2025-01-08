import glob
import os
from os.path import join

import cv2
import exiftool
import ipdb
import numpy as np
import skimage.io as io
from skimage.transform import resize as skimage_resize

expected_landscape_img_width = 2128
expected_landscape_img_height = 1424

expected_landscape_img_width = 600
expected_landscape_img_height = 400

mod = 'LOL'
downsample = 5  # 16
# LOL -> 5
# SID ->16
# downsample = 0 # use 0 or 1 for no further downsampling

# datasets_root = '..'
datasets_root = '/mnt/Datasets/Camera_pipe'

if mod == 'Sony':
    in_dir = join(datasets_root, 'Sony/gt/')
    if downsample > 1:
        out_dir = join(datasets_root, 'Sony_png_small/long/')
    else:
        out_dir = join(datasets_root, 'Sony_png/long/')
    ext = '.png'

elif mod == 'Fuji':
    in_dir = join(datasets_root, 'Fuji/gt/')
    if downsample > 1:
        out_dir = join(datasets_root, 'Fuji_png_small/long/')
    else:
        out_dir = join(datasets_root, 'Fuji_png/long/')
    ext = '.png'
elif mod == 'LOL':
    in_dir = join(datasets_root, 'LOLdataset/test/high')
    out_dir = join(datasets_root, 'LOLdataset/test_small/high/')

    ext = '.png'

if downsample > 1:
    expected_landscape_img_height = int(
        expected_landscape_img_height / downsample)
    expected_landscape_img_width = int(
        expected_landscape_img_width / downsample)


if not os.path.exists(out_dir):
    os.makedirs(out_dir)

ii = 0

data = glob.glob(join(in_dir, '*' + ext))

for file in data:
    ii += 1

    print('\t\t', end='\r')
    print('{} / {}'.format(ii, len(data)), end='\r')

    name = os.path.basename(file)
    tar_image = cv2.imread(file, cv2.IMREAD_UNCHANGED)

    # Skimage
    tar_image = skimage_resize(tar_image, [
                               expected_landscape_img_height, expected_landscape_img_width], preserve_range=True, anti_aliasing=True)
    if mod != 'LOL':
        tar_image = (tar_image / (2**16 - 1) * 255).astype(np.uint8)

    cv2.imwrite(join(out_dir, name.split('_')[0] + '.png'), tar_image)

    # # Torch
    # tar_image = torchvision.transforms.ToTensor()(tar_image.astype(np.float16)/(2**16 -1))
    # tar_image = torch.nn.Upsample(size=[expected_landscape_img_height, expected_landscape_img_width], mode='bilinear', align_corners=True)(tar_image[None,...])[0]
    # tar_image = (tar_image.numpy().transpose((1,2,0))*255).astype(np.uint8)
    # cv2.imwrite(join(out_dir, name.split('_')[0] + '.png'), tar_image)

    # # Pillow
    # tar_image = (tar_image.astype(np.float64) / (2**16 - 1) * 255).astype(np.uint8)
    # tar_image = Image.fromarray(tar_image[:,:,::-1]).resize([expected_landscape_img_width, expected_landscape_img_height], Image.Resampling.BILINEAR)
    # tar_image.save(join(out_dir, name.split('_')[0] + '.png'))

    # # OpenCV
    # tar_image = cv2.resize(tar_image, [expected_landscape_img_width, expected_landscape_img_height])
    # tar_image = (tar_image.astype(np.float64) / (2**16 - 1) * 255).astype(np.uint8)
    # cv2.imwrite(join(out_dir, name.split('_')[0]), tar_image)

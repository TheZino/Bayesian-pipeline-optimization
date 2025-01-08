import glob
import json
import os
from os.path import join

import cv2
import exiftool
import ipdb
import numpy as np
import rawpy
import skimage.io as io
from skimage.transform import resize as skimage_resize
from rawpy import ColorSpace, FBDDNoiseReductionMode, HighlightMode

expected_landscape_img_width = 2128
expected_landscape_img_height = 1424


def simple_demosaic(img, cfa_pattern):
    raw_colors = np.asarray(cfa_pattern).reshape((2, 2))
    demosaiced_image = np.zeros((img.shape[0] // 2, img.shape[1] // 2, 3))
    for i in range(2):
        for j in range(2):
            ch = raw_colors[i, j]
            if ch == 1:
                demosaiced_image[:, :, ch] += img[i::2, j::2] / 2
            else:
                demosaiced_image[:, :, ch] = img[i::2, j::2]
    return demosaiced_image.astype(np.uint16)


if __name__ == "__main__":

    mod = 'Sony'
    downsample = 16
    # downsample = 0 # use 0 or 1 for no further downsampling

    datasets_root = '..'
    # datasets_root = '/mnt/Datasets/Camera_pipe'

    if mod == 'Sony':
        in_dir = join(datasets_root, 'Sony/short/')
        if downsample > 1:
            out_dir = join(datasets_root, 'Sony_png_small/short/')
        else:
            out_dir = join(datasets_root, 'Sony_png/short/')
        ext = '.ARW'
        # Sony
        xyz_cam1 = [6263, -1964, 102,
                    -4375, 12428, 2171, -660, 1559, 7146]
    else:
        in_dir = join(datasets_root, 'Fuji/short/')
        if downsample > 1:
            out_dir = join(datasets_root, 'Fuji_png_small/short/')
        else:
            out_dir = join(datasets_root, 'Fuji_png/short/')
        ext = '.RAF'
        # Fujifilm X-T2
        xyz_cam1 = [11434, -4948, -1210, -3746, 12042, 1903, -666, 1479, 5235]

    if downsample > 1:
        expected_landscape_img_height = int(expected_landscape_img_height / downsample)
        expected_landscape_img_width = int(expected_landscape_img_width / downsample)

    xyz_cam2 = [5838, -1430, -246,
                -3497, 11477, 2297, -748, 1885, 5778]

    # Sony_cam_xyz = np.array(Sony_cam_xyz)/10000.0

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ii = 0
    data = glob.glob(in_dir + '/*' + ext)
    data.sort()
    for path in data:
        ii += 1
        print('\t\t', end='\r')
        print('{} / {}'.format(ii, len(data)), end='\r')

        # reading raw data for png generation
        raw = rawpy.imread(path)

        params = rawpy.Params(half_size=True,
                              four_color_rgb=False,
                              dcb_iterations=0,
                              dcb_enhance=False,
                              fbdd_noise_reduction=FBDDNoiseReductionMode.Off,
                              noise_thr=None,
                              median_filter_passes=0,
                              use_camera_wb=False,
                              use_auto_wb=False,
                              user_wb=[1.0, 1.0, 1.0, 1.0],
                              output_color=ColorSpace.raw,
                              output_bps=16,
                              user_flip=None,
                              user_black=None,
                              user_sat=None,
                              no_auto_bright=True,
                              auto_bright_thr=None,
                              adjust_maximum_thr=0.0,
                              bright=1.0,
                              highlight_mode=HighlightMode.Ignore,
                              exp_shift=None,
                              exp_preserve_highlights=1.0,
                              no_auto_scale=True,
                              gamma=[1.0, 1.0],
                              chromatic_aberration=None,
                              bad_pixels_path=None)

        # reading metadata with exiftool
        with exiftool.ExifToolHelper() as et:
            metadata = et.get_metadata(path)[0]

        # raw image generation
        name = os.path.basename(path)  # .split('/')[-1].split('.')[0]

        name = name.replace(ext, '')

        raw_image = raw.raw_image

        # metadata
        camera_name = metadata['EXIF:Make'] + metadata['EXIF:Model']
        if mod == 'Sony':
            black_level = metadata['MakerNotes:BlackLevel']
            cfa_pattern = metadata['EXIF:CFAPattern2']
            white_level = metadata['MakerNotes:WhiteLevel']
        else:
            black_level = '1022 1022 1022 1022'
            cfa_pattern = '2 0 1 2'
            white_level = None

    #     color_matrix_1 = metadata['MakerNotes:ColorMatrix']
    #     noise_profile only Canon
        orientation = metadata['EXIF:Orientation']

        meta = {}

        bl = []
        for val in black_level.split(' '):
            bl.append({"Fraction": [int(val), 1]})

        cfa_pattern = cfa_pattern.split(' ')
        for i in range(len(cfa_pattern)):
            cfa_pattern[i] = int(cfa_pattern[i])

        color_matrix1 = []
        for i in range(len(xyz_cam1)):
            color_matrix1.append({"Fraction": [xyz_cam1[i], 10000]})
        color_matrix2 = []
        for i in range(len(xyz_cam2)):
            color_matrix2.append({"Fraction": [xyz_cam2[i], 10000]})

        if bl is not None:
            meta["black_level"] = bl
        if camera_name is not None:
            meta["camera_name"] = camera_name
        if cfa_pattern is not None:
            meta["cfa_pattern"] = cfa_pattern
        if color_matrix1 is not None:
            meta["color_matrix_1"] = color_matrix1
        if color_matrix2 is not None:
            meta["color_matrix_2"] = color_matrix2
        # meta["linearization_table"] = None
        # meta["noise_profile"] = None
        if orientation is not None:
            meta["orientation"] = orientation

        if white_level:
            meta["white_level"] = int(white_level.split(' ')[0])

        with open(out_dir + '/' + name + '.json', 'w') as fp:
            json.dump(meta, fp,  indent=4)

        # dem_image = simple_demosaic(raw_image, cfa_pattern)
        # dem_image = dem_image[:, :-16, :]
        dem_image = raw.postprocess(params)

        # Skimage
        dem_image = skimage_resize(dem_image, [expected_landscape_img_height, expected_landscape_img_width], preserve_range=True, anti_aliasing=True)
        dem_image = dem_image.astype(np.uint16)
        cv2.imwrite(out_dir + '/' + name + '.png', dem_image)

        # # OpenCV
        # dem_image = cv2.cvtColor(dem_image, cv2.COLOR_RGB2BGR)
        # dem_image = cv2.resize(dem_image, [expected_landscape_img_width, expected_landscape_img_height], cv2.INTER_LINEAR)
        # cv2.imwrite(out_dir + '/' + name + '.png', dem_image)

        # # Torch
        # dem_image = torchvision.transforms.ToTensor()(dem_image.astype(np.float16)/(2**16 -1))
        # dem_image = torch.nn.Upsample(size=[expected_landscape_img_height, expected_landscape_img_width], mode='bilinear', align_corners=True)(dem_image[None,...])[0]
        # dem_image = (dem_image.numpy().transpose((1,2,0))*(2**16 -1)).astype(np.uint16)
        # cv2.imwrite(out_dir + '/' + name + '.png', dem_image)

        # # Pillow
        # dem_image = Image.fromarray(dem_image).resize([expected_landscape_img_width, expected_landscape_img_height], Image.Resampling.BILINEAR)
        # dem_image.save(out_dir + '/' + name + '.png')

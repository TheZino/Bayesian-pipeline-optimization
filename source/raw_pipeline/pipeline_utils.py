"""
Camera pipeline utilities.
"""

import os
from fractions import Fraction

import cv2
import exifread
import numpy as np
import rawpy
import skimage.restoration as skr  # import denoise_bilateral
import torch
# from exifread import Ratio
from exifread.utils import Ratio
from PIL import Image, ImageOps
from scipy.io import loadmat
from skimage.transform import resize as skimage_resize

from .utils.exif_utils import get_tag_values_from_ifds, parse_exif
from .utils.fs import perform_flash, perform_storm


def get_visible_raw_image(image_path):
    raw_image = rawpy.imread(image_path).raw_image_visible.copy()
    # raw_image = rawpy.imread(image_path).raw_image.copy()
    return raw_image


def get_image_tags(image_path):
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)
    return tags


def get_image_ifds(image_path):
    ifds = parse_exif(image_path, verbose=False)
    return ifds


def get_metadata(image_path):
    metadata = {}
    tags = get_image_tags(image_path)
    ifds = get_image_ifds(image_path)
    metadata['linearization_table'] = get_linearization_table(tags, ifds)
    metadata['black_level'] = get_black_level(tags, ifds)
    metadata['white_level'] = get_white_level(tags, ifds)
    metadata['cfa_pattern'] = get_cfa_pattern(tags, ifds)
    metadata['as_shot_neutral'] = get_as_shot_neutral(tags, ifds)
    color_matrix_1, color_matrix_2 = get_color_matrices(tags, ifds)
    metadata['color_matrix_1'] = color_matrix_1
    metadata['color_matrix_2'] = color_matrix_2
    metadata['orientation'] = get_orientation(tags, ifds)
    # isn't used
    metadata['noise_profile'] = get_noise_profile(tags, ifds)
    # ...
    # fall back to default values, if necessary
    if metadata['black_level'] is None:
        metadata['black_level'] = 0
        print("Black level is None; using 0.")
    if metadata['white_level'] is None:
        metadata['white_level'] = 2 ** 16
        print("White level is None; using 2 ** 16.")
    if metadata['cfa_pattern'] is None:
        metadata['cfa_pattern'] = [0, 1, 1, 2]
        print("CFAPattern is None; using [0, 1, 1, 2] (RGGB)")
    if metadata['as_shot_neutral'] is None:
        metadata['as_shot_neutral'] = [1, 1, 1]
        print("AsShotNeutral is None; using [1, 1, 1]")
    if metadata['color_matrix_1'] is None:
        metadata['color_matrix_1'] = [1] * 9
        print("ColorMatrix1 is None; using [1, 1, 1, 1, 1, 1, 1, 1, 1]")
    if metadata['color_matrix_2'] is None:
        metadata['color_matrix_2'] = [1] * 9
        print("ColorMatrix2 is None; using [1, 1, 1, 1, 1, 1, 1, 1, 1]")
    if metadata['orientation'] is None:
        metadata['orientation'] = 0
        print("Orientation is None; using 0.")
    # ...
    return metadata


def get_linearization_table(tags, ifds):
    possible_keys = ['Image Tag 0xC618', 'Image Tag 50712',
                     'LinearizationTable', 'Image LinearizationTable']
    return get_values(tags, possible_keys)


def get_black_level(tags, ifds):
    possible_keys = ['Image Tag 0xC61A', 'Image Tag 50714',
                     'BlackLevel', 'Image BlackLevel']
    vals = get_values(tags, possible_keys)
    if vals is None:
        # print("Black level not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(50714, ifds)
    return vals


def get_white_level(tags, ifds):
    possible_keys = ['Image Tag 0xC61D', 'Image Tag 50717',
                     'WhiteLevel', 'Image WhiteLevel']
    vals = get_values(tags, possible_keys)
    if vals is None:
        # print("White level not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(50717, ifds)
    return vals


def get_cfa_pattern(tags, ifds):
    possible_keys = ['CFAPattern', 'Image CFAPattern']
    vals = get_values(tags, possible_keys)
    if vals is None:
        # print("CFAPattern not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(33422, ifds)
    return vals


def get_as_shot_neutral(tags, ifds):
    possible_keys = ['Image Tag 0xC628', 'Image Tag 50728',
                     'AsShotNeutral', 'Image AsShotNeutral']
    return get_values(tags, possible_keys)


def get_color_matrices(tags, ifds):
    possible_keys_1 = ['Image Tag 0xC621', 'Image Tag 50721',
                       'ColorMatrix1', 'Image ColorMatrix1']
    color_matrix_1 = get_values(tags, possible_keys_1)
    possible_keys_2 = ['Image Tag 0xC622', 'Image Tag 50722',
                       'ColorMatrix2', 'Image ColorMatrix2']
    color_matrix_2 = get_values(tags, possible_keys_2)
    # print(f'Color matrix 1:{color_matrix_1}')
    # print(f'Color matrix 2:{color_matrix_2}')
    # print(np.sum(np.abs(np.array(color_matrix_1) - np.array(color_matrix_2))))
    return color_matrix_1, color_matrix_2


def get_orientation(tags, ifds):
    possible_tags = ['Orientation', 'Image Orientation']
    return get_values(tags, possible_tags)


def get_noise_profile(tags, ifds):
    possible_keys = ['Image Tag 0xC761', 'Image Tag 51041',
                     'NoiseProfile', 'Image NoiseProfile']
    vals = get_values(tags, possible_keys)
    if vals is None:
        # print("Noise profile not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(51041, ifds)
    return vals


def get_values(tags, possible_keys):
    values = None
    for key in possible_keys:
        if key in tags.keys():
            values = tags[key].values
    return values


def ratios2floats(ratios):
    floats = []
    for ratio in ratios:
        floats.append(float(ratio.num) / ratio.den)
    return floats


def fractions2floats(fractions):
    floats = []
    for fraction in fractions:
        floats.append(float(fraction.numerator) / fraction.denominator)
    return floats


_RGB_TO_YCBCR = np.array([[0.257, 0.504, 0.098],
                          [-0.148, -0.291, 0.439],
                          [0.439, -0.368, -0.071]])

_YCBCR_OFF = np.array([0.063, 0.502, 0.502])


def _mul(coeffs, image):

    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    r0 = np.repeat(r[:, :, np.newaxis], 3, 2) * coeffs[:, 0]
    r1 = np.repeat(g[:, :, np.newaxis], 3, 2) * coeffs[:, 1]
    r2 = np.repeat(b[:, :, np.newaxis], 3, 2) * coeffs[:, 2]

    return r0 + r1 + r2
    # return np.einsum("dc,ijc->dij", (coeffs, image))


def rgb2ycbcr(rgb):
    """sRGB to YCbCr conversion."""
    clip_rgb = False
    if clip_rgb:
        rgb = np.clip(rgb, 0, 1)
    return _mul(_RGB_TO_YCBCR, rgb) + _YCBCR_OFF


def ycbcr2rgb(rgb):
    """YCbCr to sRGB conversion."""
    clip_rgb = False
    rgb = _mul(np.linalg.inv(_RGB_TO_YCBCR), rgb - _YCBCR_OFF)
    if clip_rgb:
        rgb = np.clip(rgb, 0, 1)
    return rgb

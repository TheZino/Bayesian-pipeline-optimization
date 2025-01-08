from fractions import Fraction

import numpy as np
import torch
from exifread.utils import Ratio
from raw_pipeline.modules.utils.colors import hsv2rgb, rgb2hsv
from raw_pipeline.pipeline_utils import fractions2floats, ratios2floats


def normalize(raw_image, metadata, params):
    '''
        Image normalization.
        Given a raw image and the black and white level points,
        normalize the image with respect to these two values.

        output:
        - normalized_image in range [0, inf]
    '''

    v = torch.mean(raw_image, dim=1)

    black_level = 0
    white_level = torch.quantile(v.flatten(1), 0.9, dim=1)

    if black_level < white_level:

        if type(black_level) is list and len(black_level) == 1:
            black_level = float(black_level[0])
        if type(white_level) is list and len(white_level) == 1:
            white_level = float(white_level[0])
        black_level_mask = black_level
        if type(black_level) is list and len(black_level) == 4:
            if type(black_level[0]) is Ratio:
                black_level = ratios2floats(black_level)
            if type(black_level[0]) is Fraction:
                black_level = fractions2floats(black_level)
            black_level_mask = np.zeros(raw_image.shape)
            idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
            step2 = 2
            for i, idx in enumerate(idx2by2):
                black_level_mask[idx[0]::step2, idx[1]::step2] = black_level[i]

        normalized_image = raw_image - black_level_mask
        # if some values were smaller than black level
        normalized_image[normalized_image < 0] = 0
        normalized_image = normalized_image / \
            (white_level - black_level_mask)
    else:

        normalized_image = raw_image

    return normalized_image.clip(0, 1)


def normalize_dynamic(raw_image, metadata, params):
    '''
        Image normalization.
        Given a raw image and the black and white level points,
        normalize the image with respect to these two values.

        output:
        - normalized_image in range [0, 1]
    '''

    v = torch.mean(raw_image, dim=1)
    qB = torch.quantile(v.flatten(1), params['norm_Pb'], dim=1)
    black_level = params['norm_Wb'] * qB + params['norm_Bb']

    qW = torch.quantile(v.flatten(1), params['norm_Pw'], dim=1)
    white_level = params['norm_Ww'] * qW + params['norm_Bw']

    # Write dynamic variable to file
    if '_debug' in params and params['_debug']:
        with open(params['_debug_dir'] + 'normalization.csv', 'a') as f:
            for b, w, qb, qw in zip(black_level, white_level, qB, qW):
                # f.write('{:.10f}, {:.10f}\n'.format(b.item(), w.item()))
                f.write('{:.10f}, {:.10f}, {:.10f}, {:.10f}\n'.format(
                    b.item(), w.item(), qb.item(), qw.item()))

    # check for the inverted high-low bound
    black_level = torch.where(black_level > white_level, 0, black_level)
    white_level = torch.where(black_level > white_level, 1, white_level)

    bb, cc, hh, ww = raw_image.shape
    normalized_image = raw_image - \
        black_level.view(-1, 1, 1, 1).repeat(1, cc, hh, ww)
    # if some values were smaller than black level
    normalized_image[normalized_image < 0] = 0
    normalized_image = normalized_image / \
        (white_level - black_level).view(-1, 1, 1, 1).repeat(1, cc, hh, ww)

    return normalized_image.clip(0, 1)


def constrast_correction(image, metadata, params):
    '''
    TAKES TWO IMAGES NOT ONLY ONE
    '''
    sigma_p = 0.01

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255

    im_ycbcr = rgb2ycbcr(enhanced_image)
    or_ycbcr = rgb2ycbcr(input_image)

    y_new = im_ycbcr[:, :, 0]
    cb_new = im_ycbcr[:, :, 1]
    cr_new = im_ycbcr[:, :, 2]

    y = or_ycbcr[:, :, 0]
    cb = or_ycbcr[:, :, 1]
    cr = or_ycbcr[:, :, 2]

    # dark pixels percentage
    mask = np.logical_and(y < (35 / 255), (((cb - 0.5) * 2 +
                                            (cr - 0.5) * 2) / 2) < (20 / 255))

    dark_pixels = mask.flatten().sum()

    if dark_pixels > 0:

        ipixelCount, _ = np.histogram(y.flatten(), 256, range=(0, 1))
        cdf = np.cumsum(ipixelCount)
        idx = np.argmin(abs(cdf - (dark_pixels * 0.3)))
        b_input30 = idx

        ipixelCount, _ = np.histogram(y_new.flatten(), 256, range=(0, 1))
        cdf = np.cumsum(ipixelCount)
        idx = np.argmin(abs(cdf - (dark_pixels * 0.3)))
        b_output30 = idx

        bstr = (b_output30 - b_input30)
    else:

        bstr = np.floor(np.quantile(y_new.flatten(), 0.002) * 255)

    if bstr > 50:
        bstr = 50

    dark_bound = bstr / 255

    bright_b = np.floor(np.quantile(y_new.flatten(), 1 - 0.002) * 255)

    if (255 - bright_b) > 50:
        bright_b = 255 - 50

    bright_bound = bright_b / 255

    # y_new = (y_new - dark_bound) / (bright_bound - dark_bound)
    y_new = exposure.rescale_intensity(y_new, in_range=(
        y_new.min(), y_new.max()), out_range=(dark_bound, bright_bound))
    y_new = y_new.clip(0, 1)

    im_ycbcr[:, :, 0] = y_new
    im_new = ycbcr2rgb(im_ycbcr)

    im_new = im_new.clip(0, 1)

    return im_new


def saturation_correction(image, metadata, params):

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255

    im_ycbcr = rgb2ycbcr(enhanced_image)
    or_ycbcr = rgb2ycbcr(input_image)

    y_new = im_ycbcr[:, :, 0]
    cb_new = im_ycbcr[:, :, 1]
    cr_new = im_ycbcr[:, :, 2]

    y = or_ycbcr[:, :, 0]
    cb = or_ycbcr[:, :, 1]
    cr = or_ycbcr[:, :, 2]

    im_tmp = image.copy()

    r = im_tmp[:, :, 0]
    g = im_tmp[:, :, 1]
    b = im_tmp[:, :, 2]

    r_new = 0.5 * (((y_new / (y + 1e-40)) * (r + y)) + r - y)
    g_new = 0.5 * (((y_new / (y + 1e-40)) * (g + y)) + g - y)
    b_new = 0.5 * (((y_new / (y + 1e-40)) * (b + y)) + b - y)

    im_new[:, :, 0] = r_new
    im_new[:, :, 1] = g_new
    im_new[:, :, 2] = b_new

    im_new = im_new.clip(0, 1)

    return im_new


def saturation_scale(img, metadata, params):

    img_hsv = rgb2hsv(img)
    s = img_hsv[:, 1, :, :].clone()
    s *= params["sat_scale"]
    img_hsv[:, 1, :, :] = s

    return hsv2rgb(torch.clip(img_hsv, 0, 1))


def saturation_scale_dynamic(img, metadata, params):

    img_hsv = rgb2hsv(img)
    s = img_hsv[:, 1, :, :].clone()
    q = torch.quantile(s.flatten(1), params['sat_scale_P'], dim=1)
    s *= (params["sat_scale_W"] * q.view(-1, 1, 1).repeat([1,
                                                           s.shape[1], s.shape[2]]) + params["sat_scale_B"])
    img_hsv[:, 1, :, :] = s

    return hsv2rgb(torch.clip(img_hsv, 0, 1))

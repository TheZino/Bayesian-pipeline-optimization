import sys

import numpy as np
import torch

from .Grayness_Index import GPconstancy_GI


def white_balance(image, img_meta, params):

    thismodule = sys.modules[__name__]
    ie_algo = getattr(thismodule, params['illest_algo'])

    wb_params = ie_algo(image, img_meta, params)

    white_balanced = wb(image, wb_params)

    return white_balanced


def GI_white_balance(image, img_meta, params):

    tot_pixels = image.shape[2] * image.shape[3]
    # compute number of gray pixels
    n = params['GI_n']
    num_gray_pixels = int(np.floor(n * tot_pixels / 100))

    wb_params = GPconstancy_GI(image, num_gray_pixels, params['GI_th'])

    wb_params_norm = wb_params / \
        (wb_params.max(1)[0].view(-1, 1).repeat(1, 3) + 1e-14)

    white_balanced = wb(image, wb_params_norm)

    if white_balanced.isnan().any():
        import ipdb
        ipdb.set_trace()

    return white_balanced


def wb(demosaic_img, as_shot_neutral):

    as_shot_neutral += 1e-15
    white_balanced_image = demosaic_img
    white_balanced_image[:, 0, :, :] = demosaic_img[:,
                                                    0, :, :] / as_shot_neutral[:, 0][..., None, None]
    white_balanced_image[:, 1, :, :] = demosaic_img[:,
                                                    1, :, :] / as_shot_neutral[:, 1][..., None, None]
    white_balanced_image[:, 2, :, :] = demosaic_img[:,
                                                    2, :, :] / as_shot_neutral[:, 2][..., None, None]

    white_balanced_image = torch.clip(white_balanced_image, 0.0, 1.0)

    return white_balanced_image


def white_point(image, metadata, params):
    # TODO pytorch

    ie = np.max(image, axis=(0, 1))
    # ie = torch.max(image, axis=(2, 3))  # BCHW
    ie /= ie[1]

    return ie


def gray_world(image, metadata, params):
    # ie = torch.mean(image, axis=(1, 2)) # BWHC
    ie = torch.mean(image, axis=(2, 3))  # BCHW
    ie = ie / (ie[:, 1][..., None] + 1e-18)

    return ie


def shades_of_gray(image, metadata, params):
    # TODO pytorch

    p = 4.  # params[self.__classname__]['p']

    ie = np.mean(image**p, axis=(0, 1))**(1 / p)
    # ie = torch.mean(image**p, axis=(2, 3))**(1 / p)
    ie /= ie[1]

    return ie


def improved_white_point(image, metadata, params):
    # TODO pytorch

    samples_count = 20  # params[self.__classname__]['samples_count']
    samples_size = 20  # params[self.__classname__]['samples_count']

    rows, cols = image.shape[:2]
    data = np.reshape(image, (rows * cols, 3))
    maxima = np.zeros((samples_count, 3))
    for i in range(samples_count):
        maxima[i, :] = np.max(data[np.random.randint(
            low=0, high=rows * cols, size=(samples_size)), :], axis=0)
    ie = np.mean(maxima, axis=0)
    ie /= ie[1]

    return ie


def grayness_index(image, metadata, params):
    # TODO pytorch

    n_graypixels = 0.1  # params[self.__classname__]['n_graypixels']

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255

    tot_pixels = image.shape[0] * image.shape[1]
    num_gray_pixels = int(np.floor(self.n_graypixels * tot_pixels / 100))
    lumTriplet = GPconstancy_GI(image.copy(), num_gray_pixels, 10**(-4))
    lumTriplet /= lumTriplet.max()

    return lumTriplet

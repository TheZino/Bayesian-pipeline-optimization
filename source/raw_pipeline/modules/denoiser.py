from math import ceil

import cv2
import numpy as np
import pytorch_colors as ptcl
from bm3d import BM3DProfile, bm3d_rgb
from scipy.ndimage.filters import gaussian_filter
from skimage.restoration import denoise_nl_means, estimate_sigma

from .torch_nlm import nlm2d
from .utils.colors import *


def nlm_denoise(x, metadata, params):

    btch, _, _, _ = x.shape
    l_w = params['nlm_l_w']  # 4.5
    ch_w = params['nlm_ch_w']  # 20

    for ii in range(btch):

        yuv = rgb2yuv(x[ii, :, :, :].unsqueeze(0))[0]

        patch_kw = dict(patch_size=5,
                        patch_distance=6
                        )
        sigma_est = estimate_sigma(yuv[0, :, :].numpy())
        den_y = denoise_nl_means(yuv[0, :, :], h=l_w * sigma_est, fast_mode=True,
                                 **patch_kw)
        den_y = torch.tensor(den_y)

        patch_kw = dict(patch_size=5,
                        patch_distance=6,
                        channel_axis=-1
                        )

        sigma_est = np.mean(estimate_sigma(
            yuv[1:2, :, :].numpy().transpose([1, 2, 0]), channel_axis=-1))
        den_uv = denoise_nl_means(yuv[1:, :, :].numpy().transpose([1, 2, 0]), h=ch_w * sigma_est, fast_mode=True,
                                  **patch_kw)
        den_uv = torch.tensor(den_uv.transpose([2, 0, 1]))

        tmp = torch.concat([den_y.unsqueeze(0), den_uv], 0)
        x[ii, :, :, :] = yuv2rgb(tmp.unsqueeze(0))[0]

    return x


class denoise_bilateral():
    '''
    Bilateral Filter denoising using skimage implementation.
    '''

    def __init__(self, params):

        self.sigma_color = None
        self.sigma_spatial = 0.01
        self.d = max(5, 2 * ceil(3 * self.sigma_spatial) + 1)

    def __call__(self, image, metadata):

        out = cv2.bilateralFilter(
            image.astype(np.float32), self.d, self.sigma_color, self.sigma_spatial)

        return out


class denoise_bm3d():
    '''
    BM3D denoising with light area masks
    '''

    def __init__(self, params):

        self.with_mask = True
        self.sigma = 0.02
        self.profile = BM3DProfile()

        self.mask_sigmap = 0.005
        self.mask_scale = 0.6

    def __call__(self, image, metadata):

        noise_profile = metadata['noise_profile'][0][0]
        scale = False

        if np.issubdtype(image.dtype, np.uint8):
            scale = True
            image = image.astype(np.float32) / 255

        h, w, _ = image.shape

        if noise_profile != '':
            if noise_profile > 2e-4:
                sigma = 0.08
            elif noise_profile < 2e-4 and noise_profile > 1e-4:
                sigma = 0.06
            else:
                sigma = 0.02
        else:
            sigma = self.sigm

        im_ycbcr = rgb2ycbcr(image)
        y_noise = im_ycbcr[:, :, 0]

        profile = self.profile

        denoised_image = bm3d_rgb(image, sigma, profile)

        if self.with_mask:

            im_ycbcr = rgb2ycbcr(image)
            denoised_ycbcr = rgb2ycbcr(denoised_image)
            y_out = denoised_ycbcr[:, :, 0]

            y = y_noise
            y = (y - y.min()) / (y.max() - y.min())
            h, w = y.shape

            sigma = np.sqrt(h ** 2 + w ** 2) * self.mask_sigmap
            mask = gaussian_filter(y, sigma)

            mask = np.tile(mask, [3, 1, 1]).transpose(1, 2, 0)
            mask = (mask - mask.min()) / (mask.max() - mask.min())

            mask *= self.mask_scale

            out_ycbcr = denoised_ycbcr * (1 - mask) + im_ycbcr * mask
            out = ycbcr2rgb(out_ycbcr)

        out = out.clip(0, 1)
        denoised_image = denoised_image.clip(0, 1)

        if scale:
            out = (out * 255).astype(np.uint8)
            out = out.clip(0, 255)
            denoised_image = (denoised_image * 255).astype(np.uint8)
            denoised_image = denoised_image.clip(0, 255)

        import ipdb
        ipdb.set_trace()

        return out

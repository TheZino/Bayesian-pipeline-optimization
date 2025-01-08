
import numpy as np
import pytorch_colors as ptcl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.nn.modules.utils import _pair, _quadruple


def denoise_filter(img, metadata, params):

    if params['denoise_algo'] == 'median_filter':

        den_filter = MedianPool2d(kernel_size=3)
        # den_filter = MedianPool2d(kernel_size=params['median_kernel'])

    elif params['denoise_algo'] == 'gaussian_blur':

        # YUV
        # den_filter = GaussianBlur(params)

        # all
        den_filter = T.GaussianBlur(
            kernel_size=(3, 3),  # int(params['gauss_kernel']),
            sigma=2)  # params['gauss_sigma'])

    elif params['denoise_algo'] == 'smoothing_filter':

        den_filter = smoothing_filter

    else:

        def den_filter(x): return x

    out = den_filter(img)
    return out.clip(0, 1)


def sharpening(img, metadata, params):

    if params['sharpening_algo'] == 'unsharp_filter':
        out = unsharp_filter(img)
        # out = unsharp_filter_ch(img, params)
    elif params['sharpening_algo'] == 'edge_filter':
        out = edge_filter(img)
        # out = edge_filter_ch(img, params)
    elif params['sharpening_algo'] == 'detail_filter':
        out = detail_filter(img)
        # out = detail_filter_ch(img, params)
    else:
        out = img

    return out


def unsharp_filter(img, metadata, params):

    ch = img.shape[1]

    weights = torch.tensor([[-0.125, -0.125, -0.125],
                            [-0.125, 2.000, -0.125],
                            [-0.125, -0.125, -0.125]])
    weights = weights.view(1, 1, 3, 3)

    output = img.clone()
    for i in range(ch):
        output[:, i, :, :] = F.conv2d(
            img[:, i, :, :].unsqueeze(1), weights, padding='same').squeeze()

    return torch.clamp(output, 0, 1)


def unsharp_filter_ch(img, params):

    weights = torch.tensor([[-0.125, -0.125, -0.125],
                            [-0.125, 2.000, -0.125],
                            [-0.125, -0.125, -0.125]])
    weights = weights.view(1, 1, 3, 3)

    if params['sharpening_ch'] == 1:

        ch = img.shape[1]
        output = img.clone()
        for i in range(ch):
            output[:, i, :, :] = F.conv2d(
                img[:, i, :, :].unsqueeze(1), weights, padding='same').squeeze()

    else:
        im_yuv = ptcl.rgb_to_yuv(img)
        output_yuv = im_yuv.clone()

        output_yuv[:, 0, :, :] = F.conv2d(
            img[:, 0, :, :].unsqueeze(1), weights, padding='same').squeeze()

        output = ptcl.yuv_to_rgb(output_yuv)

    return torch.clamp(output, 0, 1)


def unsharp_filter_param(img, params):

    im_yuv = ptcl.rgb_to_yuv(img)
    output_yuv = im_yuv.clone()

    im_y = output_yuv[:, 0, :, :]

    gauss = T.GaussianBlur(
        kernel_size=int(params['unsh_kernel']),
        sigma=params['unsh_sigma'])

    g_out = gauss(im_y)
    diff = im_y - g_out
    im_y += diff

    output_yuv[:, 0] = im_y
    output = ptcl.yuv_to_rgb(output_yuv)

    return torch.clamp(output, 0, 1)


def unsharp_matlab(img, metadata, params):

    im_lab = ptcl.rgb_to_lab(img)
    output_lab = im_lab.clone()

    filtRadius = np.ceil(params['unsh_radius'] * 2)
    filtSize = int(filtRadius * 2 + 1)

    gaussFilt = matlab_style_gauss2D((filtSize, filtSize), filtRadius)

    filtRadius = int(filtRadius)
    sharpFilt = torch.zeros(filtSize, filtSize)
    sharpFilt[filtRadius + 1, filtRadius + 1] = 1
    sharpFilt = sharpFilt - gaussFilt

    sharpFilt = params['unsh_amount'] * sharpFilt
    # Add 1 to the center element of sharpFilt effectively add a unit
    # impulse kernel to sharpFilt.
    sharpFilt[filtRadius + 1, filtRadius +
              1] = sharpFilt[filtRadius + 1, filtRadius + 1] + 1
    sharpFilt = sharpFilt.view(1, 1, filtSize, filtSize)

    output_lab[:, 0, :, :] = torch.clamp(F.conv2d(
        im_lab[:, 0, :, :].unsqueeze(1), sharpFilt, padding='same').squeeze(), 0, 100)

    output = ptcl.lab_to_rgb(output_lab)

    return torch.clamp(output, 0, 1)


def edge_filter(img):

    ch = img.shape[1]

    weights = torch.tensor([[-0.5, -0.5, -0.5],
                            [-0.5, 5., -0.5],
                            [-0.5, -0.5, -0.5]])
    weights = weights.view(1, 1, 3, 3)

    output = img.clone()
    for i in range(ch):
        output[:, i, :, :] = F.conv2d(
            img[:, i, :, :].unsqueeze(1), weights, padding='same').squeeze()

    return torch.clamp(output, 0, 1)


def edge_filter_ch(img, params):

    weights = torch.tensor([[-0.5, -0.5, -0.5],
                            [-0.5, 5., -0.5],
                            [-0.5, -0.5, -0.5]])
    weights = weights.view(1, 1, 3, 3)

    if params['sharpening_ch'] == 1:

        ch = img.shape[1]
        output = img.clone()
        for i in range(ch):
            output[:, i, :, :] = F.conv2d(
                img[:, i, :, :].unsqueeze(1), weights, padding='same').squeeze()

    else:
        im_yuv = ptcl.rgb_to_yuv(img)
        output_yuv = im_yuv.clone()

        output_yuv[:, 0, :, :] = F.conv2d(
            img[:, 0, :, :].unsqueeze(1), weights, padding='same').squeeze()

        output = ptcl.yuv_to_rgb(output_yuv)

    return torch.clamp(output, 0, 1)


def detail_filter(img):

    weights = torch.tensor([[0, -0.17, 0],
                            [-0.17, 1.67, -0.17],
                            [0, -0.17, 0]])
    weights = weights.view(1, 1, 3, 3)

    ch = img.shape[1]
    output = img.clone()
    for i in range(ch):
        output[:, i, :, :] = F.conv2d(
            img[:, i, :, :].unsqueeze(1), weights, padding='same').squeeze()

    return torch.clamp(output, 0, 1)


def detail_filter_ch(img, params):

    weights = torch.tensor([[0, -0.17, 0],
                            [-0.17, 1.67, -0.17],
                            [0, -0.17, 0]])
    weights = weights.view(1, 1, 3, 3)

    if params['sharpening_ch'] == 1:

        ch = img.shape[1]
        output = img.clone()
        for i in range(ch):
            output[:, i, :, :] = F.conv2d(
                img[:, i, :, :].unsqueeze(1), weights, padding='same').squeeze()

    else:
        im_yuv = ptcl.rgb_to_yuv(img)
        output_yuv = im_yuv.clone()

        output_yuv[:, 0, :, :] = F.conv2d(
            img[:, 0, :, :].unsqueeze(1), weights, padding='same').squeeze()

        output = ptcl.yuv_to_rgb(output_yuv)

    return torch.clamp(output, 0, 1)


def smoothing_filter(img):

    ch = img.shape[1]

    weights = torch.tensor([[0.077, 0.077, 0.077],
                            [0.077, 0.385, 0.077],
                            [0.077, 0.077, 0.077]])
    weights = weights.view(1, 1, 3, 3)

    output = img.clone()
    for i in range(ch):
        output[:, i, :, :] = F.conv2d(
            img[:, i, :, :].unsqueeze(1), weights, padding='same').squeeze()

    return torch.clamp(output, 0, 1)


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(
            3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


class GaussianBlur():

    def __init__(self, params):

        self.GB_y = T.GaussianBlur(
            # kernel_size=3,
            kernel_size=int(params['gauss_kernel_y']),
            sigma=params['gauss_sigma_y'])

        self.GB_uv = T.GaussianBlur(
            # kernel_size=3,
            kernel_size=int(params['gauss_kernel_uv']),
            sigma=params['gauss_sigma_uv'])

    def __call__(self, image):
        im_yuv = ptcl.rgb_to_yuv(image)

        out = im_yuv.clone()

        out[:, 0] = self.GB_y(im_yuv[:, 0, :, :])
        out[:, 1:3] = self.GB_uv(im_yuv[:, 1:3, :, :])

        return ptcl.yuv_to_rgb(out)


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return torch.from_numpy(h.astype('float32'))


def sharpening_night(img, metadata, params):

    sigma = params['sharpening_sigma']  # 2  # params['sharpening_sigma']
    scale = params['sharpening_scale']  # 1

    kernel = round(sigma)*8+1

    try:
        GBlur = T.GaussianBlur(
            kernel_size=(kernel, kernel),
            sigma=sigma)
    except:
        import ipdb
        ipdb.set_trace()

    blurred = GBlur(img)
    unsharp_image = img + scale * (img - blurred)

    return unsharp_image

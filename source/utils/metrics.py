import cv2
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from utils.pytorch_colors import rgb2hsv


class MSE():

    def __init__(self):
        pass

    def __call__(self, A, B):

        return torch.mean((A - B)**2)


class MLum():

    def __init__(self):
        pass

    def __call__(self, A, B):

        A = A / 255
        B = B / 255

        A_y = rgb2hsv(A)[:, 2, :, :]
        B_y = rgb2hsv(B)[:, 2, :, :]

        return torch.abs(torch.mean(A_y) - torch.mean(B_y))


class histogramSimilarity():

    def __init__(self):
        pass

    def __call__(self, A, B):

        # A_h = torch.tensor([])
        [tmp, _] = torch.histogram(A[:, 0, ...], 256)
        tmp = tmp / torch.sqrt(torch.sum(tmp**2.0))
        A_h = tmp[None]  # torch.cat((A_h[None], tmp[None]),0)
        [tmp, _] = torch.histogram(A[:, 1, ...], 256)
        tmp = tmp / torch.sqrt(torch.sum(tmp**2.0))
        A_h = torch.cat((A_h, tmp[None]), 0)
        [tmp, _] = torch.histogram(A[:, 2, ...], 256)
        tmp = tmp / torch.sqrt(torch.sum(tmp**2.0))
        A_h = torch.cat((A_h, tmp[None]), 0)

        # B_h = torch.tensor([])
        [tmp, _] = torch.histogram(B[:, 0, ...], 256)
        tmp = tmp / torch.sqrt(torch.sum(tmp**2.0))
        B_h = tmp[None]  # torch.cat((B_h[None], tmp[None]),0)
        [tmp, _] = torch.histogram(B[:, 1, ...], 256)
        tmp = tmp / torch.sqrt(torch.sum(tmp**2.0))
        B_h = torch.cat((B_h, tmp[None]), 0)
        [tmp, _] = torch.histogram(B[:, 2, ...], 256)
        tmp = tmp / torch.sqrt(torch.sum(tmp**2.0))
        B_h = torch.cat((B_h, tmp[None]), 0)

        return torch.mean((A_h - B_h)**2)


def torch_PSNR(A, B):

    mse = torch.mean(((A - B) ** 2), [1, 2, 3])
    torch.where(mse != 0, mse, 1e-9)
    return torch.mean(20 * torch.log10(1 / torch.sqrt(mse)))

def calc_psnr(im1, im2):
    '''
    im1 and im2 range [0,255]
    '''
    # im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    # im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im1_y = cv2.cvtColor(im1, cv2.COLOR_RGB2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_RGB2YCR_CB)[:, :, 0]
    return peak_signal_noise_ratio(im1_y, im2_y)


def calc_ssim(im1, im2):
    '''
    im1 and im2 range [0,255]
    '''
    # im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    # im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im1_y = cv2.cvtColor(im1, cv2.COLOR_RGB2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_RGB2YCR_CB)[:, :, 0]
    return structural_similarity(im1_y, im2_y)

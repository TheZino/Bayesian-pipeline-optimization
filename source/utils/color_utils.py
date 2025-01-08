import numpy as np

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

import numpy as np
import PIL.Image as Image
import torch
from skimage.transform import resize as skimage_resize


def to_uint8(srgb, img_meta, params):
    return (srgb * 255).clamp(0, 255)

# RESIZE IMAGE


def resize(img, img_meta, params):

    if params['out_landscape_width'] is None or params['out_landscape_height'] is None:
        return img
    return _resize_using_pytorch(img, params['out_landscape_width'], params['out_landscape_height'])


def _resize_using_skimage(img, width=1296, height=864):
    out_shape = (height, width) + img.shape[2:]
    if img.shape == out_shape:
        return img
    out_img = skimage_resize(
        img, out_shape, preserve_range=True, anti_aliasing=True)
    out_img = out_img.astype(np.uint8)
    return out_img


def _resize_using_pil(img, width=1296, height=864):
    img_pil = Image.fromarray(img)
    out_size = (width, height)
    if img_pil.size == out_size:
        return img
    out_img = img_pil.resize(out_size, Image.ANTIALIAS)
    out_img = np.array(out_img)
    return out_img


def _resize_using_pytorch(img, width=1296, height=864):
    out_size = (height, width)
    # if img.shape[1:3] == out_size: # BWHC
    if img.shape[2:4] == out_size:  # BCHW
        return img
    out_img = torch.nn.Upsample(
        size=out_size, mode='bilinear', align_corners=True)(img)
    return out_img


# FIX ORIENTATION

def orientation(image, img_meta, params):

    # TODO pytorch

    # 1 = Horizontal(normal)
    # 2 = Mirror horizontal
    # 3 = Rotate 180
    # 4 = Mirror vertical
    # 5 = Mirror horizontal and rotate 270 CW
    # 6 = Rotate 90 CW
    # 7 = Mirror horizontal and rotate 90 CW
    # 8 = Rotate 270 CW

    orientation = img_meta['orientation']

    return image

    if type(orientation) is list:
        orientation = orientation[0]

    if orientation == 1:
        pass
    elif orientation == 2:
        image = cv2.flip(image, 0)
    elif orientation == 3:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 4:
        image = cv2.flip(image, 1)
    elif orientation == 5:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == 6:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 7:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 8:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image

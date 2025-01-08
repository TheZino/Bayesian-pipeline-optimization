from fractions import Fraction

import torch
from raw_pipeline.pipeline_utils import fractions2floats


def xyz_transform(demosaiced_image, img_meta, params):

    color_matrix_1 = img_meta['color_matrix_1']

    if isinstance(color_matrix_1[0], Fraction):
        color_matrix_1 = fractions2floats(color_matrix_1)

    temp = torch.cat([i[None] for i in color_matrix_1]).permute((1, 0)).float()
    xyz2cam1 = temp.reshape((temp.shape[0], 3, 3))

    xyz2cam1 = xyz2cam1 / torch.sum(xyz2cam1, axis=2, keepdims=True)

    cam2xyz1 = torch.linalg.inv(xyz2cam1)

    xyz_image = demosaiced_image.clone()
    # BCWH
    for b in range(demosaiced_image.shape[0]):
        xyz_image[b, ...] = torch.einsum(
            'jc,ckl', cam2xyz1[0], demosaiced_image[b])

    xyz_image = torch.clip(xyz_image, 0.0, 1.0)

    return xyz_image


def srgb_transform(xyz_image, img_meta, params):

    xyz2srgb = torch.Tensor([[2.0413690, -0.5649464, -0.3446944],
                             [-0.9692660, 1.8760108, 0.0415560],
                             [0.0134474, -0.1183897, 1.0154096]])

    # normalize rows
    xyz2srgb = xyz2srgb / torch.sum(xyz2srgb, axis=-1, keepdims=True)

    srgb_image = xyz2srgb[None, :, :, None, None] * \
        xyz_image[:, None, :, :, :]  # BCHW

    srgb_image = torch.sum(srgb_image, axis=2)  # BCHW
    srgb_image = torch.clip(srgb_image, 0.0, 1.0)
    return srgb_image

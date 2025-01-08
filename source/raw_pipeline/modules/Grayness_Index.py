import cv2
import numpy as np
import torch
import torch.nn.functional as F


def DerivGauss(im, sigma=0.5):

    btc, _, _ = im.shape

    gaussian_die_off = 0.000001
    var = sigma ** 2
    # compute filter width
    width = None
    for i in range(1, 51):
        if np.exp(-(i ** 2) / (2 * var)) > gaussian_die_off:
            width = i
    if width is None:
        width = 1

    # create filter (derivative of Gaussian filter)
    x = np.arange(-width, width + 1)
    y = np.arange(-width, width + 1)
    coordinates = np.meshgrid(x, y)
    x = coordinates[0]
    y = coordinates[1]
    derivate_gaussian_2D = -x * \
        np.exp(-(x * x + y * y) / (2 * var)) / (var * np.pi)

    # apply filter and return magnitude
    # ay = cv2.filter2D(im, -1, np.transpose(derivate_gaussian_2D))

    weights = torch.tensor(derivate_gaussian_2D).view(
        1, 1, derivate_gaussian_2D.shape[0], derivate_gaussian_2D.shape[1])

    im = im.type('torch.DoubleTensor')
    im_pad = F.pad(im, (width, width, width, width),
                   mode='reflect').unsqueeze(1)
    ax = F.conv2d(im_pad, weights, padding='valid',
                  bias=None)
    ay = F.conv2d(im_pad, weights.permute(
        [0, 1, 3, 2]), padding='valid', bias=None)

    magnitude = torch.sqrt((ax ** 2) + (ay ** 2))

    return magnitude.type('torch.FloatTensor')


def GPconstancy_GI(im, gray_pixels, delta_th=1e-4):

    btc, _, hh, ww = im.shape
    # mask saturated pixels and mask very dark pixels
    mask = torch.logical_or(torch.max(im, axis=1)[0] >= 0.95,
                            torch.sum(im, axis=1)[0] <= 0.0315)

    # remove noise with mean filter
    # mean_kernel = np.ones((7, 7), np.float32) / 7**2
    # im = cv2.filter2D(im, -1, mean_kernel)
    # decompose rgb values
    r = im[:, 0, :, :]
    g = im[:, 1, :, :]
    b = im[:, 2, :, :]

    # mask 0 elements
    # mask = np.logical_or.reduce((mask, r == 0, g == 0, b == 0))

    mask = torch.logical_or(torch.logical_or(
        torch.logical_or(mask, r == 0), g == 0), b == 0)

    # replace 0 values with machine epsilon
    eps = np.finfo(np.float32).eps
    r = r.clamp(min=eps)
    g = g.clamp(min=eps)
    b = b.clamp(min=eps)
    norm = r + g + b  # piccolo errore di approssimazione

    # mask low contrast pixels
    delta_r = DerivGauss(r).squeeze()
    delta_g = DerivGauss(g).squeeze()
    delta_b = DerivGauss(b).squeeze()
    mask = torch.logical_or(mask,
                            torch.logical_and(torch.logical_and(
                                delta_r <= delta_th, delta_g <= delta_th), delta_b <= delta_th))

    # compute colors in log domain, only red and blue
    log_r = torch.log(r) - torch.log(norm)
    log_b = torch.log(b) - torch.log(norm)

    # mask low contrast pixels in the log domain
    delta_log_r = DerivGauss(log_r).squeeze()
    delta_log_b = DerivGauss(log_b).squeeze()
    mask = torch.logical_or(torch.logical_or(
        mask, delta_log_r == torch.inf), delta_log_b == torch.inf)
    mask = mask.bool()

    # normalize each channel in log domain
    # errore di approssimazione anche qui
    data = torch.cat(
        [delta_log_r.view(btc, -1, 1), delta_log_b.view(btc, -1, 1)], axis=2)

    mink_norm = 2
    norm2_data = torch.sum(data ** mink_norm, axis=2) ** (1 / mink_norm)

    map_uniquelight = norm2_data.view(
        btc, hh, ww)

    # make masked pixels to max value
    map_uniquelight = torch.where(
        mask,
        torch.max(torch.max(map_uniquelight, 1)[0], 1)[0].view(-1, 1, 1).repeat(1, hh, ww), map_uniquelight)

    # denoise
    # map_uniquelight = cv2.filtxer2D(map_uniquelight, -1, mean_kernel)

    # filter using map_uniquelight
    gray_index_unique = map_uniquelight
    sort_unique = torch.sort(gray_index_unique.view(btc, -1))[0]
    gindex_unique = torch.full(gray_index_unique.shape, False, dtype=bool)
    gindex_unique[gray_index_unique <= sort_unique[:,
                                                   gray_pixels - 1].view(-1, 1, 1).repeat(1, hh, ww)] = True

    # choosen_pixels = im[gindex_unique.unsqueeze(1).repeat(1, 3, 1, 1)]
    # choosen_pixels = choosen_pixels.view(btc, 3, -1).transpose(2, 1)

    # mean = torch.mean(choosen_pixels, axis=1)

    # result = mean / torch.linalg.norm(mean, dim=1)

    result = torch.zeros([btc, 3])
    for i in range(btc):
        choosen_pixels = im[i][gindex_unique[i].unsqueeze(0).repeat(3, 1, 1)]
        choosen_pixels = choosen_pixels.view(3, -1).transpose(1, 0)
        mean = torch.mean(choosen_pixels, axis=0)
        result[i] = mean / (torch.linalg.norm(mean, dim=0) + 1e-12)

    return result


if __name__ == "__main__":

    # read image and convert to 0 1
    im_path = 'example.png'
    im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
    im = np.float64(im) / 255
    im_t = torch.DoubleTensor(im).permute([2, 0, 1]).unsqueeze(0)

    im_path = 'example_2.png'
    im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
    im = np.float64(im) / 255
    im_t2 = torch.DoubleTensor(im).permute([2, 0, 1]).unsqueeze(0)

    inputt = torch.cat([im_t, im_t2], 0)

    tot_pixels = im.shape[0] * im.shape[1]
    # compute number of gray pixels
    n = 0.1  # 0.01%
    num_gray_pixels = int(np.floor(n * tot_pixels / 100))
    # compute global illuminant values
    lumTriplet = GPconstancy_GI(inputt, num_gray_pixels, 10**(-4))
    # show results and angular error w.r.t gt
    print(lumTriplet)
    # print(np.arccos(np.dot(lumTriplet, gt)) * 180 / np.pi)

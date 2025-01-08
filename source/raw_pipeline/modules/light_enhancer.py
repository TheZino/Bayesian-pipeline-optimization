
import numpy as np
import pytorch_colors as ptcl
# import raw_pipeline.modules.utils.colors as cl
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import resize as tresize

# from raw_pipeline.modules.utils.bicubic_spline import compute_spline
from .utils.colors import *


def brightness(image, metadata, params):
    channel = None
    mod = image.clone()
    if channel is not None:
        mod[:, channel, :, :] = torch.clamp(
            mod[:, channel, :, :] + bright, 0, 1)
    else:
        v = torch.mean(image, dim=1)
        q = torch.quantile(v.flatten(1), params['bright_P'], dim=1)
        bright = params['bright_W'] * q + params['bright_B']
        _, cc, hh, ww = mod.shape
        mod = torch.clamp(mod + bright.view(-1, 1, 1,
                                            1).repeat(1, cc, hh, ww), 0, 1)
    return mod


def spline(img, metadata, params):

    x = [0, 0.2, 0.5, 0.7, 1]
    y = [0, params['spline_1'], params['spline_2'], params['spline_3'], 1]

    bspline = compute_spline(x, y)

    return bspline(img)


def gamma_correction(img, metadata, params):
    return img ** params['gamma_exp']


def gamma_correction_dynamic(img, metadata, params):
    v = torch.mean(img, dim=1)
    q = torch.quantile(v.flatten(1), params['gamma_exp_P'], dim=1)

    gamma = q * params['gamma_exp_W'] + params['gamma_exp_B']

    # Write dynamic variable to file
    if '_debug' in params and params['_debug']:
        with open(params['_debug_dir'] + 'gamma_correction_dynamic.csv', 'a') as f:
            for p, qt in zip(gamma, q):
                f.write('{:.10f}, {:.10f}\n'.format(p.item(), qt.item()))
                # f.write('%.10f\n' % p.item())

    out = img ** gamma[..., None, None, None]

    if np.isinf(out).any():
        import ipdb
        ipdb.set_trace()

    return out


def rescale_intensity(image, in_range, out_range):
    imin, imax = in_range
    omin, omax = out_range

    image = image.clamp(imin, imax)

    if imin != imax:
        image = (image - imin) / (imax - imin)
        return (image * (omax - omin) + omin)
    else:
        return torch.clip(image, omin, omax)


def contrast_saturation_fix(enhanced_image, input_image, mode="LCC", n_bits=8):

    # for numpy images
    # Contrast enhancement step

    im_ycbcr = rgb2ycbcr(enhanced_image)
    or_ycbcr = rgb2ycbcr(input_image)

    y_new = im_ycbcr[:, 0, :, :]
    cb_new = im_ycbcr[:, 1, :, :]
    cr_new = im_ycbcr[:, 2, :, :]

    y = or_ycbcr[:, 0, :, :]
    cb = or_ycbcr[:, 1, :, :]
    cr = or_ycbcr[:, 2, :, :]

    # dark pixels percentage
    mask = torch.logical_and(y < (35 / 255), (((cb - 0.5) * 2 +
                                               (cr - 0.5) * 2) / 2) < (20 / 255))

    dark_pixels = mask.flatten().sum()

    if dark_pixels > 0:

        ipixelCount, _ = torch.histogram(y.flatten(), 256, range=(0, 1))

        cdf = torch.cumsum(ipixelCount, 0)
        idx = torch.argmin(abs(cdf - (dark_pixels * 0.3)))
        b_input30 = idx

        ipixelCount, _ = torch.histogram(y_new.flatten(), 256, range=(0, 1))
        cdf = torch.cumsum(ipixelCount, 0)
        idx = torch.argmin(abs(cdf - (dark_pixels * 0.3)))
        b_output30 = idx

        bstr = (b_output30 - b_input30)
    else:

        bstr = torch.floor(torch.quantile(y_new.flatten(), 0.002) * 255)

    if bstr > 50:
        bstr = 50

    dark_bound = bstr / 255

    tmp = tresize(y_new, [y_new.shape[1]//2, y_new.shape[2]//2])
    bright_b = torch.floor(torch.quantile(tmp.flatten(), 1 - 0.002) * 255)

    if (255 - bright_b) > 50:
        bright_b = 255 - 50

    bright_bound = bright_b / 255

    # y_new = (y_new - dark_bound) / (bright_bound - dark_bound)
    y_new = rescale_intensity(y_new, in_range=(
        y_new.min(), y_new.max()), out_range=(dark_bound, bright_bound))
    y_new = y_new.clip(0, 1)

    im_ycbcr[:, 0, :, :] = y_new
    im_new = ycbcr2rgb(im_ycbcr)

    im_new = im_new.clip(0, 1)

    # Saturation

    im_tmp = input_image

    r = im_tmp[:, 0, :, :]
    g = im_tmp[:, 1, :, :]
    b = im_tmp[:, 2, :, :]

    r_new = 0.5 * (((y_new / (y + 1e-40)) * (r + y)) + r - y)
    g_new = 0.5 * (((y_new / (y + 1e-40)) * (g + y)) + g - y)
    b_new = 0.5 * (((y_new / (y + 1e-40)) * (b + y)) + b - y)

    im_new[:, 0, :, :] = r_new
    im_new[:, 1, :, :] = g_new
    im_new[:, 2, :, :] = b_new

    # im_new = im_new.clip(0, 1)

    return im_new


def local_contrast_correction(image, metadata, params):

    b, c, h, w = image.shape

    sigma = params['LCC_sigma_p']

    ycbcr = ptcl.rgb_to_ycbcr(image)

    y = (ycbcr[:, 0, :, :] - 16) / 219
    cb = ycbcr[:, 1, :, :]
    cr = ycbcr[:, 2, :, :]

    truncate = 4.0
    kernel_size = int(truncate * sigma)

    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1

    gaussfilt = T.GaussianBlur(
        kernel_size=kernel_size,    # int(params['gauss_kernel']),
        sigma=sigma)                # params['gauss_sigma'])

    blurred_y = gaussfilt(y)

    mask = 1 - blurred_y

    mean_intensity = torch.mean(y, [1, 2])

    alpha_lower = torch.log(mean_intensity) / np.log(0.5)
    alpha_upper = np.log(0.5) / torch.log(mean_intensity)

    condition = mean_intensity < 0.5
    condition = condition.view(-1, 1, 1).repeat(1, h, w)

    alpha = torch.where(condition,
                        alpha_lower.view(-1, 1, 1).repeat(1,
                                                          mask.shape[1], mask.shape[2]),
                        alpha_upper.view(-1, 1, 1).repeat(1,
                                                          mask.shape[1], mask.shape[2]))

    gamma = alpha ** ((0.5 - mask) / 0.5)

    new_y = y ** gamma
    new_y = new_y * 219 + 16

    new_ycbcr = torch.stack([new_y, cb, cr], 1)
    im_rgb = ptcl.ycbcr_to_rgb(new_ycbcr)
    # im_rgb = torch.clamp(im_rgb, 0, 1)

    im_rgb = contrast_saturation_fix(im_rgb, image)

    return im_rgb.clip(0)


def global_mean_contrast(x, metadata, params):
    '''
    NOTA: from Cotogni et al.
    We considered the variants with β ∈ {0.8, 2.0}
    and applied to all the channels, or just one.
    we can use another parameter to determine this behaviour.

    GMC_chsel: categorical {0, 1, 2, 3} if 3 = on all channels
    '''

    beta = params['GMC_beta']
    channel_selection = 3  # params['GMC_chsel']

    x_mean = torch.mean(x, axis=[2, 3])

    if channel_selection == 3:
        # scale all channels
        out = x_mean.unsqueeze(2).unsqueeze(2).repeat([1, 1, x.shape[2], x.shape[3]]) + \
            beta * \
            (x -
             x_mean.unsqueeze(2).unsqueeze(2).repeat([1, 1, x.shape[2], x.shape[3]]))

        if np.isnan(out).any():
            import ipdb
            ipdb.set_trace()
    else:
        # scale only the selected channel
        out = x.clone()
        out[:, channel_selection] = x_mean[:, channel_selection].unsqueeze(1).unsqueeze(1).repeat([1, x.shape[2], x.shape[3]]) \
            + beta \
            * (x[:, channel_selection] - x_mean[:, channel_selection].unsqueeze(1).unsqueeze(1).repeat([1, x.shape[2], x.shape[3]]))

    # out = torch.clamp(out, min=0, max=1)

    return out


def global_mean_contrast_perch(x, metadata, params):
    '''
    NOTA: from Cotogni et al.
    We considered the variants with β ∈ {0.8, 2.0}
    and applied to all the channels, or just one.
    we can use another parameter to determine this behaviour.
    '''

    betas = [params['GMC_beta_R'], params['GMC_beta_G'], params['GMC_beta_B']]

    x_mean = torch.mean(x, axis=[2, 3])

    # scale only the selected channel
    out = x.clone()

    for i in range(2):
        out[:, i] = x_mean[:, i].unsqueeze(1).unsqueeze(1).repeat([1, x.shape[2], x.shape[3]]) \
            + betas[i] \
            * (x[:, i] - x_mean[:, i].unsqueeze(1).unsqueeze(1).repeat([1, x.shape[2], x.shape[3]]))

    out = torch.clamp(out, min=0, max=1)

    return out


def global_mean_contrast_dynamic(x, metadata, params):
    '''
    NOTA: from Cotogni et al.
    We considered the variants with β ∈ {0.8, 2.0}
    and applied to all the channels, or just one.
    we can use another parameter to determine this behaviour.

    GMC_chsel: categorical {0, 1, 2, 3, 4} if 3 = on all channels 4 = skip module
    '''

    tmp = torch.mean(x, dim=1).flatten(1)
    v = torch.abs(tmp - torch.mean(tmp, dim=1).view(-1,
                                                    1).repeat(1, tmp.shape[1]))
    q = torch.quantile(v, params['GMC_beta_P'], dim=1)

    beta = (1 - q) * params['GMC_beta_W'] + params['GMC_beta_B']

    # beta = ((q - params['GMC_Im']) / (params['GMC_IM'] - params['GMC_Im']) )** params['GMC_exp'] * (params['GMC_OM'] - params['GMC_Om']) + params['GMC_Om']

    # Write dynamic variable to file
    if '_debug' in params and params['_debug']:
        with open(params['_debug_dir'] + 'global_mean_contrast_dynamic.csv', 'a') as f:
            for p, qt in zip(beta, q):
                f.write('{:.10f}, {:.10f}\n'.format(p.item(), qt.item()))
                # f.write('%.10f\n' % p.item())

    channel_selection = 3  # params['GMC_chsel']

    x_mean = torch.mean(x, axis=[2, 3])

    if channel_selection == 3:
        # scale all channels
        out = x_mean.view(x_mean.shape[0], x_mean.shape[1], 1, 1).repeat([1, 1, x.shape[2], x.shape[3]]) + \
            beta.view(-1, 1, 1, 1).repeat([1, x.shape[1], x.shape[2], x.shape[3]]) * \
            (x -
             x_mean.view(x_mean.shape[0], x_mean.shape[1], 1, 1).repeat([1, 1, x.shape[2], x.shape[3]]))

    else:
        # scale only the selected channel
        out = x.clone()
        out[:, channel_selection] = x_mean[:, channel_selection].view(-1, 1, 1).repeat([1, x.shape[2], x.shape[3]]) \
            + beta.view(-1, 1, 1).repeat([1, x.shape[2], x.shape[3]])  \
            * (x[:, channel_selection] - x_mean[:, channel_selection].view(-1, 1, 1).repeat([1, x.shape[2], x.shape[3]]))

    out = torch.clamp(out, min=0, max=1)

    return out


def bicubic_spline(img, metadata, params):

    xs = [0, 0.25, 0.5, 0.75, 1]
    ys = [0,
          params['CS_y1'],
          params['CS_y2'],
          params['CS_y3'],
          1
          ]

    spline = compute_spline(xs, ys)

    hsv = ptcl.rgb_to_hsv(img)
    x = hsv[:, 2, :, :]

    hsv[:, 2, :, :] = out
    out = hsv_to_rgb(hsv)

    out = torch.clamp(out, min=0, max=1)

    return out


def scurve(img, metadata, params):

    alpha = params['SCurve_alpha']  # [0, 1]
    lmbd = params['SCurve_lambda']  # [0, 1]

    channel_selection = 2  # params['SCurve_chsel']
    '''
    chsel:  - 0 Y from YCbCr
            - 1 V from HSV
            - 2 RGB
    '''
    # import ipdb; ipdb.set_trace()
    if channel_selection == 0:
        ycbcr = rgb2ycbcr(img)
        x = ycbcr[:, 0, :, :]
    elif channel_selection == 1:
        hsv = rgb2hsv(img)
        x = hsv[:, 2, :, :]
    else:
        x = img

    out = torch.where(x <= alpha,
                      x,  # alpha - alpha * (1 - x / alpha) ** lmbd,
                      alpha + (1 - alpha) * ((x - alpha) / (1 - alpha)).clip(min=0) ** lmbd)

    if channel_selection == 0:
        ycbcr[:, 0, :, :] = out
        out = ycbcr2rgb(ycbcr)
    elif channel_selection == 1:
        hsv[:, 2, :, :] = out
        out = hsv2rgb(hsv)

    # out = torch.clamp(out, min=0, max=1)

    return out


def scurve_dynamic(img, metadata, params):

    v = torch.mean(img, dim=1)
    q = torch.quantile(v.flatten(1), params['SCurve_alpha_P'], dim=1)
    s = torch.std(v.flatten(1), dim=1)

    alpha = q * params['SCurve_alpha_W'] + params['SCurve_alpha_B']
    lmbd = s * params['SCurve_lambda_W'] + params['SCurve_lambda_B']
    # Add dimensions for later broadcasting
    alpha = alpha[..., None, None]
    lmbd = lmbd[..., None, None]

    # Write dynamic variable to file
    if '_debug' in params and params['_debug']:
        with open(params['_debug_dir'] + 'scurve_dynamic.csv', 'a') as f:
            for (p1, p2, q1, q2) in zip(alpha, lmbd, q, s):
                # f.write('{:.10f},{:.10f}\n'.format(p1.item(), p2.item()))
                f.write('{:.10f},{:.10f},{:.10f},{:.10f}\n'.format(
                    p1.item(), p2.item(), q1.item(), q2.item()))

    channel_selection = params['SCurve_chsel']
    '''
    chsel:  - 0 Y from YCbCr
            - 1 V from HSV
            - 2 RGB
    '''
    # import ipdb; ipdb.set_trace()
    if channel_selection == 0:
        # ycbcr = ptcl.rgb_to_ycbcr(img)
        ycbcr = rgb2ycbcr(img)
        x = ycbcr[:, 0, :, :]
    elif channel_selection == 1:
        hsv = rgb2hsv(img)
        x = hsv[:, 2, :, :]
    else:
        x = img
        alpha = alpha[..., None]
        lmbd = lmbd[..., None]

    out = torch.where(x <= alpha,
                      alpha - alpha * (1 - x / alpha) ** lmbd,
                      alpha + (1 - alpha) * ((x - alpha) / (1 - alpha)) ** lmbd)

    if out.isnan().any().item():
        import ipdb
        ipdb.set_trace()

    if channel_selection == 0:
        ycbcr[:, 0, :, :] = out
        # out = ptcl.ycbcr_to_rgb(ycbcr)
        out = ycbcr2rgb(ycbcr)
    elif channel_selection == 1:
        hsv[:, 2, :, :] = out
        out = hsv2rgb(hsv)

    out = torch.clamp(out, min=0, max=1)

    return out


def imadjust(img, metadata, params):
    '''
    Python version of matlab imadjust
    '''
    b, cc, hh, ww = img.shape
    hsv = rgb2hsv(img.clip(0, 1))
    v = hsv[:, 2, :, :]
    v = tresize(v, [v.shape[1]//2, v.shape[2]//2])

    hi = torch.quantile(
        v.view(b, -1), params['Hist_max'], dim=1, interpolation='nearest')  # [0, 1]
    li = torch.quantile(
        v.view(b, -1), params['Hist_min'], dim=1, interpolation='nearest')  # [0, 1]

    hi = torch.where(hi < 0.7,
                     torch.quantile(v.view(b, -1), 0.995, dim=1,
                                    interpolation='nearest'),
                     hi)

    # mancano le casistiche hi = 1 e li = 0 !!!
    for i in range(b):
        if hi[i] == 1:
            v_tmp = v[i, :, :].flatten()
            v_tmp = v_tmp[v_tmp != 1]
            hi[i] = torch.quantile(v_tmp, 0.9995, interpolation='nearest')

        if li[i] == 0:
            v_tmp = v[i, :, :].flatten()
            v_tmp = v_tmp[v_tmp != 0]
            li[i] = torch.quantile(v_tmp, 0.0001, interpolation='nearest')

    # default matlab imasdjust values
    lo = torch.zeros(b)
    ho = torch.ones(b) * 0.9
    gamma = 1

    eps = 0

    if (hi == li).any():
        eps = 1e-10

    x = img

    li = li.view(-1, 1, 1, 1).repeat(1, cc, hh, ww)
    hi = hi.view(-1, 1, 1, 1).repeat(1, cc, hh, ww)
    lo = lo.view(-1, 1, 1, 1).repeat(1, cc, hh, ww)
    ho = ho.view(-1, 1, 1, 1).repeat(1, cc, hh, ww)

    # x = torch.clip(x, li, hi)
    out = ((x - li) / (hi - li + eps)) ** gamma
    out = out * (ho - lo) + lo

    if np.isnan(out).any():
        import ipdb
        ipdb.set_trace()

    return out


def histogram_stretch(img, meta, params):
    # return img
    low_perc = params['hist_low_perc']
    high_perc = params['hist_high_perc']

    # channel_selection = params['HS_chsel']
    '''
    chsel:  - 0 Y from YCbCr
            - 1 V from HSV
            - 2 RGB
    '''
    # if channel_selection == 0:
    #     ycbcr = cl.rgb2ycbcr(img)
    #     x = ycbcr[:, 0, :, :]
    # elif channel_selection == 1:
    #     hsv = cl.rgb2hsv(img)
    #     x = hsv[:, 2, :, :]
    # else:
    #     x = img
    x = img

    dark_bound = torch.quantile(x.flatten(1), low_perc)
    white_bound = torch.quantile(x.flatten(1), high_perc)

    if white_bound > dark_bound:
        x_new = (x - dark_bound) / (white_bound - dark_bound + 1e-14)
    else:
        x_new = x

    out = x_new
    # if channel_selection == 0:
    #     ycbcr[:, 0, :, :] = x_new
    #     out = cl.ycbcr2rgb(torch.clip(ycbcr, 0, 1))
    # elif channel_selection == 1:
    #     hsv[:, 2, :, :] = x_new
    #     out = cl.hsv2rgb(torch.clip(hsv, 0, 1))

    return out.clip(0, 1)


def histogram_stretch_dynamic(img, meta, params):

    channel_selection = params['HS_chsel']
    '''
    chsel:  - 0 Y from YCbCr
            - 1 V from HSV
            - 2 RGB
    '''

    if channel_selection == 0:
        ycbcr = cl.rgb2ycbcr(img)
        x = ycbcr[:, 0, :, :]
    elif channel_selection == 1:
        hsv = cl.rgb2hsv(img)
        x = hsv[:, 2, :, :]
    else:
        x = img

    dark_bound = torch.quantile(x.flatten(1),
                                params['HS_low_P'], dim=1) * params['HS_low_W'] + params['HS_low_B']
    dark_bound = dark_bound.unsqueeze(1).unsqueeze(
        1).repeat([1, x.shape[1], x.shape[2]])
    white_bound = torch.quantile(x.flatten(1),
                                 params['HS_high_P'], dim=1) * params['HS_high_W'] + params['HS_high_B']
    white_bound = white_bound.unsqueeze(1).unsqueeze(
        1).repeat([1, x.shape[1], x.shape[2]])

    x_new = (x - dark_bound) / (white_bound - dark_bound + 1e-14)

    if channel_selection == 0:
        ycbcr[:, 0, :, :] = x_new
        out = cl.ycbcr2rgb(torch.clip(ycbcr, 0, 1))
    elif channel_selection == 1:
        hsv[:, 2, :, :] = x_new
        out = cl.hsv2rgb(torch.clip(hsv, 0, 1))

    return out.clip(0, 1)


def black_stretch(img, meta, params):

    perc = params['bs_perc']
    im_hsv = rgb2hsv(img)
    v = im_hsv[:, :, 2]

    dark_bound = torch.quantile(v.flatten(1), perc, dim=1)

    v_new = (v - dark_bound) / (1 - dark_bound)

    im_hsv[:, :, 2] = v_new.clip(0, 1)

    out = hsv2rgb(im_hsv)

    return out.clip(0, 1)


def scurve_central(img, lmbd=1 / 1.4):

    b, c, h, w = img.shape
    x = img

    im_hsv = rgb2hsv(img.clip(0, 1))
    v = im_hsv[:, 2, :, :]
    v = tresize(v, [v.shape[1]//2, v.shape[2]//2])

    alpha1 = torch.quantile(v.view(b, -1), 0.2, dim=1, interpolation='nearest')
    alpha2 = torch.quantile(v.view(b, -1), 0.9, dim=1, interpolation='nearest')
    alpha1 = alpha1.view(b, 1, 1, 1).repeat(1, c, h, w)
    alpha2 = alpha2.view(b, 1, 1, 1).repeat(1, c, h, w)

    # print(alpha1)
    # print(alpha2)

    out = torch.where(x <= alpha1,
                      x,
                      torch.where(x >= alpha2,
                                  x,
                                  alpha1 + (alpha2 - alpha1) *
                                  ((x - alpha1) / (alpha2 - alpha1)
                                   ).clamp(min=0) ** lmbd
                                  )
                      )

    return out


def conditional_contrast_correction(img, meta, params):

    lambda_1 = params['ccc_lambda_1']  # (1 / 1.8)
    lambda_2 = params['ccc_lambda_2']  # (1 / 1.4)
    gamma = params['ccc_gamma']  # 1.6

    b, c, h, w = img.shape
    im_h = rgb2hsv(img)[:, 2, :, :]

    mean_hs = im_h.mean([1, 2])

    mask1 = (mean_hs < 0.2).view(b, 1, 1, 1).repeat(1, c, h, w)
    mask2 = (mean_hs < 0.25).view(b, 1, 1, 1).repeat(1, c, h, w)
    mask3 = (mean_hs > 0.4).view(b, 1, 1, 1).repeat(1, c, h, w)

    # su ogni immagine, dove la maschera ha dato 1 (quindi per forza un'immagine intera)
    # ritorno l'applicazione della funzione, altrimenti lascio come era

    out = img.clone()
    out = torch.where(mask1, scurve_central(img, lmbd=lambda_1), out)
    out = torch.where(mask2, scurve_central(img, lmbd=lambda_2), out)
    out = torch.where(mask3, img ** gamma, out)

    return out.clamp(0, 1)


if __name__ == "__main__":

    import skimage.io as io

    im = io.imread(
        '/home/zino/Datasets/Camera_pipe/Sony/Sony_png_small/test/long/10003.png')

    im_t = torch.Tensor(im.transpose([2, 0, 1])).unsqueeze(0) / 255

    out = conditional_contrast_correction(im_t, None, None)

    import ipdb
    ipdb.set_trace()

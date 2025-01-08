import bisect
from typing import List, Tuple

import torch


def compute_changes(x: List[float]) -> List[float]:
    return [x[i + 1] - x[i] for i in range(len(x) - 1)]


def create_tridiagonalmatrix(n: int, h: List[float]) -> Tuple[List[float], List[float], List[float]]:
    A = [h[i] / (h[i] + h[i + 1]) for i in range(n - 2)] + [0]
    B = [2] * n
    C = [0] + [h[i + 1] / (h[i] + h[i + 1]) for i in range(n - 2)]
    return A, B, C


def create_target(n: int, h: List[float], y: List[float]):
    return [0] + [6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]) / (h[i] + h[i - 1]) for i in range(1, n - 1)] + [0]


def solve_tridiagonalsystem(A: List[float], B: List[float], C: List[float], D: List[float]):
    c_p = C + [0]
    d_p = [0] * len(B)
    X = [0] * len(B)

    c_p[0] = C[0] / B[0]
    d_p[0] = D[0] / B[0]
    for i in range(1, len(B)):
        c_p[i] = c_p[i] / (B[i] - c_p[i - 1] * A[i - 1])
        d_p[i] = (D[i] - d_p[i - 1] * A[i - 1]) / \
            (B[i] - c_p[i - 1] * A[i - 1])

    X[-1] = d_p[-1]
    for i in range(len(B) - 2, -1, -1):
        X[i] = d_p[i] - c_p[i] * X[i + 1]

    return X


def compute_spline(x: List[float], y: List[float]):
    n = len(x)
    if n < 3:
        raise ValueError('Too short an array')
    if n != len(y):
        raise ValueError('Array lengths are different')

    h = compute_changes(x)
    if any(v < 0 for v in h):
        raise ValueError('X must be strictly increasing')

    A, B, C = create_tridiagonalmatrix(n, h)
    D = create_target(n, h, y)

    M = solve_tridiagonalsystem(A, B, C, D)

    coefficients = [[(M[i + 1] - M[i]) * h[i] * h[i] / 6, M[i] * h[i] * h[i] / 2,
                     (y[i + 1] - y[i] - (M[i + 1] + 2 * M[i]) * h[i] * h[i] / 6), y[i]] for i in range(n - 1)]

    def spline(val):

        # bisect for matrices
        idxs = val.clone()
        for ii, vv in enumerate(x):
            if ii + 1 < len(x):
                idxs = torch.where(x[ii] <= idxs,
                                   torch.where(x[ii + 1] > idxs,
                                               min(ii, n - 2), idxs),
                                   idxs)
            else:
                idxs = torch.where(x[ii] <= idxs, min(ii, n - 2), idxs)

        x_t = torch.tensor(x)
        x_t = x_t.view(
            1, -1, 1, 1).repeat([val.shape[0], 1, val.shape[2], val.shape[3]])
        h_t = torch.tensor(h)
        h_t = h_t.view(
            1, -1, 1, 1).repeat([val.shape[0], 1, val.shape[2], val.shape[3]])

        xs = torch.gather(x_t, 1, idxs.type(torch.LongTensor))
        hs = torch.gather(h_t, 1, idxs.type(torch.LongTensor))

        z = (val - xs) / hs

        c_t = torch.tensor(coefficients)

        c_t = c_t.view(
            1, 4, 1, 1, 4).repeat([val.shape[0], 1, val.shape[2], val.shape[3], 1])
        idxs_c = idxs.unsqueeze(-1).repeat(1, 1, 1, 1, 4)
        cs = torch.gather(c_t, 1, idxs_c.type(torch.LongTensor))

        out = (((cs[:, :, :, :, 0] * z) + cs[:, :, :, :, 1])
               * z + cs[:, :, :, :, 2]) * z + cs[:, :, :, :, 3]

        return out.clamp(0, 1)

        # bb, cc, hh, ww = val.shape
        # val = val.view(-1)
        # for vv in val:
        #     idx = min(bisect.bisect(x, vv) - 1, n - 2)
        #     z = (vv - x[idx]) / h[idx]
        #     C = coefficients[idx]
        #     vv = (((C[0] * z) + C[1]) * z + C[2]) * z + C[3]
        # val = val.view(bb, cc, hh, ww)
        # return val

    return spline


if __name__ == "__main__":
    import torch

    import skimage.io as io

    x = [0, 0.2, 0.5, 0.7, 1]
    y = [0, 0.8, 0.5, 0.7, 1]

    spline = compute_spline(x, y)

    y = spline(a)

"""
Dynamic pipeline executor
"""

import glob
import importlib
from os.path import basename, dirname, isfile, join

import matplotlib.pyplot as plt
import numpy as np
import pytorch_colors as ptcl
import torch
from torch import tensor

from .modules.illuminant_estimator import wb, white_balance
from .modules.transforms import *


class PipelineExecutor:
    """

    Pipeline executor class.

    This class can be used to successively execute the steps of some image 
    pipeline passed as list of functions.

    It is assumed that each operations of the pipeline has 2 parameters:
    raw_img : ndarray
        Array with images data.
    img_meta : Dict
        Some meta data of image.

    Also each such public method must return an image (ndarray) as the result of processing.
    """

    def __init__(self, op_sequence, params):
        """
        PipelineExecutor __init__ method.

        Parameters
        ----------
        op_sequence:
            List of strings with the names of the processing blocks 
            which compose the processing pipeline.
        params:
            dictionary containing the parameters : value pairs used by the blocks in the op_sequence
        """
        self.params = params

        modules_list = self.get_modules_list()

        self.op_sequence = []
        for op in op_sequence:
            check = True
            for mod in modules_list:
                mdl = importlib.import_module(mod)
                if op in dir(mdl):
                    self.op_sequence.append(getattr(mdl, op))
                    check = False
                    break
            if check:
                raise Exception(
                    "PipelineExecutor: operation {} not available".format(op))

    def get_modules_list(self):

        modules_pkg = 'raw_pipeline.modules'
        excluded_files = (
            '__init__.py',
            'colors.py'
            'colors.py'
        )
        modules_paths = glob.glob(join(dirname(__file__), 'modules', "*.py"))
        modules = []
        for f in modules_paths:
            if isfile(f):
                check = True
                for ex in excluded_files:
                    if f.endswith(ex):
                        check = False
                        break
                if check:
                    modules.append(modules_pkg + "." + basename(f)[:-3])

        return modules

    def preliminary_steps(self, x, metadata):

        self.params['illest_algo'] = "gray_world"

        x = white_balance(x, metadata, self.params)

        # import ipdb
        # ipdb.set_trace()
        # x = wb(x, tensor([]))

        metadata['color_matrix_1'] = [tensor([1.06835938], dtype=torch.float64),
                                      tensor([-0.29882812],
                                             dtype=torch.float64),
                                      tensor([-0.14257812],
                                             dtype=torch.float64),
                                      tensor([-0.43164062],
                                             dtype=torch.float64),
                                      tensor([1.35546875],
                                             dtype=torch.float64),
                                      tensor([0.05078125],
                                             dtype=torch.float64),
                                      tensor([-0.1015625],
                                             dtype=torch.float64),
                                      tensor([0.24414062],
                                             dtype=torch.float64),
                                      tensor([0.5859375], dtype=torch.float64)]

        xyz_image = xyz_transform(x, metadata, self.params)
        out = srgb_transform(xyz_image, metadata, self.params)

        return out

    def __call__(self, image, metadata, visualization=False):
        """`
        PipelineExecutor __call__ method.

        This method will sequentially execute the methods defined in the op_sequence.

        Returns
        -------
        ndarray
            Resulted processed raw image.
        """
        X = image
        X_prev = None

        if visualization:
            fig, axs = plt.subplots(
                2, len(self.op_sequence) + 1, figsize=(18, 6))

        X = self.preliminary_steps(X.clone(), metadata)

        for i, fun in enumerate(self.op_sequence):
            X_prev = X.clone()

            X = fun(X, metadata.copy(), self.params).clip(0)

            # print("Fun: {} \t Range: {} - {}".format(fun.__name__, X.min(), X.max()))
            # import ipdb
            # ipdb.set_trace()
            # import skimage.io as io
            # io.imsave('./dbg/'+str(i)+'_'+fun.__name__+'.png',
            #           (X[0].numpy().transpose([1, 2, 0]).clip(0, 1)*255).astype('uint8'))
            # import ipdb
            # ipdb.set_trace()
            # print("Debug -- running: " + fun.__name__)
            # if (X > 1).any() and X.dtype != torch.uint8:
            #     import ipdb
            #     ipdb.set_trace()

            # if (X > 1).any() or (X < 0).any():
            #     import ipdb
            #     ipdb.set_trace()
            if np.isnan(X).any():
                import ipdb
                ipdb.set_trace()
                raise Exception(
                    "Module {} returned image with Nan values!".format(fun.__name__))
            if np.isinf(X).any():
                raise Exception(
                    "Module {} returned image with inf values!".format(fun.__name__))

            if visualization:

                X_v = ptcl.rgb_to_hsv(X)[:, 2, :, :]
                X_s = ptcl.rgb_to_hsv(X)[:, 1, :, :]
                axs[0, i].hist(X_v[0].flatten(), 256, [0, 1], density=True)
                axs[0, i].axvline(X_v[0].flatten().mean(),
                                  color='r', linewidth=1)
                axs[0, i].set_title(fun.__name__)
                axs[1, i].hist(X_s[0].flatten(), 256, [0, 1], density=True)
                axs[1, i].axvline(X_s[0].flatten().mean(),
                                  color='r', linewidth=1)
                axs[1, i].set_title(fun.__name__)

        # import ipdb
        # ipdb.set_trace()
        if visualization:
            axs[0, -1].imshow(X.squeeze().numpy().transpose([1, 2, 0]))
            axs[0, -1].get_xaxis().set_visible(False)
            axs[0, -1].get_yaxis().set_visible(False)
            axs[1, -1].get_xaxis().set_visible(False)
            axs[1, -1].get_yaxis().set_visible(False)
            plt.tight_layout()
            plt.show()
            # import ipdb
            # ipdb.set_trace()

        return X

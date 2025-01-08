import glob
import os
from os.path import join
from pathlib import Path

import cv2
import torchvision.transforms as trfs
import torchvision.transforms.functional as TF
from skimage.transform import resize as skimage_resize
from torch.utils.data.dataset import Dataset
from utils import fraction_from_json, json_read


class SIDDataset_thumb(Dataset):
    def __init__(self, root='./data/test/', in_filter='*.png', gt_filter='*.png', transforms=None, mode='train', random_crop=False, crop_size=128):

        self.mode = mode

        self.transforms = transforms

        self.in_path = root + '/short/' + in_filter
        self.gt_path = root + '/long/'

        self.in_list = glob.glob(self.in_path)
        self.gt_list = glob.glob(self.gt_path + in_filter)
        self.in_list.sort()
        self.gt_list.sort()

        self.random_crop = random_crop
        self.crop_size = crop_size

        self.len = len(self.in_list)

        print('Total files with GT: {:d}.'.format(self.len))

    def __getitem__(self, index):
        # start = time.time()
        in_path = self.in_list[index]
        # gt_path = self.gt_list[index]

        # print(os.path.basename(in_path))
        name = os.path.basename(in_path).split('_')[0] + '.png'

        gt_path = join(self.gt_path, name)

        # With OpenCV
        try:
            raw_image = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)[
                :, :, ::-1]  # .astype('int16')
            raw_image = skimage_resize(
                raw_image, [900, 1200], preserve_range=True, anti_aliasing=True)
            tar_image = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)[
                :, :, ::-1]  # .astype('int16')
            tar_image = skimage_resize(
                tar_image, [84, 125], preserve_range=True, anti_aliasing=True)
        except:
            import ipdb
            ipdb.set_trace()

        raw_image = (raw_image.astype('float32') / (2.0**16 - 1))
        tar_image = tar_image.astype('float32') / 255

        if self.transforms is not None:
            raw_image = self.transforms(raw_image)
            tar_image = self.transforms(tar_image)

        if self.random_crop:

            i, j, h, w = trfs.RandomCrop.get_params(
                raw_image, output_size=(self.crop_size, self.crop_size))
            raw_image = TF.crop(raw_image, i, j, h, w)
            tar_image = TF.crop(tar_image, i, j, h, w)

        # parse metadata
        metadata = json_read(Path(in_path).with_suffix(
            '.json'), object_hook=fraction_from_json)
        # print("Loading time {}".format(time.time()-start))
        if self.mode == 'train':
            return raw_image, tar_image, metadata
        elif self.mode == 'test':
            return raw_image, tar_image, metadata, os.path.basename(in_path)

    def __len__(self):
        return self.len


class SIDDataset(Dataset):
    def __init__(self, root='./data/test/', in_filter='*.png', gt_filter='*.png', transforms=None, mode='train', random_crop=False, crop_size=128):

        self.mode = mode

        self.transforms = transforms

        self.in_path = root + '/short/' + in_filter
        self.gt_path = root + '/long/'

        self.in_list = glob.glob(self.in_path)
        self.gt_list = glob.glob(self.gt_path + in_filter)
        self.in_list.sort()
        self.gt_list.sort()

        self.random_crop = random_crop
        self.crop_size = crop_size

        self.len = len(self.in_list)

        print('Total files with GT: {:d}.'.format(self.len))

    def __getitem__(self, index):
        # start = time.time()
        in_path = self.in_list[index]
        # gt_path = self.gt_list[index]

        # print(os.path.basename(in_path))
        name = os.path.basename(in_path).split('_')[0] + '.png'

        gt_path = join(self.gt_path, name)
        # With OpenCV
        try:
            raw_image = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)[
                :, :, ::-1]  # .astype('int16')
            tar_image = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)[
                :, :, ::-1]  # .astype('int16')
        except:
            import ipdb
            ipdb.set_trace()

        raw_image = (raw_image.astype('float32') / (2.0**16 - 1))
        tar_image = tar_image.astype('float32') / 255

        if self.transforms is not None:
            raw_image = self.transforms(raw_image)
            tar_image = self.transforms(tar_image)

        if self.random_crop:

            i, j, h, w = trfs.RandomCrop.get_params(
                raw_image, output_size=(self.crop_size, self.crop_size))
            raw_image = TF.crop(raw_image, i, j, h, w)
            tar_image = TF.crop(tar_image, i, j, h, w)

        # parse metadata
        metadata = json_read(Path(in_path).with_suffix(
            '.json'), object_hook=fraction_from_json)
        # print("Loading time {}".format(time.time()-start))
        if self.mode == 'train':
            return raw_image, tar_image, metadata
        elif self.mode == 'test':
            return raw_image, tar_image, metadata, os.path.basename(in_path)

    def __len__(self):
        return self.len



class Night22Dataset(Dataset):
    def __init__(self, root='./data/test/', in_filter='*.png', gt_filter='*.png', transforms=None, mode='train'):

        self.mode = mode

        self.transforms = transforms

        self.in_path = root+'/'+in_filter

        self.in_list = glob.glob(self.in_path)
        self.in_list.sort()

        self.len = len(self.in_list)

        # self.normalize = trfs.Normalize(
        #     (0.038445, 0.07159842, 0.01086618),
        #     (0.04925333, 0.05836971, 0.0416128)
        # )

        self.normalize = trfs.Normalize(
            (0.03867557, 0.04543437, 0.03265842),
            (0.00991819, 0.01330488, 0.00814472)
        )

        print('Total files with GT: {:d}.'.format(self.len))

    def __getitem__(self, index):
        in_path = self.in_list[index]

        # With OpenCV
        raw_image = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)[:, :, ::-1]

        raw_image = raw_image.astype('float32') / (2.0**16 - 1)

        if self.transforms is not None:
            raw_image = self.transforms(raw_image)

        # raw_image = (self.normalize(raw_image) + 3) / 6

        # parse metadata
        metadata = json_read(Path(in_path).with_suffix(
            '.json'), object_hook=fraction_from_json)

        metadata.pop('linearization_table')
        if self.mode == 'train':
            return raw_image, metadata
        elif self.mode == 'test':
            return raw_image, metadata, os.path.basename(in_path)

    def __len__(self):
        return self.len

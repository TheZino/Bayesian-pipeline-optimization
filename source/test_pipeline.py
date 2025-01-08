import argparse
import os
import pickle as pkl
from pprint import pprint as pprint

import pandas as pd
from pypapi import events, papi_high
from pytorch_msssim import SSIM
from raw_pipeline.pipeline import PipelineExecutor
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils.camera_dataset import SIDDataset_thumb
from utils.metrics import *


expected_img_ext = '.png'

expected_landscape_img_height = 84
expected_landscape_img_width = 125
dataset_path_test = './dataset/test'

SEED = 1234

transforms = transforms.Compose([transforms.ToTensor()])

dataset = SIDDataset_thumb(root=dataset_path_test,
                           in_filter='*.png',
                           gt_filter='*' + expected_img_ext,
                           transforms=transforms,
                           mode='test')
dataloader = DataLoader(
    dataset, batch_size=1, num_workers=0, shuffle=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Demo script for processing PNG images with given metadata files.')
    parser.add_argument('-ckp', '--checkpoint', type=str,
                        help='Pipeline checkpoint to use')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='plot histograms')
    parser.add_argument('-si', '--save_images',
                        action='store_true')
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()

    with (open(args.checkpoint, "rb")) as openfile:
        ckpt = pkl.load(openfile)

    print("====\nRunning pipeline:\n")
    pprint(ckpt['pipeline'])
    print("\nBest value : {}".format(ckpt['best_value']))
    print("\nBest params:\n")
    pprint(ckpt['best_params'])
    print("\n====")

    ckpt['best_params']['out_landscape_width'] = expected_landscape_img_width
    ckpt['best_params']['out_landscape_height'] = expected_landscape_img_height
    save_dir = os.path.dirname(args.checkpoint) + '/test_images/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    pipeline_executor = PipelineExecutor(
        ckpt['pipeline'], ckpt['best_params'])

    ############################################################################
    avg_psnr = 0
    avg_ssim = 0

    ssim = SSIM(data_range=1, size_average=True, channel=3)

    names = []
    psnrs = []
    ssims = []
    avg_flops = 0
    for batch_idx, (raw_image, tar_image, metadata, name) in enumerate(tqdm(dataloader, desc="Test")):

        papi_high.start_counters([events.PAPI_FP_OPS,])
        output_image = pipeline_executor(
            raw_image, metadata, visualization=args.verbose)
        output_image = output_image.clip(0, 1)
        avg_flops += papi_high.stop_counters()[0]

        names.append(name[0])
        psnrs.append(0)

        ssims.append(0)

        avg_psnr += psnrs[-1]
        avg_ssim += ssims[-1]

        if args.save_images:
            output_image = (output_image[0, ...].permute(
                1, 2, 0).cpu().numpy() * 255).astype('uint8')
            cv2.imwrite(save_dir + name[0],
                        cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

    avg_flops = avg_flops / len(dataloader)
    print("flops: {}".format(avg_flops))
    print("AVG PSNR: {}\nAVG SSIM: {}".format(
        avg_psnr / len(dataloader), avg_ssim / len(dataloader)))

    import ipdb
    ipdb.set_trace()

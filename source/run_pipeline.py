import argparse
import pickle as pkl
from os.path import join
from pathlib import Path
from pprint import pprint

import cv2
import torch
from raw_pipeline.pipeline import PipelineExecutor
from utils import fraction_from_json, json_read

expected_img_ext = '.jpg'
expected_landscape_img_height = 86
expected_landscape_img_width = 130


def parse_args():
    parser = argparse.ArgumentParser(
        description='Demo script for processing PNG images with given metadata files.')
    parser.add_argument('-p', '--png_dir', type=Path, required=True,
                        help='Path of the directory containing PNG images with metadata files')
    parser.add_argument('-o', '--out_dir', type=Path, default='./',
                        help='Path to the directory where processed images will be saved. Images will be saved in JPG format.')
    parser.add_argument('-ie', '--illumination_estimation', type=str, default='gw',
                        help='Options for illumination estimation algorithms: "gw", "wp", "sog", "iwp".')
    parser.add_argument('-tm', '--tone_mapping', type=str, default='Flash',
                        help='Options for tone mapping algorithms: "Base", "Flash", "Storm", "Linear", "Drago", "Mantiuk", "Reinhard".')
    parser.add_argument('-n', '--denoising_flg', action='store_false',
                        help='Denoising flag. By default resulted images will be denoised with some default parameters.')
    parser.add_argument('-c', '--checkpoint', type=str,
                        help='Pipeline checkpoint to use')
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = args.png_dir

    return args


def exec_on_png(png_path, out_path, pipeline_executor, save=False):
    # parse raw img
    raw_image = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)[:, :, ::-1]

    # parse metadata
    metadata = json_read(png_path.with_suffix(
        '.json'), object_hook=fraction_from_json)

    raw_image = torch.Tensor(raw_image.astype(
        'float32').transpose([2, 0, 1])).unsqueeze(0)

    for key in metadata.keys():
        if not isinstance(metadata[key], str):
            if isinstance(metadata[key], list):
                els = []
                for el in metadata[key]:
                    els.append(torch.Tensor([el]))
                metadata[key] = els
            else:
                metadata[key] = torch.Tensor(metadata[key])

    output_image = pipeline_executor(raw_image, metadata)

    output_image = output_image.squeeze(0).numpy().transpose([1, 2, 0])

    name = png_path.name[:-4]
    print('Saving {} in {}'.format(name, str(out_path)))
    out_path = join(str(out_path), (name + '.jpg'))

    # save results
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), output_image, [
                cv2.IMWRITE_JPEG_QUALITY, 100])


if __name__ == "__main__":

    args = parse_args()

    with (open(args.checkpoint, "rb")) as openfile:
        ckpt = pkl.load(openfile)

    print("====\nRunning pipeline:\n")
    pprint(ckpt['pipeline'])
    print("\nBest params:\n")
    pprint(ckpt['best_params'])
    print("\n====")

    pipeline_executor = PipelineExecutor(ckpt['pipeline'], ckpt['best_params'])

    exec_on_png(args.png_dir, args.out_dir, pipeline_executor)

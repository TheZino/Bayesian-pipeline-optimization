import argparse
import os
import pickle as pkl
from pprint import pprint as pprint

from raw_pipeline.pipeline import PipelineExecutor
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils.camera_dataset import DIDDataset, Night22Dataset
from utils.metrics import *

expected_landscape_img_height = 3464
expected_landscape_img_width = 5202


def parse_args():
    parser = argparse.ArgumentParser(
        description='Demo script for processing PNG images with given metadata files.')
    parser.add_argument('-ckp', '--checkpoint', type=str,
                        help='Pipeline checkpoint to use')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='plot histograms')
    parser.add_argument('-d', '--dataset', type=str,
                        help='[ Night | DID ]')
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    expected_img_ext = '.png'
    transforms = transforms.Compose([transforms.ToTensor()])

    args = parse_args()

    with (open(args.checkpoint, "rb")) as openfile:
        ckpt = pkl.load(openfile)

    if args.dataset.lower() == "night":

        dataset = Night22Dataset(root='/home/zino/Datasets/Camera_pipe/night23/night23_demosaiced',
                                 in_filter='*.png',
                                 gt_filter='*' + expected_img_ext,
                                 transforms=transforms,
                                 mode='test')

    elif args.dataset.lower() == "did":

        dataset = DIDDataset(root='/home/zino/Datasets/Camera_pipe/DID/remapped_test',
                             in_filter='*.png',
                             gt_filter='*' + expected_img_ext,
                             transforms=transforms,
                             mode='only_input')

    else:
        raise Exception("Dataset not defined.")

    dataloader = DataLoader(
        dataset, batch_size=1, num_workers=0, shuffle=False)

    ckpt['pipeline'].pop(-2)
    print("====\nRunning pipeline:\n")
    pprint(ckpt['pipeline'])
    print("\nBest value : {}".format(ckpt['best_value']))
    print("\nBest params:\n")
    pprint(ckpt['best_params'])
    print("\n====")

    ckpt['best_params']['out_landscape_width'] = expected_landscape_img_width
    ckpt['best_params']['out_landscape_height'] = expected_landscape_img_height
    save_dir = os.path.dirname(args.checkpoint) + \
        '/test_images_' + args.dataset.lower() + '23_big/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    pipeline_executor = PipelineExecutor(
        ckpt['pipeline'], ckpt['best_params'])

    names = []

    for batch_idx, (raw_image, metadata, name) in enumerate(tqdm(dataloader, desc="Test")):

        output_image = pipeline_executor(
            raw_image, metadata, visualization=args.verbose)

        output_image = output_image.clip(0, 1)

        names.append(name[0])

        output_image = (output_image[0, ...].permute(
            1, 2, 0).cpu().numpy() * 255).astype('uint8')
        cv2.imwrite(save_dir + name[0], output_image[:, :, ::-1])

    import ipdb
    ipdb.set_trace()

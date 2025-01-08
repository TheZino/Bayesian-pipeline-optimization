import os
import pickle
import pprint
import time
import cv2

import matplotlib.pyplot as plt
import optuna
from raw_pipeline.pipeline_24 import PipelineExecutor
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils.camera_dataset import SIDDataset, SIDDataset_thumb
from utils.metrics import torch_PSNR, calc_ssim
from utils.parser import parse_options
from utils.plot_utils import plotCallback
from utils.utils import seed_everything


# SMALL-image dataset
expected_img_ext = '.png'

expected_landscape_img_height = 84
expected_landscape_img_width = 125
dataset_path = '/mnt/Datasets/Camera_pipe/Fuji/Fuji_thumb_remapped5/train/'
dataset_path_test = '/mnt/Datasets/Camera_pipe/Fuji/Fuji_thumb_remapped5/test/'
random_crop = False

SEED = 1234


op_sequence = [
    'nlm_denoise',
    'local_contrast_correction',
    'global_mean_contrast',
    'scurve',
    'imadjust',
    'conditional_contrast_correction',
    'sharpening_night',
    'GI_white_balance',
    'orientation'
]


transforms = transforms.Compose([transforms.ToTensor()])
dataset = SIDDataset_thumb(root=dataset_path,
                           in_filter='*.png',
                           gt_filter='*' + expected_img_ext,
                           transforms=transforms,
                           mode='train',
                           random_crop=random_crop,
                           crop_size=expected_landscape_img_height)
dataloader = DataLoader(dataset, batch_size=8, num_workers=0, shuffle=False)

dataset_test = SIDDataset(root=dataset_path_test,
                          in_filter='*.png',
                          gt_filter='*' + expected_img_ext,
                          transforms=transforms,
                          mode='test')
dataloader_test = DataLoader(
    dataset_test, batch_size=1, num_workers=8, shuffle=False)


def inference_pipe(params, save_dir=None):

    pipeline_executor = PipelineExecutor(op_sequence, params)

    avg_psnr = 0
    avg_ssim = 0

    for batch_idx, (raw_image, tar_image, metadata, name) in enumerate(dataloader_test):

        output_image = pipeline_executor(raw_image, metadata)

        avg_psnr += torch_PSNR(output_image, tar_image).item()
        avg_ssim += calc_ssim(output_image[0].permute(
            1, 2, 0).cpu().numpy(), tar_image[0].permute(
            1, 2, 0).cpu().numpy()).item()

        if save_dir is not None:
            output_image = (output_image[0, ...].permute(
                1, 2, 0).cpu().numpy() * 255).astype('uint8')
            cv2.imwrite(im_save_dir + name[0],
                        cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

    print("AVG PSNR: {}\nAVG SSIM: {}".format(
        avg_psnr / len(dataloader_test), avg_ssim / len(dataloader_test)))

    return avg_psnr / len(dataloader_test), avg_ssim / len(dataloader_test)


def objective(trial):
    '''
    Objective function for optuna.

    1: optuna trial parameter definition

    2: pipeline construction and execution

    3: error evaluation
    '''
    pipeline_params = {}

    if "nlm_denoise" in op_sequence:
        pipeline_params['nlm_l_w'] = trial.suggest_float(
            'nlm_l_w', 0, 30)
        pipeline_params['nlm_ch_w'] = trial.suggest_float(
            'nlm_ch_w', 0, 30)

    if "local_contrast_correction" in op_sequence:
        pipeline_params['LCC_sigma_p'] = trial.suggest_float(
            'LCC_sigma_p', 0, 12)

    if "global_mean_contrast" in op_sequence:
        pipeline_params['GMC_beta'] = trial.suggest_float('GMC_beta', 0.8, 2.0)
        pipeline_params['GMC_chsel'] = 0

    if "scurve" in op_sequence:
        pipeline_params['SCurve_alpha'] = trial.suggest_float(
            'SCurve_alpha', 0, 1)
        pipeline_params['SCurve_lambda'] = trial.suggest_float(
            'SCurve_lambda', 0.1, 10)

    if "imadjust" in op_sequence:
        pipeline_params['Hist_min'] = trial.suggest_float(
            'Hist_min', 0, 0.5)
        pipeline_params['Hist_max'] = trial.suggest_float(
            'Hist_max', 0.5, 1)

    if "conditional_contrast_correction" in op_sequence:
        pipeline_params['ccc_lambda_1'] = trial.suggest_float(
            'ccc_lambda_1', 0, 1)
        pipeline_params['ccc_lambda_2'] = trial.suggest_float(
            'ccc_lambda_2', 0, 1)
        pipeline_params['ccc_gamma'] = trial.suggest_float('ccc_gamma', 1, 10)

    if "sharpening_night" in op_sequence:
        pipeline_params['sharpening_sigma'] = trial.suggest_float(
            'sharpening_sigma', 1, 3)
        pipeline_params['sharpening_scale'] = trial.suggest_float(
            'sharpening_scale', 0, 2)

    if "GI_white_balance" in op_sequence:
        pipeline_params['GI_n'] = trial.suggest_float(
            'GI_n', 0, 1)
        pipeline_params['GI_th'] = trial.suggest_float(
            'GI_th', 1e-6, 1e-2)

    pipeline_params['out_landscape_width'] = expected_landscape_img_width
    pipeline_params['out_landscape_height'] = expected_landscape_img_height
    pipeline_params['debug'] = False

    pipeline_executor = PipelineExecutor(op_sequence, pipeline_params)

    metric1 = torch_PSNR


    error = 0

    print("\n\n")
    for batch_idx, (raw_image, tar_image, metadata) in enumerate(tqdm(dataloader, desc="Batch")):


        output_image = pipeline_executor(raw_image, metadata)
        error += metric1(output_image, tar_image)


    error /= len(dataloader)

    return error


if __name__ == "__main__":

    '''
    Here create the study object and define optimization procedure.
    The result of the optimization is stored in study
    '''

    opt = parse_options()

    seed_everything(opt.seed)

    global clbk
    global plot

    plot = opt.plot

    if opt.plot:
        clbk = plotCallback()
    else:
        clbk = None

    n_trials = opt.trials

    save_dir = opt.save_dir
    exp_name = opt.exp_id

    im_save_dir = save_dir + '/' + exp_name + '/test_images/'

    dirs = [save_dir, save_dir + '/' + exp_name + '/', im_save_dir]

    for dd in dirs:
        if not os.path.exists(dd):
            os.mkdir(dd)

    sampler = optuna.samplers.TPESampler(
        n_ei_candidates=64,
        multivariate=True,
        group=False,
        seed=opt.seed,
        warn_independent_sampling=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )

    ############################################################################

    print("===== CAMERA PIPELINE OPTIMIZATION - Parameters =====\n")
    print(f"Selected sampler: {study.sampler.__class__.__name__}\n\n")

    # study.enqueue_trial(
    #     {
    #         'GMC_beta': 1.5,
    #         'illest_algo': 'gray_world',
    #         'LCC_sigma_p': 7.24,
    #         'SCurve_alpha': 0,
    #         'SCurve_lambda': 0.5555555555555556,
    #         'Hist_max': 0.9999,
    #         'Hist_min': 0.0001,
    #         'GI_n': 0.1,
    #         'GI_th': 1e-4,
    #         'nlm_l_w': 4.5,
    #         'nlm_ch_w': 20,
    #         'sharpening_sigma': 2,
    #         'sharpening_scale': 1,
    #         'ccc_lambda_1': 1/1.8,
    #         'ccc_lambda_2': 1/1.4,
    #         'ccc_gamma': 2.2,
    #     }
    # )

    t0 = time.perf_counter()
    if clbk is not None:
        study.optimize(objective, n_trials=n_trials, callbacks=[clbk])
    else:
        study.optimize(objective, n_trials=n_trials)
    t1 = time.perf_counter() - t0
    print("\nTime elapsed: ", t1)  # CPU seconds elapsed (floating point)

    ############################################################################

    params = study.best_params

    extra_params = {
        'illest_algo': 'gray_world',
        'resize_function': 'pil',
        'debug': False,
        'out_landscape_width': expected_landscape_img_width,
        'out_landscape_height': expected_landscape_img_height,
    }

    params = {**params, **extra_params}

    ckpt = {
        'pipeline': op_sequence,
        'n_trials': n_trials,
        'best_value': study.best_value,
        'best_params': params,
    }

    with open(save_dir + '/' + exp_name + '/best.pkl', 'wb') as f:
        pickle.dump(ckpt, f)

    plt.savefig(save_dir + '/' + exp_name + '/steps.png',
                dpi=300, bbox_inches='tight')

    ############################################################################
    print("\n\n Experiment: " + exp_name)
    print("\n=== BEST PARAMETERS ===\n")
    print("= Best value: " + str(study.best_value) + "\n")
    pprint.pprint(study.best_params)

    plt.ioff()

    # inference with best parameters
    print("\n\n= Running pipeline with best parameters...")
    out_img = inference_pipe(params, im_save_dir)

    plt.show()

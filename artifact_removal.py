import argparse
import os
import shutil
from functools import partial

import numpy as np
import torch
import torchvision.io
from torchmetrics import PeakSignalNoiseRatio

from diffusion.gaussian_diffusion import GaussianDiffusion, get_beta_schedule
from train import define_network

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch import nn
from torch.utils import data
from torch.utils.data import DistributedSampler, TensorDataset
import torch.nn.functional as F
import torch.multiprocessing as mp
from torchvision.utils import save_image
from tqdm import tqdm
import core.parser as Parser
import core.util as Util

from core.logger import InfoLogger, VisualWriter
from core.parser import init_obj
from diffusion import gaussian_diffusion as gd, create_diffusion


def show(im):
    import matplotlib.pyplot as plt
    plt.imshow((0.5 + 0.5 * im)[0].permute(1, 2, 0).detach().cpu().numpy())
    plt.axis(False)
    plt.show()


def calibrate(model_low, model_high, calib_loader, diffusion, t, alpha, device):
    scores = []
    for x in tqdm(calib_loader):
        x = x.to(device)
        t_tag = torch.ones(x.shape[0], device=x.device).long() * t
        noise = torch.randn_like(x).to(device)
        high_quantile, _ = diffusion.clean_image(model_low, x, noise=noise, t=t_tag)
        low_quantile, _ = diffusion.clean_image(model_high, x, noise=noise, t=t_tag)
        score = torch.maximum(low_quantile - x, x - high_quantile)
        scores.append(score)
    scores = torch.cat(scores, dim=0)
    quantile = torch.quantile(scores, alpha, dim=0)
    return quantile

def remove_artifacts(x_start, quantiles, bounds_diffusion, sampling_diffusion, models):
    model_low, model_high, model = models
    for t, p, j in zip([750, 750, 750, 750, 250], [0.5, 0.5, 0.5, 0.5, 0.5], range(5)):
        quantile = quantiles[t]
        avg_mask_of_validity = torch.zeros_like(x_start)
        num = 4
        for i in range(num):
            noise = torch.randn_like(x_start)
            high_quantile, x_t_high = bounds_diffusion.clean_image(model_low, x_start, noise=noise, t=t)
            low_quantile, x_t_low = bounds_diffusion.clean_image(model_high, x_start, noise=noise, t=t)
            high_quantile += quantile
            low_quantile -= quantile

            mask_of_validity = torch.logical_and(x_start <= high_quantile, x_start >= low_quantile).any(dim=1, keepdim=True)
            avg_mask_of_validity += mask_of_validity / num

        mask_of_validity = (avg_mask_of_validity > p)[:, 0:1]
        # show(mask_of_validity)
        # mask_of_validity = (1 - torch.nn.functional.max_pool2d((1 - mask_of_validity.float()), 3, stride=1, padding=(1, 1))).bool()

        z = torch.randn_like(x_start)
        cleaned = sampling_diffusion.ddim_sample_loop(model, z.shape, z, clip_denoised=False,
                                                     progress=True, eta=0.85, device=z.device,
                                                    mask=mask_of_validity, y=mask_of_validity * x_start)

        save_image((0.5 + 0.5 * cleaned[0]), f"{j}.png")
        x_start = cleaned
    return x_start

def evaluate(val_loader, artifact_func, device, quantiles, bounds_diffusion, sampling_diffusion, models):
    psnr = PeakSignalNoiseRatio(2)
    psnr2 = PeakSignalNoiseRatio(2)
    psnr.to(device)
    psnr2.to(device)
    for im in tqdm(val_loader):
        im = im.to(device)
        im_w_artifact = artifact_func(im)
        cleaned_image = remove_artifacts(im_w_artifact, quantiles, bounds_diffusion, sampling_diffusion, models)
        psnr(cleaned_image, im)
        psnr2(im_w_artifact, im)

    print(f"Total PSNR: {psnr.compute()}")
    print(f"Total PSNR2: {psnr2.compute()}")


def main(opt, args):
    torch.backends.cudnn.enabled = True
    # warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
    Util.set_seed(opt['seed'])
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load model:
    model = define_network(phase_logger, opt, opt['model']['network'])
    model_low = define_network(phase_logger, opt, opt['model']['network'])
    model_high = define_network(phase_logger, opt, opt['model']['network'])
    sd = torch.load("ckpt.pth")
    print(model.load_state_dict(sd, strict=True))
    sd_low = torch.load(args.model_path_low)
    print(model_low.load_state_dict(sd_low, strict=True))
    sd_high = torch.load(args.model_path_high)
    print(model_high.load_state_dict(sd_high, strict=True))
    model_low.to(device)
    model_high.to(device)
    model.to(device)

    diffusion = GaussianDiffusion(betas=get_beta_schedule(**opt['model']['diffusion']['beta_schedule']),
                                  model_mean_type=gd.ModelMeanType.EPSILON,
                                  model_var_type=gd.ModelVarType.FIXED_LARGE,
                                  loss_type=gd.LossType.MSE)
    diffusion2 = create_diffusion("50")
    diffusion2.gamma = 1

    opt['datasets']['train']['which_dataset']['args']['data_root'] = "/home/noamelata/tmp/conff/datasets/calibration"
    opt['datasets']['train']['which_dataset']['args']['data_root'] = "/home/noamelata/tmp/conff/datasets/test/celebahq_test"
    train_dataset = init_obj(opt['datasets']['train']['which_dataset'], phase_logger, default_file_name='data.dataset',
                             init_type='Dataset')
    val_dataset = init_obj(opt['datasets']['validation']['which_dataset'], phase_logger,
                           default_file_name='data.dataset', init_type='Dataset')

    data_sampler = None
    loader_opts = dict(**opt['datasets']['train']['dataloader']['args'])
    val_loader_opts = dict(**opt['datasets']['validation']['dataloader']['args'])
    if opt['distributed']:
        data_sampler = DistributedSampler(train_dataset,
                                          shuffle=opt['datasets']['train']['dataloader']['args']['shuffle'],
                                          num_replicas=opt['world_size'],
                                          rank=opt['global_rank'])
        loader_opts["shuffle"] = False

    calib_loader = data.DataLoader(train_dataset, sampler=data_sampler, **loader_opts)
    val_loader = data.DataLoader(val_dataset, **val_loader_opts)
    quantile = opt['model']['quantile']

    # quantile_750 = calibrate(model_low, model_high, calib_loader, diffusion, 750, 0.9, device)
    # quantile_500 = calibrate(model_low, model_high, calib_loader, diffusion, 500, 0.9, device)
    # quantile_250 = calibrate(model_low, model_high, calib_loader, diffusion, 250, 0.9, device)
    # np.save("quantile_750_09.npy", quantile_750.detach().cpu().numpy())
    # np.save("quantile_500_09.npy", quantile_500.detach().cpu().numpy())
    # np.save("quantile_250_09.npy", quantile_250.detach().cpu().numpy())

    quantile_750 = torch.from_numpy(np.load(f"quantile_750_09.npy")).to(device)
    quantile_500 = torch.from_numpy(np.load(f"quantile_500_09.npy")).to(device)
    quantile_250 = torch.from_numpy(np.load(f"quantile_250_09.npy")).to(device)
    quantiles = {750: quantile_750, 500: quantile_500, 250: quantile_250}

    models = (model_low, model_high, model)

    loaded = np.load("inp_masks/lorem3.npy")
    rainbow = (torchvision.io.read_image("rainbow256.png").to(device) / 255) * 2 - 1
    # butterflys = (torchvision.io.read_image("butterflys2.png").to(device) / 255)
    # mask = 1 - butterflys[3]
    # butterflys = butterflys[0:3] * 2 - 1
    # loaded = np.load("inp_masks/lolcat_extra.npy")
    mask = torch.from_numpy(loaded).to(device)[None, None, :, :]

    # x_start = first_image * mask + (1-mask) * torch.cat([torch.ones_like(first_image[:, :1, :, :]), -torch.ones_like(first_image[:, 1:, :, :])], dim=1)
    # x_start = first_image * mask + (1-mask) * torch.cat([-torch.ones_like(first_image[:, :1, :, :]), torch.ones_like(first_image[:, :2, :, :])], dim=1)
    # x_start = first_image * mask + (1-mask) * torch.ones_like(first_image)
    # x_start = first_image * mask + (1-mask) * rainbow
    # x_start = first_image * mask
    # artifact_func = lambda x: x * mask + (1-mask) * butterflys
    artifact_func = lambda x: x * mask + (1-mask) * torch.cat([torch.ones_like(x[:, :1, :, :]), -torch.ones_like(x[:, 1:, :, :])], dim=1)

    evaluate(val_loader, artifact_func, device, quantiles, diffusion, diffusion2, models)






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/colorization_mirflickr25k.json',
                        help='JSON file for configuration')
    parser.add_argument('-ml', '--model-path-low', type=str, default="ckpt.pth")
    parser.add_argument('-mh', '--model-path-high', type=str, default="ckpt.pth")
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('-s', '--steps', type=int, default=50)
    # parser.add_argument('-e', '--eta', type=float, default=0.0)
    # parser.add_argument('--ddim', action='store_true', default=False)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('-P', '--port', default='21012', type=str)
    parser.add_argument('-gpu', '--gpu_ids', type=str, default="0")
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-p', '--phase', type=str, choices=['test'], help='Run train or test', default='test')

    ''' parser configs '''
    args = parser.parse_args()
    opt = Parser.parse(args)

    ''' cuda devices '''
    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    main(opt, args)

import argparse
import os
import shutil
from functools import partial

import numpy as np
import torch

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
    plt.show()



def main(opt, args):
    torch.backends.cudnn.enabled = True
    # warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
    Util.set_seed(opt['seed'])
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load model:
    model_low = define_network(phase_logger, opt, opt['model']['network'])
    model_high = define_network(phase_logger, opt, opt['model']['network'])
    sd_low = torch.load(args.model_path_low)
    print(model_low.load_state_dict(sd_low, strict=True))
    sd_high = torch.load(args.model_path_high)
    print(model_high.load_state_dict(sd_high, strict=True))
    model_low.to(device)
    model_high.to(device)

    diffusion = GaussianDiffusion(betas=get_beta_schedule(**opt['model']['diffusion']['beta_schedule']),
                                  model_mean_type=gd.ModelMeanType.EPSILON,
                                  model_var_type=gd.ModelVarType.FIXED_LARGE,
                                  loss_type=gd.LossType.MSE)

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

    train_loader = data.DataLoader(train_dataset, sampler=data_sampler, **loader_opts)
    val_loader = data.DataLoader(val_dataset, **val_loader_opts)
    quantile = opt['model']['quantile']

    first_image = next(iter(val_loader)).to(device)
    loaded = np.load("inp_masks/lorem3.npy")
    # loaded = np.load("inp_masks/lolcat_extra.npy")
    mask = torch.from_numpy(loaded).to(device)[None, None, :, :]
    # show(first_image)
    # show(first_image * mask)

    x_start = first_image * mask + (1-mask) * torch.cat([torch.ones_like(first_image[:, :1, :, :]), -torch.ones_like(first_image[:, 1:, :, :])], dim=1)
    # x_start = first_image * mask + (1-mask) * torch.ones_like(first_image)
    x_start = first_image * mask
    show(x_start)
    t = 500
    avg_mask_of_validity = torch.zeros_like(x_start)
    num = 4
    for i in range(num):
        noise = torch.randn_like(x_start)
        high_quantile, x_t_high = diffusion.clean_image(model_low, x_start, noise=noise, t=t)
        low_quantile, x_t_low = diffusion.clean_image(model_high, x_start, noise=noise, t=t)

        mask_of_validity = torch.logical_and(x_start <= high_quantile, x_start >= low_quantile).any(dim=1, keepdim=True)
        avg_mask_of_validity += mask_of_validity / num

    mask_of_validity = avg_mask_of_validity > 0.5
    show(mask_of_validity)
    show(1 - torch.nn.functional.max_pool2d((1 - mask_of_validity.float()), 3, padding=(1, 1)))

    pass







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

import argparse
import os
from functools import partial

import numpy as np
import torch
import torchvision
import wandb
from PIL import Image
from torch import nn
from torch.utils import data
from torch.utils.data import DistributedSampler
from torchvision.transforms import transforms
import torch.nn.functional as F
import torch.multiprocessing as mp

from Trainer import Trainer
from core.logger import InfoLogger, VisualWriter
import core.parser as Parser
import core.util as Util
from diffusion.gaussian_diffusion import GaussianDiffusion, get_beta_schedule
from core.parser import init_obj
from diffusion import gaussian_diffusion as gd


def mse_loss(output, target):
    return F.mse_loss(output, target)

def pinball_loss(output, target, alpha):
    return torch.where(target > output, (target - output) * alpha, (output - target) * (1 - alpha)).mean()


def mae(input, target):
    with torch.no_grad():
        loss = nn.L1Loss()
        output = loss(input, target)
    return output


def define_network(logger, opt, network_opt):
    """ define network with weights initialization """
    net = init_obj(network_opt, logger)

    if opt['phase'] == 'train':
        logger.info('Network [{}] weights initialize using [{:s}] method.'.format(net.__class__.__name__,
                                                                                  network_opt['args'].get('init_type',
                                                                                                          'default')))
        # net.init_weights()
    return net


def main_worker(gpu, ngpus_per_node, opt, wandb_run=False):
    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = gpu
    if opt['distributed']:
        torch.cuda.set_device(int(opt['local_rank']))
        print('using GPU {} for training'.format(int(opt['local_rank'])))
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=opt['init_method'],
                                             world_size=opt['world_size'],
                                             rank=opt['global_rank'],
                                             group_name='mtorch'
                                             )
    '''set seed and and cuDNN environment '''
    torch.backends.cudnn.enabled = True
    # warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
    Util.set_seed(opt['seed'])
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    if gpu == 0 and wandb_run:
        wandb_run = wandb.init(project="Conffusion", entity=wandb_run, config={})
        wandb_run.config.update(opt)
    else:
        wandb_run = None

    # Load model:
    model = define_network(phase_logger, opt, opt['model']['network'])
    sd = torch.load("ckpt.pth")
    # conv_in_weight = sd.pop("conv_in.weight")
    # sd["conv_in.weight"] = torch.cat([conv_in_weight, torch.nn.init.kaiming_normal_(torch.randn_like(conv_in_weight), nonlinearity='sigmoid')], 1)
    print(model.load_state_dict(sd, strict=True))

    diffusion = GaussianDiffusion(betas=get_beta_schedule(**opt['model']['diffusion']['beta_schedule']),
                                  model_mean_type=gd.ModelMeanType.EPSILON,
                                  model_var_type=gd.ModelVarType.FIXED_LARGE,
                                  loss_type=gd.LossType.MSE)

    train_dataset = init_obj(opt['datasets']['train']['which_dataset'], phase_logger, default_file_name='data.dataset', init_type='Dataset')
    val_dataset = init_obj(opt['datasets']['validation']['which_dataset'], phase_logger, default_file_name='data.dataset', init_type='Dataset')

    data_sampler = None
    loader_opts = dict(**opt['datasets']['train']['dataloader']['args'])
    val_loader_opts = dict(**opt['datasets']['validation']['dataloader']['args'])
    if opt['distributed']:
        data_sampler = DistributedSampler(train_dataset, shuffle=opt['datasets']['train']['dataloader']['args']['shuffle'], num_replicas=opt['world_size'],
                                          rank=opt['global_rank'])
        loader_opts["shuffle"] = False

    train_loader = data.DataLoader(train_dataset, sampler=data_sampler, **loader_opts)
    val_loader = data.DataLoader(val_dataset, **val_loader_opts)

    trainer = Trainer(
        network=model,
        diffusion=diffusion,
        phase_loader=train_loader,
        val_loader=val_loader,
        losses=[partial(pinball_loss, alpha=opt['model']['quantile'])],
        metrics=[mae],
        logger=phase_logger,
        writer=phase_writer,
        wandb_run=wandb_run,
        sample_num=opt['model']['diffusion']['beta_schedule']["num_diffusion_timesteps"],
        task="unconditional",
        optimizers=opt['model']['trainer']['args']['optimizers'],
        ema_scheduler=opt['model']['trainer']['args']['ema_scheduler'],
        mode=opt['model']['trainer']['args']['mode'],
        opt=opt,
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/celeba_hq_q005.json',
                        help='JSON file for configuration')
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'], help='Run train or test', default='train')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('--wandb', type=str, default='', help='W & B entity to use for wandb, leave empty for no W & B sync')

    ''' parser configs '''
    args = parser.parse_args()
    opt = Parser.parse(args)

    ''' cuda devices '''
    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    ''' use DistributedDataParallel(DDP) and multiprocessing for multi-gpu training'''
    # [Todo]: multi GPU on multi machine
    if opt['distributed']:
        ngpus_per_node = len(opt['gpu_ids'])  # or torch.cuda.device_count()
        opt['world_size'] = ngpus_per_node
        opt['init_method'] = 'tcp://127.0.0.1:' + args.port
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt, args.wandb))
    else:
        opt['world_size'] = 1
        main_worker(0, 1, opt, args.wandb)

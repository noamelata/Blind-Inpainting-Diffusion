import argparse
import os
import shutil
from functools import partial

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch import nn
from torch.utils import data
from torch.utils.data import DistributedSampler, TensorDataset
import torch.nn.functional as F
import torch.multiprocessing as mp
from torchvision.utils import save_image
from tqdm import tqdm

from core.logger import InfoLogger, VisualWriter
import core.parser as Parser
import core.util as Util
from diffusion.gaussian_diffusion import GaussianDiffusion, get_beta_schedule
from core.parser import init_obj
from diffusion import gaussian_diffusion as gd, create_diffusion
from torchvision.utils import make_grid

def mse_loss(output, target):
    return F.mse_loss(output, target)


def mae(input, target):
    with torch.no_grad():
        loss = nn.L1Loss()
        output = loss(input, target)
    return output

def show(im):
    import matplotlib.pyplot as plt
    plt.imshow((0.5 + 0.5 * im)[0].permute(1, 2, 0).detach().cpu().numpy())
    plt.show()


def define_network(logger, opt, network_opt):
    """ define network with weights initialization """
    net = init_obj(network_opt, logger)

    if opt['phase'] == 'train':
        logger.info('Network [{}] weights initialize using [{:s}] method.'.format(net.__class__.__name__,
                                                                                  network_opt['args'].get('init_type',
                                                                                                          'default')))
        # net.init_weights()
    return net


def main_worker(gpu, ngpus_per_node, opt, args):
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
    set_device = partial(Util.set_device, rank=opt["global_rank"])
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    # Load model:
    model = define_network(phase_logger, opt, opt['model']['network'])
    state_dict = torch.load(args.model_path)
    if args.model_path == "ckpt.pth":
        state_dict = state_dict[-1]
    print(model.load_state_dict(state_dict, strict=False))
    if torch.__version__.split(".")[0] == 2:
        model = torch.compile(set_device(model, distributed=opt['distributed']))
    else:
        model = set_device(model, distributed=opt['distributed'])
    model.eval()

    diffusion = create_diffusion(str(args.steps))

    if opt['global_rank'] == 0:
        if os.path.exists(args.output):
            shutil.rmtree(args.output)
        os.makedirs(args.output, exist_ok=True)
        if opt['distributed']:
            torch.distributed.barrier()
    else:
        if opt['distributed']:
            torch.distributed.barrier()
    dataset = TensorDataset(torch.arange(args.number))
    data_sampler = None
    if opt['distributed']:
        data_sampler = DistributedSampler(dataset, shuffle=False, num_replicas=opt['world_size'], rank=opt['global_rank'])
    class_loader = data.DataLoader(
        dataset,
        sampler=data_sampler,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )
    for (n,) in tqdm(class_loader):
        b = n.shape[0]
        z = torch.randn(b, 3, 64, 64)
        z = Util.set_device(z, distributed=opt['distributed'])
        if args.ddim:
            samples = diffusion.ddim_sample_loop(model, z.shape, z, clip_denoised=False,
                                                 progress=True, eta=args.eta, device=z.device)
        else:
            samples = diffusion.p_sample_loop(model, z.shape, z, clip_denoised=False,
                                              progress=True, device=z.device)
        for j in range(b):
            number = n[j].item()
            save_image((0.5 + 0.5 * samples[j]), os.path.join(args.output, f"{number:06d}.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/colorization_mirflickr25k.json',
                        help='JSON file for configuration')
    parser.add_argument('-n', '--number', type=int, default=10000)
    parser.add_argument('-m', '--model-path', type=str, default="ckpt.pth")
    parser.add_argument('-b', '--batch', type=int, default=64)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-s', '--steps', type=int, default=50)
    parser.add_argument('-e', '--eta', type=float, default=0.0)
    parser.add_argument('--ddim', action='store_true', default=False)
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

    ''' use DistributedDataParallel(DDP) and multiprocessing for multi-gpu training'''
    # [Todo]: multi GPU on multi machine
    if opt['distributed']:
        ngpus_per_node = len(opt['gpu_ids'])  # or torch.cuda.device_count()
        opt['world_size'] = ngpus_per_node
        opt['init_method'] = 'tcp://127.0.0.1:' + args.port
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt, args))
    else:
        opt['world_size'] = 1
        main_worker(0, 1, opt, args)

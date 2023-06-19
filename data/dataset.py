import torch.utils.data as data
import torchvision
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
from torchvision.io import encode_jpeg, decode_jpeg
import torchvision.transforms.functional as F


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

class DS(data.Dataset):
    def __init__(self, data_root, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.tfs(self.loader(path))

        return img

    def __len__(self):
        return len(self.imgs)


class DirDataset(data.Dataset):
    def __init__(self, data_root, data_len=-1, image_size=[64, 64], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        return img

    def __len__(self):
        return len(self.imgs)


class PreporcessedDataset(data.Dataset):
    def __init__(self, data_root, data_len=-1, image_size=[64, 64], grayscale=False, loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                (transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) if not grayscale else
                 transforms.Normalize(mean=[0.5], std=[0.5]))
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = torch.from_numpy(np.load(path.replace("image", "mask").replace("png", "npy")))

        return {"image": img, "mask": mask}

    def __len__(self):
        return len(self.imgs)

from functools import reduce  # Required in Python 3
import operator
def prod(iterable):
    return reduce(operator.mul, iterable, 1)


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


class CelebAUnconditional(torchvision.datasets.CelebA):
    def __init__(self, data_root, data_len=-1, image_size=[64, 64], rand_flip=False, loader=pil_loader, **kwargs):
        cx = 89
        cy = 121
        x1 = cy - 64
        x2 = cy + 64
        y1 = cx - 64
        y2 = cx + 64
        self.tfs = transforms.Compose([
                Crop(x1, x2, y1, y2),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                # transforms.RandomHorizontalFlip() if rand_flip else lambda x: x,
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.data_root = data_root
        super(CelebAUnconditional, self).__init__(root=data_root, transform=self.tfs, download=True, **kwargs)
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        img = super(CelebAUnconditional, self).__getitem__(index)

        return img[0]

class CifarUnconditional(torchvision.datasets.CIFAR10):
    def __init__(self, data_root, data_len=-1, image_size=[32, 32], rand_flip=False, loader=pil_loader, **kwargs):
        print(f"Rand Flip {rand_flip}")
        self.tfs = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip() if rand_flip else lambda x: x,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.data_root = data_root
        super(CifarUnconditional, self).__init__(root=data_root, transform=self.tfs, download=True, **kwargs)
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        img = super(CifarUnconditional, self).__getitem__(index)

        return img[0]


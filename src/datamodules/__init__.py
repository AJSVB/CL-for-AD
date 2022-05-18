
from torchvision.transforms import transforms


import torchvision.transforms as transforms

from PIL import ImageFilter
import random
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


transform_color = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_resnet18 = transforms.Compose([
    transforms.Resize(224, interpolation=BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])




moco_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])



transform_diagvib = transforms.Compose([
    transforms.Resize(224, interpolation=BICUBIC),
    transforms.CenterCrop(224)
])



class Transform:
    def __init__(self):
        self.moco_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __call__(self, x):
        x_1 = self.moco_transform(x)
        x_2 = self.moco_transform(x)
        return x_1, x_2



import os

import numpy as np
import torch
from torch.utils.data.dataset import Subset

from torchvision import datasets, transforms

DATA_PATH = '~/data/'
IMAGENET_PATH = '~/data/ImageNet'


CIFAR10_SUPERCLASS = list(range(10))  # one class
IMAGENET_SUPERCLASS = list(range(30))  # one class

CIFAR100_SUPERCLASS = [
    [4, 31, 55, 72, 95],
    [1, 33, 67, 73, 91],
    [54, 62, 70, 82, 92],
    [9, 10, 16, 29, 61],
    [0, 51, 53, 57, 83],
    [22, 25, 40, 86, 87],
    [5, 20, 26, 84, 94],
    [6, 7, 14, 18, 24],
    [3, 42, 43, 88, 97],
    [12, 17, 38, 68, 76],
    [23, 34, 49, 60, 71],
    [15, 19, 21, 32, 39],
    [35, 63, 64, 66, 75],
    [27, 45, 77, 79, 99],
    [2, 11, 36, 46, 98],
    [28, 30, 44, 78, 93],
    [37, 50, 65, 74, 80],
    [47, 52, 56, 59, 96],
    [8, 13, 48, 58, 90],
    [41, 69, 81, 85, 89],
]


class MultiDataTransform(object):
    def __init__(self, transform):
        self.transform1 = transform
        self.transform2 = transform

    def __call__(self, sample):
        x1 = self.transform1(sample)
        x2 = self.transform2(sample)
        return x1, x2


class MultiDataTransformList(object):
    def __init__(self, transform, clean_trasform, sample_num):
        self.transform = transform
        self.clean_transform = clean_trasform
        self.sample_num = sample_num

    def __call__(self, sample):

        sample_list = []
        for i in range(self.sample_num):
            sample_list.append(self.transform(sample))

        return sample_list, self.clean_transform(sample)





def get_subset_with_len(dataset, length, shuffle=False):
    dataset_size = len(dataset)

    index = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(index)

    index = torch.from_numpy(index[0:length])
    subset = Subset(dataset, index)

    assert len(subset) == length

    return subset

from . import *


def get_dataset( dataset, test_only=True, image_size=1, download=True):

    train_transform, test_transform = Transform(), transform_resnet18

    train_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=train_transform)
    val_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=test_transform)
    test_set_id = datasets.CIFAR10(DATA_PATH, train=False, download=download, transform=test_transform)

    if dataset == 'cifar10':
        image_size = (32, 32, 3)
        n_classes = 10
        test_set = datasets.CIFAR10(DATA_PATH, train=False, download=download, transform=test_transform)

    elif dataset == 'cifar100':
        image_size = (32, 32, 3)
        n_classes = 100
        train_set = datasets.CIFAR100(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR100(DATA_PATH, train=False, download=download, transform=test_transform)

    elif dataset == 'stl10':
        assert test_only and image_size is not None
        test_set = datasets.STL10(DATA_PATH, split='test', download=download, transform=test_transform)


    elif dataset == 'svhn':
        assert test_only and image_size is not None
        test_set = datasets.SVHN(DATA_PATH, split='test', download=download, transform=test_transform)

    elif dataset == 'lsun_resize':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'LSUN_resize')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'lsun_fix':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'LSUN_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet_resize':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'Imagenet_resize')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet_fix':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'Imagenet_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet':
        image_size = (224, 224, 3)
        n_classes = 30
        train_dir = os.path.join(IMAGENET_PATH, 'one_class_train')
        test_dir = os.path.join(IMAGENET_PATH, 'one_class_test')
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'stanford_dogs':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'stanford_dogs')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'cub':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'cub200')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'flowers102':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'flowers102')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'places365':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'places365')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'food_101':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'food-101', 'images')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'caltech_256':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'caltech-256')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'dtd':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'dtd', 'images')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'pets':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'pets')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    else:
        raise NotImplementedError()


    return train_set, val_set, test_set_id, test_set, image_size




def get_dataset1( dataset, test_only=True, image_size=1, download=True):

    train_transform, test_transform = Transform(), transform_resnet18

    f = datasets.FGVCAircraft

    train_set = f(DATA_PATH, split="train", download=download, transform=train_transform)
    val_set = f(DATA_PATH, split="train", download=download, transform=test_transform)
    test_set_id = f(DATA_PATH, split="test", download=download, transform=test_transform)
    test_set = f(DATA_PATH, split="test", download=download, transform=test_transform)
    se = np.array(set(train_set._labels))
    print(se)
    train_set = torch.utils.data.Subset(train_set, np.argwhere(np.array(train_set._labels)==0).flatten())
    val_set = torch.utils.data.Subset(val_set, np.argwhere(np.array(val_set._labels)==0).flatten())
    print(sum(np.array(test_set_id._labels)==0))
    test_set_id = torch.utils.data.Subset(test_set_id, np.argwhere(np.array(test_set_id._labels)==0).flatten())
    print(len(test_set_id))

    print(sum(np.array(test_set._labels)==1))
    test_set = torch.utils.data.Subset(test_set, np.argwhere(np.array(test_set._labels)==1).flatten())
    print(len(test_set))
    return train_set, val_set, test_set_id, test_set, image_size


def get_superclass_list(dataset):
    if dataset == 'cifar10':
        return CIFAR10_SUPERCLASS
    elif dataset == 'cifar100':
        return CIFAR100_SUPERCLASS
    elif dataset == 'imagenet':
        return IMAGENET_SUPERCLASS
    else:
        raise NotImplementedError()


def get_subclass_dataset(dataset, classes):
    if not isinstance(classes, list):
        classes = [classes]

    indices = []
    for idx, tgt in enumerate(dataset.targets):
        if tgt in classes:
            indices.append(idx)

    dataset = Subset(dataset, indices)
    return dataset


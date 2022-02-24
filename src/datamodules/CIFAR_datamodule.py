from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import faiss
import torchvision.models as models
import torch.nn.functional as F
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




class CIFARDataModule(LightningDataModule):
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        backbone: int = 152,
        num_workers: int = 0,
        label_class= 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)


        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        ds = torchvision.datasets.CIFAR10
        transform = transform_color if self.hparams.backbone == 152 else transform_resnet18
        coarse = {}
        self.trainset = ds(root=self.hparams.data_dir, train=True, download=True, transform=transform, **coarse)
        self.testset = ds(root=self.hparams.data_dir, train=False, download=True, transform=transform, **coarse)
        self.trainset_1 = ds(root=self.hparams.data_dir, train=True, download=True, transform=Transform(), **coarse)
        self.is_setup = False

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        # load datasets only if they're not loaded already
        if not self.is_setup:
            self.is_setup = True
            idx = np.array(self.trainset.targets) == self.hparams.label_class
            self.testset.targets = [int(t != self.hparams.label_class) for t in self.testset.targets]
            self.trainset.data = self.trainset.data[idx]
            self.trainset.targets = [self.trainset.targets[i] for i, flag in enumerate(idx, 0) if flag]
            self.trainset_1.data = self.trainset_1.data[idx]
            self.trainset_1.targets = [self.trainset_1.targets[i] for i, flag in enumerate(idx, 0) if flag]

    def train_dataloader(self):

        return torch.utils.data.DataLoader(self.trainset_1, batch_size=self.hparams.batch_size, shuffle=True,
                                               num_workers=self.hparams.num_workers, drop_last=False)

    def val_dataloader(self):
        print(len(self.trainset))
        return torch.utils.data.DataLoader(self.trainset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers,
                                    drop_last=False)


    def test_dataloader(self):
        print(len(self.testset))
        return torch.utils.data.DataLoader(self.testset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers,
                                                  drop_last=False)

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
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader



from . import *


class WILDSModule(LightningDataModule):
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
            MSCL=True
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.is_setup = False

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None



    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        get_dataset(dataset="rxrx1", download=True, root_dir=self.hparams.data_dir)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        # load datasets only if they're not loaded already
        if not self.is_setup:
            self.is_setup = True


            transform = transform_color if self.hparams.backbone == 152 else transform_resnet18

            if not self.hparams.MSCL:
                tx = lambda: transform_resnet18
            else:
                tx= Transform


            dataset = get_dataset(dataset="rxrx1", download=False,
                                  root_dir=self.hparams.data_dir)


            self.trainset = dataset.get_subset(
            "train",transform = transform )
            self.testset = dataset.get_subset(
            "test",transform = transform )
            self.trainset_1 = dataset.get_subset(
            "train",transform = tx() )

            self.trainset = torch.utils.data.Subset(self.trainset, torch.argwhere(self.trainset.y_array == 0))
            self.trainset_1 = torch.utils.data.Subset(self.trainset_1, torch.argwhere(self.trainset_1.y_array == 0))
            print("trainset size "+ str(len(self.trainset)))

            self.testset = torch.utils.data.Subset(self.testset, torch.argwhere((self.testset.y_array == 1) |(self.testset.y_array == 0)))
            print("testset size " +str(len(self.testset)))


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trainset_1, batch_size=self.hparams.batch_size, shuffle=True,
                                               num_workers=self.hparams.num_workers, drop_last=False)

    def val_dataloader(self):
        a = torch.utils.data.DataLoader(self.trainset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers,
                                    drop_last=False)
        return a

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.testset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers,
                                                  drop_last=False)

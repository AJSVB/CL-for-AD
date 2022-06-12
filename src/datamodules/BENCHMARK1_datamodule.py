from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
import numpy as np
import faiss
import torchvision.models as models
import torch.nn.functional as F
import torch
import torchvision
from . import *

class Benchmark1DataModule(LightningDataModule):
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
            MSCL=True,
            dataset = None
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.is_setup = False

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        from . import get_dataset
        _ = get_dataset1(self.hparams.dataset)


    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        # load datasets only if they're not loaded already
        if not self.is_setup:
            self.is_setup = True

            from . import get_dataset
            train, val, test_id, test_ood,_  = get_dataset1(self.hparams.dataset, self.hparams.MSCL)
            #train.target_transform = lambda id: 0
            #val.target_transform = lambda id: 0
            #test_id.target_transform = lambda id: 0
            #test_ood.target_transform = lambda id: 1
            self.trainset_1 = train #get_subset_with_len(train,20000)
            self.trainset = val #get_subset_with_len(val,20000)
            self.testset = torch.utils.data.ConcatDataset((test_id,test_ood)) #get_subset_with_len(torch.utils.data.ConcatDataset((test_id,test_ood)),20000)
            #for a in self.testset:
            #    print(a)


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

from typing import Any, List
import numpy as np
#import faiss
import faiss
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.components.simple_dense_net import SimpleDenseNet

from src.models.components.Mean_shifted_AD_net import Mean_shifted_AD_net

from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
import torchvision.transforms as transforms
from PIL import ImageFilter
import random
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


class MNISTLitModule(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model = SimpleDenseNet(hparams=self.hparams)

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )


class MSAD(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        backbone,pretrained, label, lr,batch_size,angular
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model = Mean_shifted_AD_net(hparams=self.hparams)

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

        self.first_epoch = True

        self.train_feature_space = []
        self.val_feature_space= []
        self.test_feature_space = []
        self.val_labels = []
        self.test_labels = []
        self.total_loss, self.total_num = 0.0, 0



    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        if self.first_epoch:
            pass
        else:

            loss = self.run_epoch(batch)

            # log train metrics
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)

            # we can return here dict with any tensors
            # and then read it in some callback or in `training_epoch_end()`` below
            # remember to always return loss from `training_step()` or else backpropagation will fail!
            return {"loss": loss}


    def training_epoch_end(self, outputs: List[Any]):
        print("training finished")
        if self.first_epoch:
            #training_epoch_end is called after validation_epoch_end
            self.first_epoch = False
            loss = 100
        # `outputs` is a list of dicts returned from `training_step()`
        else:
            loss = self.total_loss / self.total_num
            self.total_loss, self.total_num = 0.0, 0

        self.log("train/acc", loss, on_epoch=True, prog_bar=True)


    def validation_step(self, batch: Any, batch_idx: int):
        print(batch_idx)

        if self.first_epoch:
            x, y = batch
            features = self.model(x)
            self.train_feature_space.append(features)
        else:
            x, y = batch
            features = self.model(x)
            print(self.val_feature_space)
            print(features)
            self.val_feature_space.append(features)

    def validation_epoch_end(self, outputs: List[Any]):
        print("FIRST EPOCH VAL FINISHED")
        if self.first_epoch:
            self.train_feature_space = torch.cat(self.train_feature_space, dim=0).contiguous().cpu().numpy() #TODO cpu?
            self.center = torch.FloatTensor(self.train_feature_space).mean(dim=0)
            if self.hparams.angular:
                self.center = F.normalize(self.center, dim=-1)
            auc=0
            self.center = self.center.to(self.device)

        else:
            self.treated_val_feature_space = torch.cat(self.val_feature_space, dim=0).contiguous().cpu().numpy()
            #val_labels = torch.cat(self.val_labels, dim=0).cpu().numpy()
            #distances = knn_score(self.train_feature_space, self.val_feature_space)
            #print(val_labels)
            #print(distances)
            #auc = roc_auc_score(val_labels, distances)

        self.log("val/acc", 0, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        if self.first_epoch:
            pass
        else:
            x, y = batch
            features = self.model(x)
            self.test_feature_space.append(features)
            self.test_labels.append(y)

    def test_epoch_end(self, outputs: List[Any]):
        print("test end")
        if self.first_epoch:
            auc = 0
        else:
            self.treated_test_feature_space = torch.cat(self.test_feature_space, dim=0).contiguous().cpu().numpy()
            test_labels = torch.cat(self.test_labels, dim=0).cpu().numpy()
            distances = knn_score(self.treated_val_feature_space, self.treated_test_feature_space)
            auc = roc_auc_score(test_labels, distances)
            self.val_feature_space= []
            self.val_labels = []
            self.test_feature_space = []
            self.test_labels = []

        self.log("test/acc", auc, on_epoch=True, prog_bar=True)


    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        print("when does this happens?")
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()






    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.SGD(params=self.parameters(), lr=self.hparams.lr, weight_decay=0.00005)

    def run_epoch(self, batch):
        print("run epoch")
        (img1, img2), y = batch

    #    self.optimizer.zero_grad()

        out_1 = self.model(img1)
        out_2 = self.model(img2)
        out_1 = out_1 - self.center
        out_2 = out_2 - self.center

        loss = contrastive_loss(out_1, out_2)

        if self.hparams.angular:
            loss += ((out_1 ** 2).sum(dim=1).mean() + (out_2 ** 2).sum(dim=1).mean())


        self.total_num += img1.size(0)
        self.total_loss += loss.item() * img1.size(0)

        return loss


def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)


def contrastive_loss(out_1, out_2):
    out_1 = F.normalize(out_1, dim=-1)
    out_2 = F.normalize(out_2, dim=-1)
    bs = out_1.size(0)
    temp = 0.25
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temp)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * bs, device=sim_matrix.device)).bool()
    # [2B, 2B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * bs, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temp)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss







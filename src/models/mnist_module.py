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
        self.center_not_computed = True
        self.train_feature_space = []
        self.automatic_optimization = False
        self.total_loss, self.total_num = 0.0, 0
        self.test_feature_space = []
        self.test_labels= []
    def on_validation_model_train(self):
        pass #we overwrite model.train with nothing, therefor, model stays in eval mode

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        self.model.eval()
        return torch.optim.SGD(params=self.model.parameters(), lr=self.hparams.lr, weight_decay=0.00005)

    def validation_step(self, batch: Any, batch_idx: int):
        if self.center_not_computed:
            features = get_feature_space(self.model, batch)
            self.train_feature_space.append(features)
        elif self.trainer.max_epochs -1== self.current_epoch:
            features = get_feature_space(self.model, batch)
            self.train_feature_space.append(features)


    def on_validation_epoch_end(self):
        if self.center_not_computed:
            print("la")
            self.center_not_computed=False
            train_feature_space = torch.cat(self.train_feature_space, dim=0).contiguous().cpu().numpy()
            self.center = torch.FloatTensor(train_feature_space).mean(dim=0)
            if self.hparams.angular:
                self.center = F.normalize(self.center, dim=-1)
            self.center = self.center.to(self.device)
            loss = 10 #Dummy value
            self.train_feature_space = []
        else:
            print(self.current_epoch)
            print(self.trainer.max_epochs)
            if(self.current_epoch==self.trainer.max_epochs-1):
                self.treated_val_feature_space = torch.cat(self.train_feature_space, dim=0).contiguous().cpu().numpy()

            loss = self.total_loss / self.total_num
            self.total_loss, self.total_num = 0.0, 0
        self.log("val/acc", -loss, on_epoch=True, prog_bar=True)

    def training_step(self, batch: Any, batch_idx: int):
      #  self.opt = self.optimizers()
        loss = run_epoch(self.model,batch,self.optimizers(),self.center,self.hparams.angular)
    #    self.manual_backward(loss) TODO
      #  self.opt.step()
        self.total_num += batch[0][0].size(0)
        self.total_loss += loss.item() * batch[0][0].size(0)
        # log train metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)

    def test_step(self, batch: Any, batch_idx: int):
            features = get_feature_space(self.model, batch)
            self.test_feature_space.append(features)
            (_, y) = batch
            self.test_labels.append(y)

    def test_epoch_end(self, outputs: List[Any]):
            self.treated_test_feature_space = torch.cat(self.test_feature_space, dim=0).contiguous().cpu().numpy()
            test_labels = torch.cat(self.test_labels, dim=0).cpu().numpy()
            distances = knn_score(self.treated_val_feature_space, self.treated_test_feature_space)
            auc = roc_auc_score(test_labels, distances)
            self.log("test/acc", auc, on_epoch=True, prog_bar=True)


def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)

def get_feature_space(model, batch):
    (imgs, _) = batch
    features = model(imgs)
    return features


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

def run_epoch(model, batch, optimizer, center, is_angular):
    ((img1, img2), _) = batch

    optimizer.zero_grad()

    out_1 = model(img1)
    out_2 = model(img2)
    out_1 = out_1 - center
    out_2 = out_2 - center

    loss = contrastive_loss(out_1, out_2)
    if is_angular:
        loss += ((out_1 ** 2).sum(dim=1).mean() + (out_2 ** 2).sum(dim=1).mean())
    print(loss)

    loss.backward()

    optimizer.step()
    return loss



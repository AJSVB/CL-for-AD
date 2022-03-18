

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

        self.first_epoch = True

        self.train_feature_space = []
        self.val_feature_space= []
        self.test_feature_space = []
        self.val_labels = []
        self.test_labels = []
        self.total_loss, self.total_num = 0.0, 0
        self.model.eval()
        self.loss = 100 #Dummy value
        self.printing_cosine_similarity_experiment = False


    def training_step(self, batch: Any, batch_idx: int):
            self.model.eval()

            loss = self.run_epoch(batch)

            # log train metrics
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
            # we can return here dict with any tensors
            # and then read it in some callback or in `training_epoch_end()`` below
            # remember to always return loss from `training_step()` or else backpropagation will fail!
            return {"loss": loss}


    def training_epoch_end(self, outputs: List[Any]):
        loss = self.total_loss / self.total_num
        self.total_loss, self.total_num = 0.0, 0
        self.loss = loss
      #  self.log("train/loss", loss, on_epoch=True, prog_bar=True)


    def validation_step(self, batch: Any, batch_idx: int):

        if self.first_epoch:
            x, y = batch
            features = self.model(x)
            self.train_feature_space.append(features)
            self.val_labels.append(y)

        elif self.trainer.max_epochs - 1 == self.current_epoch:
            x, y = batch
            features = self.model(x)
            self.val_feature_space.append(features)
            self.val_labels.append(y)

    def validation_epoch_end(self, outputs: List[Any]):
        if self.first_epoch:
            self.train_feature_space = torch.cat(self.train_feature_space, dim=0).contiguous().cpu().numpy()
            val_labels = torch.cat(self.val_labels, dim=0).cpu().numpy()
            if(self.printing_cosine_similarity_experiment):
                idx = np.array(val_labels) != 0
                Joao_similarity(self.train_feature_space[idx],val_labels[idx],self.train_feature_space[~idx],0,"before_training")
                self.train_feature_space = self.train_feature_space[idx]

#            self.center,_ = torch.FloatTensor(self.train_feature_space).median(dim=0)
            self.center = torch.FloatTensor(self.train_feature_space).mean(dim=0)

            if self.hparams.angular:
                self.center = F.normalize(self.center, dim=-1)
            self.center = self.center.to(self.device)
            self.first_epoch = False
            self.val_labels = []
        elif self.trainer.max_epochs - 1 == self.current_epoch:
            self.treated_val_feature_space = torch.cat(self.val_feature_space, dim=0).contiguous().cpu().numpy()
            if(self.printing_cosine_similarity_experiment):
                val_labels = torch.cat(self.val_labels, dim=0).cpu().numpy()
                idx = np.array(val_labels) != 0
                Joao_similarity(self.treated_val_feature_space[idx],val_labels[idx],self.treated_val_feature_space[~idx],0,"after_training")
                self.treated_val_feature_space = self.treated_val_feature_space[idx]

           # tsne(self.treated_val_feature_space,val_labels,"after_training")



        self.log("val/acc", -self.loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        x, y = batch
        print(y)
        features = self.model(x)
        self.test_feature_space.append(features)
        self.test_labels.append(y)

    def test_epoch_end(self, outputs: List[Any]):
        self.treated_test_feature_space = torch.cat(self.test_feature_space, dim=0).contiguous().cpu().numpy()
        test_labels = torch.cat(self.test_labels, dim=0).cpu().numpy()
        distances = knn_score(self.treated_val_feature_space, self.treated_test_feature_space)
        auc = roc_auc_score(test_labels, distances)

        self.log("test/acc", auc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.SGD(params=self.parameters(), lr=self.hparams.lr, weight_decay=0.00005)

    def run_epoch(self, batch):
        (img1, img2), y = batch
        print(y)
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


def tsne(data,labels,name):
    from sklearn.manifold import TSNE  # Picking the top 1000 points as TSNE takes a lot of time for 15K points
    import pandas as pd
    import seaborn as sn
    import matplotlib.pyplot as plt
    import os
    model = TSNE(n_components=2, random_state=0)
    tsne_data = model.fit_transform(data)# creating a new data frame which help us in ploting the result data
    tsne_data = np.vstack((tsne_data.T, labels)).T
    tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))  # Ploting the result of tsne
    print(os.getcwd())
    sn.FacetGrid(tsne_df, hue="label", size = 6).map(plt.scatter, "Dim_1", "Dim_2").add_legend().savefig(name+".pdf")
    print("this worked?")



def Joao_similarity(normal_data, normal_labels, anomalous_data,anomalous_labels , name,reduction_factor=10,constant_closest = 2):
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics.pairwise import euclidean_distances

    def f(X):
        return X[:int(len(X)/reduction_factor)]

    def g(X):
        return X[:constant_closest]
    all_labels = np.unique(normal_labels)

    lis = [euclidean_distances,cosine_similarity]
    for i in range(2):
        dist_metric = lis[i]
        for label in all_labels:
            label_idx = normal_labels == label
            label_data = normal_data[label_idx]
            anomalous_outer_similarity=0
            for anomalous_sample in f(anomalous_data):
                distances = dist_metric([anomalous_sample],label_data)
                distance_idx = np.argsort(distances)[0]
                if(i==1):
                    distance_idx=distance_idx[::-1]
                all_closest = g(label_data[distance_idx])
                anomalous_outer_similarity+=np.mean(dist_metric(all_closest,[anomalous_sample]))
            print("similarity between anomalies and closest samples from environment " + str(label) + " using distance metric " + str(dist_metric) +
                      " : " + str((anomalous_outer_similarity/len(f(anomalous_data)))))



        anomalous_outer_similarity=0
        for anomalous_sample in f(anomalous_data):
            label_data= normal_data
            distances = dist_metric([anomalous_sample],label_data)
            distance_idx = np.argsort(distances)[0]
            if (i ==1):
                distance_idx = distance_idx[::-1]
            all_closest = g(label_data[distance_idx])
            anomalous_outer_similarity+=np.mean(dist_metric(all_closest,[anomalous_sample]))

        normal_outer_similarity=0
        for normal_sample in f(f(normal_data)):
            label_data= normal_data
            distances = dist_metric([normal_sample],label_data)
            distance_idx = np.argsort(distances)[0]
            if (i ==1):
                distance_idx = distance_idx[::-1]
            all_closest = g(label_data[distance_idx][1:])
            normal_outer_similarity+=np.mean(dist_metric(all_closest,[normal_sample]))


        print("similarity between anomalies and closest normal samples using distance metric " + str(dist_metric) +
                      " : " + str(anomalous_outer_similarity/len(f(anomalous_data))))

        print("similarity between normal samples and closest normal samples using distance metric " + str(dist_metric) +
                      " : " + str(normal_outer_similarity/len(f(f(normal_data)))))





def Cosine_similarity(normal_data, normal_labels, anomalous_data,anomalous_labels , name):
    from sklearn.metrics.pairwise import cosine_similarity

    reduction_factor = 10
    def f(X):
        return X[:int(len(X)/reduction_factor)]

    def cosine_similarity1(X,Y=None):
        return [np.mean(cosine_similarity(X,Y))]
    #TODO Here I assume that anomalous_labels = 0
    print("we get here I suppose?")
    normal_inner_similarity = []
    normal_outer_similarity = []
    anomalous_inner_similarity = []
    anomalous_outer_similarity = []
    #inner normal
    for label in range(1,10):
        idx = np.array(normal_labels) == label
        normal_inner_similarity+=cosine_similarity1(f(normal_data[idx]))
    #outer normal
    for label in range(1,10):
        for label2 in range(label+1,10):
            idx = np.array(normal_labels) == label
            idx2 = np.array(normal_labels) == label2
            normal_outer_similarity+=cosine_similarity1(f(normal_data[idx]),f(normal_data[idx2]))
    #inner anomalous
    anomalous_inner_similarity+=cosine_similarity1(f(anomalous_data))
    #outer anomalous
    anomalous_outer_similarity+=cosine_similarity1(f(normal_data),f(anomalous_data))

    print("normal_inner_similarity "+ str(normal_inner_similarity))
    print("normal_outer_similarity" +str(normal_outer_similarity))
    print("avg_normal_inner_similarity "+ str(np.mean(normal_inner_similarity)))
    print("avg_normal_outer_similarity" +str(np.mean(normal_outer_similarity)))
    print("anomalous_inner_similarity "+str(anomalous_inner_similarity))
    print("anomalous_outer_similarity "+str(anomalous_outer_similarity))



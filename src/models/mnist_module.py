

from typing import Any, List

import numpy
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
import scipy.stats
import numpy as np
import n_sphere
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute,dual_annealing
import time
import torch.nn as nn
from sklearn.mixture import BayesianGaussianMixture,GaussianMixture
from scipy.stats import vonmises,norm,multivariate_normal

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
        backbone,pretrained, label, lr,batch_size,angular,MSCL,vanilla
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
        self.vae = False
        self.mu = nn.Linear(512, 512)
        self.var = nn.Linear(512,512)
        self.automatic_optimization = False
        self.sum_mu = []
        self.sum_sig = []
        self.vae2 = True
        self.multi_univariate = False
        self.multi_t = False
        self.dir_pro = True
        self.vonmises = False
        import random
        self.tau = random.random()
        self.diagvib_framework = True
        self.panda = self.hparams.MSCL!=True and not self.hparams.vanilla
        self.MSCL = self.hparams.MSCL==True
        self.tau = .001


    def training_step(self, batch: Any, batch_idx: int):
        self.model.eval()#Unsure
        if self.MSCL:
            loss = self.run_epoch(batch)
        elif self.panda:
            loss = self.run_panda(batch)
        else:
            loss = 0
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
            self.total_loss,self.total_num = 0,1
            return
        # log train metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        #return {"loss": loss}
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

    def training_epoch_end(self, outputs: List[Any]):

        loss = self.total_loss / self.total_num
        self.total_loss, self.total_num = 0.0, 0
        self.loss = loss
      #  self.log("train/loss", loss, on_epoch=True, prog_bar=True)


    def validation_step(self, batch: Any, batch_idx: int):
        if self.first_epoch:
            x, y = batch[:2]
            features = self.model(x)
            self.train_feature_space.append(features)
            self.val_labels.append(y)

        if self.trainer.max_epochs - 1 == self.current_epoch:
                x, y = batch[:2]
                features = self.model(x)
                self.val_feature_space.append(features)
                self.val_labels.append(y)


                if self.vae:
                    mu,sig = self.encode(features)
                    self.sum_mu += [mu]
                    self.sum_sig += [sig]



    def validation_epoch_end(self, outputs: List[Any]):
        if self.first_epoch:
            self.train_feature_space = torch.cat(self.train_feature_space, dim=0).contiguous().cpu().numpy()
            val_labels = torch.cat(self.val_labels, dim=0).cpu().numpy()
            idx = np.array(val_labels) == 0

            if(self.printing_cosine_similarity_experiment):
                Joao_similarity(self.train_feature_space[idx],val_labels[idx],self.train_feature_space[~idx],0,"before_training")
#            self.center,_ = torch.FloatTensor(self.train_feature_space).median(dim=0)
            self.train_feature_space = self.train_feature_space[idx]
            self.center = torch.FloatTensor(self.train_feature_space).mean(dim=0)
            if self.hparams.angular:
                self.center = F.normalize(self.center, dim=-1)
            self.center = self.center.to(self.device)
            self.first_epoch = False
            self.val_labels = []
            if self.panda:
                self.criterion = CompactnessLoss(self.center)


        if self.trainer.max_epochs - 1 == self.current_epoch:
            self.treated_val_feature_space = torch.cat(self.val_feature_space, dim=0).contiguous().cpu().numpy()
            val_labels = torch.cat(self.val_labels, dim=0).cpu().numpy()
            idx = np.array(val_labels) == 0
            if(self.printing_cosine_similarity_experiment):
                Joao_similarity(self.treated_val_feature_space[idx],val_labels[idx],self.treated_val_feature_space[~idx],0,"after_training")
            self.treated_val_feature_space = self.treated_val_feature_space[idx]
          #  print(np.min(np.linalg.norm(self.treated_val_feature_space, -1)))
          #  print(np.max(np.linalg.norm(self.treated_val_feature_space, -1)))
           # print(self.treated_val_feature_space @ self.treated_val_feature_space.T)
           # print(self.treated_val_feature_space @ self.treated_val_feature_space.T)
            self.normaliser = np.mean(self.treated_val_feature_space,axis=0)
            data = self.treated_val_feature_space #- self.normaliser
            self.weight = get_pca(data)

            self.dpgmm = []
#                self.dpgmm.append(BayesianGaussianMixture(n_components=20,n_init = 1, max_iter =100,covariance_type = 'tied',weight_concentration_prior_type = "dirichlet_process").fit(data))
#                data = theta(data)#[:,1:]
#                n = get_pca(data)
            n,data1,mean = GDA(data)
            self.mean = mean
            self.proj = lambda x :  x- (np.dot(x, n) / np.sqrt(sum(n**2)) ** 2).reshape(-1,1) * n.reshape(1,x.shape[1])
           # proj_data = (np.dot(data, n) / np.sqrt(sum(n**2)) ** 2).reshape(-1,1) * n.reshape(1,512)
           # print(proj_data.shape)
           # proj_data2 = data - proj_data
            data = self.proj(data)
#            self.dpgmm.append(BayesianGaussianMixture(n_components=100,n_init = 2, max_iter =1000, covariance_type = 'full').fit(data))
            self.dpgmm.append(BayesianGaussianMixture(n_components=int(min(len(data),data.shape[1])),n_init = 1, max_iter =100, covariance_type = 'full').fit(data))
            likelihood_mono_train = np.mean(self.dpgmm[0].score_samples(data))
           # likelihood_dual_train = np.mean(self.dpgmm[1].score_samples(data))
           # likelihood_trial_train = np.mean(self.dpgmm[2].score_samples(data))
            print("likelihood_train_mono" + str(likelihood_mono_train))
           # print("likelihood_train_dual" + str(likelihood_dual_train))
           # print("likelihood_train_trial" + str(likelihood_trial_train))





        self.log("val/acc", -self.loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        x, y = batch[:2]
        features = self.model(x)
        self.test_feature_space.append(features)
        self.test_labels.append(y)

    def test_epoch_end(self, outputs: List[Any]):
        self.treated_test_feature_space = torch.cat(self.test_feature_space, dim=0).contiguous().cpu().numpy()
        test_labels = torch.cat(self.test_labels, dim=0).cpu().numpy()
        test_data = self.treated_test_feature_space #- self.normaliser


        distances = knn_score(self.treated_val_feature_space, self.treated_test_feature_space)
        print("\nknn")
        evaluate_gen_short(test_labels,distances,  self.diagvib_framework)
        auc = roc_auc_score1(test_labels, distances)
   #     print(auc)



        distances = mahalanobis_score(self.treated_val_feature_space,self.treated_test_feature_space)
        print("\nmahalanobis")
        evaluate_gen_short(test_labels,distances,  self.diagvib_framework)
        auc = roc_auc_score1(test_labels, distances)
#        print(auc)



        print("\nGDA+GMM")
       # test_data = theta(test_data)#[:, 1:]
        _,test_data1,_ = GDA(test_data,self.mean)
        test_data = self.proj(test_data)
    #    test_data = np.hstack((np.ones((test_data.shape[0], 1)), test_data))
    #    test_data = n_sphere.convert_rectangular(test_data)
        idx = (test_labels == 0)
        likelihood_mono_norm = np.mean(self.dpgmm[0].score_samples(test_data[idx]))
   #     likelihood_dual_norm = np.mean(self.dpgmm[1].score_samples(test_data[idx]))
   #     likelihood_trial_norm = np.mean(self.dpgmm[2].score_samples(test_data[idx]))
        print("likelihood_norm" + str(likelihood_mono_norm))
   #     print("likelihood_dual" + str(likelihood_dual_norm))
   #     print("likelihood_trial" + str(likelihood_trial_norm))
        likelihood_mono_an = np.mean(self.dpgmm[0].score_samples(test_data[~idx]))
   #     likelihood_dual_an = np.mean(self.dpgmm[1].score_samples(test_data[~idx]))
   #     likelihood_trial_an = np.mean(self.dpgmm[2].score_samples(test_data[~idx]))
        print("likelihood_anomal" + str(likelihood_mono_an))
    #    print("likelihood_dual" + str(likelihood_dual_an))
    #    print("likelihood_trial" + str(likelihood_trial_an))

        a = self.dpgmm[0]
        pdf = a.score_samples(test_data)
        evaluate_gen_short(test_labels, - pdf.transpose(), self.diagvib_framework)
        auc = roc_auc_score(test_labels, - pdf.transpose())

        self.log("test/acc", auc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.SGD(params=self.parameters(), lr=self.hparams.lr, weight_decay=0.00005)

    def run_epoch(self, batch):
        (img1, img2), y = batch[:2]
    #    self.optimizer.zero_grad()
        out_1 = self.model(img1)
        out_2 = self.model(img2)
        out_1 = out_1 - self.center
        out_2 = out_2 - self.center
        loss = 0
        loss += contrastive_loss(out_1, out_2)# * 1e-10
        if self.hparams.angular:
            loss += ((out_1 ** 2).sum(dim=1).mean() + (out_2 ** 2).sum(dim=1).mean())# * 1e-10
        self.total_num += img1.size(0)
        self.total_loss += loss.item() * img1.size(0)
        return loss

    def run_panda(self, batch):
        img1, y = batch[:2]
    #    self.optimizer.zero_grad()
        out_1 = self.model(img1)
        loss = self.criterion(out_1)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e-3)

        self.total_num += img1.size(0)
        self.total_loss += loss.item() * img1.size(0)
        return loss




import torch
import torch.nn as nn

class CompactnessLoss(nn.Module):
    def __init__(self, center):
        super(CompactnessLoss, self).__init__()
        self.center = center

    def forward(self, inputs):
        m = inputs.size(1)
        variances = (inputs - self.center).norm(dim=1).pow(2) / m
        return variances.mean()


class EWCLoss(nn.Module):
    def __init__(self, frozen_model, fisher, lambda_ewc=1e4):
        super(EWCLoss, self).__init__()
        self.frozen_model = frozen_model
        self.fisher = fisher
        self.lambda_ewc = lambda_ewc

    def forward(self, cur_model):
        loss_reg = 0
        for (name, param), (_, param_old) in zip(cur_model.named_parameters(), self.frozen_model.named_parameters()):
            if 'fc' in name:
                continue
            loss_reg += torch.sum(self.fisher[name]*(param_old-param).pow(2))/2
        return self.lambda_ewc * loss_reg



def GDA(y,mean=None):
    def geodesic_distance(x,y):
        dot_product = np.sum(x*y)
        mag_x = np.linalg.norm(x)
        mag_y = np.linalg.norm(y)
        cosine = dot_product/(mag_x*mag_y)
        if cosine>1: cosine = 1
        if cosine<-1: cosine = -1
        return np.arccos(cosine)

    def log_map(x,y):
        d = geodesic_distance(x,y)
        temp = y - np.sum(x*y) * x
        if np.linalg.norm(temp) != 0:
            mapped_value = d * (temp/np.linalg.norm(temp))
        else:
            mapped_value = np.array([0.0,0.0,0.0])
        return mapped_value

    def exp_map(p,v):
        mag_v = np.linalg.norm(v)
        if mag_v == 0:
            return p
        v_normalized = v/mag_v
        mapped_value = p * np.cos(mag_v) + v_normalized * np.sin(mag_v)
        return mapped_value

    def parallel_transport(v,p,q):
        logmap1 = log_map(p,q)
        logmap2 = log_map(q,p)
        if np.linalg.norm(logmap1)!=0 and np.linalg.norm(logmap2)!=0:
            transported_value = v - (np.dot(logmap1 , v)/geodesic_distance(p,q)) * (logmap1+logmap2)
        else:
            transported_value = v
        return transported_value

    def calculate_mean(data):
        iter = 50
        lr = 0.01
        mean = np.ones(data.shape[1])/2
        for i in range(iter):
            grad = 0
            for j in range(data.shape[0]):
                grad -= log_map(mean,data[j])
            mean = exp_map(mean, -1*lr*grad)
        return mean
    if mean is None:
        mean = calculate_mean(y)
    mapped_points = np.array([log_map(mean,y[i]) for i in range(len(y))])
    principal_vectors = np.linalg.svd(mapped_points.T)[0]
    magnitudes = np.linalg.svd(mapped_points.T)[1]
    return principal_vectors[0], mapped_points,mean

def evaluate_gen_short(test_idx,predictions,boolean):
    if boolean:
        gen = predictions[test_idx==0]
        short = predictions[test_idx==1]
        median = np.median(predictions)
        #genacc = np.mean(test_idx[test_idx==0] == (gen>median).astype(int))
        #shortacc = np.mean(test_idx[test_idx==1] == (short>median).astype(int))
        genauc = roc_auc_score1(test_idx[predictions<=median], predictions[predictions<=median])
        shortauc = roc_auc_score1(test_idx[predictions>median], predictions[predictions>median])

        p = np.sum(test_idx)
        n = len(test_idx) - np.sum(test_idx)
        tp = np.sum(predictions[predictions>median])
        tn = len(predictions[predictions<=median]) - np.sum(predictions[predictions<=median])
        fp = len(predictions[predictions>median]) - np.sum(predictions[predictions>median])
        fn = np.sum(predictions[predictions<=median])
        tpr = tp/p
        tnr = tn/n
        ppv=tp/(tp+fp)
        npv=tn/(tn+fn)
        fnr = fn/p
        fpr = fp/n
        fdr = fp/(fp+tp)
        for_ = fn/(fn+tn)
        lrp = tpr/fpr
        lrm = fnr/tnr
        acc = (tp+tn)/(p+n)
        dor = lrp/lrm
        print(["tp","tn","fp","fn","tpr","tnr","ppv","npv","fnr","fpr","fdr","for_","lrp","lrm","acc","dor"])
        print([tp,tn,fp,fn,tpr,tnr,ppv,npv,fnr,fpr,fdr,for_,lrp,lrm,acc,dor])


        plt.clf()
    #    print(test_idx)
    #    print([np.argsort(predictions)])
    #    print(test_idx[np.argsort(predictions)])
        plt.plot(range(len(predictions)),test_idx[np.argsort(predictions)],"o")
        plt.vlines(len(predictions)/2,0,1)
        plt.savefig(str(time.time())+".pdf")


     #   print("generalisation score: " + str(genauc))
     #   print("shortcut resistance score: " + str(shortauc))
        print("auc: " + str(roc_auc_score(test_idx, predictions)))
    return

def get_pca(x):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    pca.fit(x)
    print(pca.components_[0].shape)

    return pca.components_[0]

def preprocess(x):
    epsilon = 1e-8
    min = x.min(1).reshape(-1, 1)
    large = (x + epsilon - min).astype(np.float64)
    norm = large.sum(1)
    scaled = (large / norm.reshape(-1, 1).astype(np.float64))
    return scaled


def roc_auc_score1(a,b):
    from sklearn.metrics import RocCurveDisplay
    import matplotlib.pyplot as plt
    RocCurveDisplay.from_predictions(a,b)
    plt.savefig(str(time.time())+".pdf")
    return roc_auc_score(a,b)



def mahalanobis_score(train_set, test_set, n_neighbours=2):
    from sklearn.covariance import EmpiricalCovariance, MinCovDet
    """
    Calculates the KNN distance
    """
    emp_cov = EmpiricalCovariance().fit(train_set)

    return emp_cov.mahalanobis(test_set)


def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)

def prob(args,test_set,distribution):

    from torch.distributions.multivariate_normal import MultivariateNormal
    multivariate = distribution(*args)
    return 1-multivariate.log_prob(test_set)
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

        if False:
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

        if True:
            anomalous_outer_similarity = 0
            label_data = normal_data

            distances = dist_metric(anomalous_data, label_data)
            distances = np.mean(distances,axis = 1)
            distance_idx = np.argsort(distances)
            if (i == 1):
                distance_idx = distance_idx[::-1]
            anomalous_data = anomalous_data[distance_idx]
            for anomalous_sample in f(anomalous_data):
                distances = dist_metric([anomalous_sample], label_data)
                distance_idx = np.argsort(distances)[0]
                if (i == 1):
                    distance_idx = distance_idx[::-1]
                all_closest = g(label_data[distance_idx])
                anomalous_outer_similarity += np.mean(dist_metric(all_closest, [anomalous_sample]))
            print("similarity between 10% hardest anomalies and closest normal samples using distance metric " + str(dist_metric) +
                          " : " + str(anomalous_outer_similarity/len(f(anomalous_data))))

        else:
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



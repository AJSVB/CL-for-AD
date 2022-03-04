import os
os.system("conda env create -f environment.yml")
os.system("source activate diagvibsix")
os.system("pip install -e .")
#os.system("python diagvibsix/generate_study/generate_studies_ZSO_ZGO_FGO.py")
#os.system("python diagvibsix/generate_study/generate_studies_CGO-123.py")
#os.system("python diagvibsix/generate_study/generate_studies_CHGO.py")
os.system("python diagvibsix/dataset/preprocess_mnist.py")

import os
import time
from argparser import make_parser
import trainer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from diagvibsix import TorchDatasetWrapper
from diagvibsix.auxiliaries import save_obj, save_yaml
from utils.metrics import Losses, Metrics
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
args = Namespace(adam_b1=0.9, adam_b2=0.999, cache=False, class_criterion='ce', dataset_sample='0',
                 dataset_seed=1332, device=-1, experiment='CORR_PRED-shape', lr=0.0001,
                 mbs=128, method='ResNet18Trainer', num_epochs=1, num_workers=0, optimizer='adam',
                 results_path='tmp/results', sgd_dampening=0.0, sgd_momentum=0.0, study='study_ZSO',
                 study_folder='tmp/diagvibsix/studies', training_seed=1332)        
method_trainer = getattr(trainer, args.method)
this_trainer = method_trainer(args)

torch.save(this_trainer.data_loader['train'], 'data_train.pt')
torch.save(this_trainer.data_loader['val'], 'data_val.pt')
torch.save(this_trainer.data_loader['test'], 'data_test.pt')




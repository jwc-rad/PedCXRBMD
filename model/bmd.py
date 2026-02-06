import copy
import itertools
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc, RocCurveDisplay

from omegaconf import OmegaConf
from hydra.utils import instantiate

import torch
import torch.nn.functional as F
import wandb

from networks.utils import load_pretrained_net

from .regression import ImageRegressionModel

class BMDRegModel(ImageRegressionModel):
    ### custom methods
    def define_networks(self, **opt):
        if self.is_train:
            self.net_names = ['netB','netC']
        else:
            self.net_names = ['netB','netC']
                    
        for net in self.net_names:
            setattr(self, net, instantiate(OmegaConf.select(opt['networks'], net), _convert_='partial'))
            pretrained = OmegaConf.select(opt['networks'], f'pretrained.{net}')
            if pretrained:
                snet = getattr(self, net)
                snet = load_pretrained_net(snet, pretrained)
                                   
    def define_etc(self, **opt):   
        pass
                 
    def set_input(self, batch):
        self.image = batch['image']
        self.table = batch['table'].float()
        if 'label' in batch.keys():
            self.label = self.pp_label = batch['label'].float() # shape (batch)            
            
    def convert_output(self, x):
        return x
    
    def _step_forward(self, batch, batch_idx):
        self.set_input(batch)
        
        self.image_cls = self._forward_train(self.image, self.table)
        
    def _step_forward_infer(self, batch, batch_idx):
        self.set_input(batch)
        
        outputs = self.forward(self.image, self.table)
        return outputs
      
    def _forward_train(self, x1, x2):
        x1 = self.netB(x1)
        out = self.netC(x1, x2)
        return out
      
    ### pl methods
    def forward(self, x1, x2):
        x1 = self.netB(x1)
        out = self.netC(x1, x2)
        return out

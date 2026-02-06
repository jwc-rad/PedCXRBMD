import copy
import itertools
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc, RocCurveDisplay

from omegaconf import OmegaConf
from hydra.utils import instantiate

import torch
import lightning.pytorch as pl
import wandb

from monai.metrics import ROCAUCMetric, ConfusionMatrixMetric, get_confusion_matrix, compute_confusion_matrix_metric
from monai.networks.utils import one_hot

from networks.utils import load_pretrained_net
from .utils import instantiate_scheduler

class BaseModel(pl.LightningModule):           
    def load_pretrained(self, path):
        self.load_pretrained_nets(path, nets=self.net_names)
    
    def load_pretrained_nets(self, path, nets=[]):
        '''For loading state_dict of part of the model.
        Loading full model should be done by "load_from_checkpoint" (Lightning)
        '''
        
        device = next(self.parameters()).device
        
        # load from checkpoint or state_dict
        print(f'trying to load pretrained from {path}')
        try:
            state_dict = torch.load(path, map_location=device)['state_dict']
        except:
            state_dict = torch.load(path, map_location=device)
        
        if len(nets)==0:
            self.load_state_dict(state_dict)
        
        all_keys_match = True
        changed = False
        for name in nets:
            if hasattr(self, name):
                net = getattr(self, name)
                new_weights = net.state_dict()
                
                # first check if pretrained has all keys
                keys_match = True
                for k in new_weights.keys():
                    if not f'{name}.{k}' in state_dict.keys():
                        keys_match = False
                        all_keys_match = False
                        print(f"not loading {name} because keys don't match")
                        break
                if keys_match:
                    for k in new_weights.keys():
                        new_weights[k] = state_dict[f'{name}.{k}']
                    net.load_state_dict(new_weights)
                    changed = True
                        
        if changed:
            if all_keys_match:
                print('<All keys matched successfully>')
        else:
            print(f'nothing is loaded from {path}')

class ImageRegressionModel_Base(BaseModel):
    def __init__(self, **opt):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.is_train = opt['train']
        self.use_wandb = 'wandb' in opt['logger']
        
        # define networks
        self.define_networks(**opt)
        
        # define loss functions
        self.define_losses(**opt)

        # define inferer
        if 'inferer' in opt:
            self.inferer = instantiate(opt['inferer'], _convert_='partial')
            
        # define metrics
        if 'metrics' in opt and opt['metrics'] is not None:
            self.metrics = {}
            for k, v in opt['metrics'].items():
                if '_target_' in v:
                    self.metrics[k] = instantiate(v, _convert_='partial')
                    
        self.define_etc(**opt)
                    
    ### custom methods
    
    def define_networks(self, **opt):
        if self.is_train:
            self.net_names = ['netC']
        else:
            self.net_names = ['netC']
                    
        for net in self.net_names:
            setattr(self, net, instantiate(OmegaConf.select(opt['networks'], net), _convert_='partial'))
            pretrained = OmegaConf.select(opt['networks'], f'pretrained.{net}')
            if pretrained:
                snet = getattr(self, net)
                snet = load_pretrained_net(snet, pretrained)
                
    def define_losses(self, **opt):
        if self.is_train:            
            self.criterionCls = instantiate(opt['loss_cls'], _convert_='partial')        
                   
    def define_etc(self, **opt):   
        pass
                 
    def set_input(self, batch):
        self.image = batch['image']
        if 'label' in batch.keys():
            self.label = self.pp_label = batch['label']#.float()
            
    def convert_output(self, x):
        return x
    
    def _step_forward(self, batch, batch_idx):
        self.set_input(batch)
        
        self.image_cls = self.forward(self.image)
        
    def _step_forward_infer(self, batch, batch_idx):
        self.set_input(batch)
        
        outputs = self.inferer(self.image, self.forward)
        return outputs
      
    ### pl methods
    def forward(self, x):
        out = self.netC(x)
        return out
    
    def configure_optimizers(self):        
        netparams = [getattr(self, n).parameters() for n in self.net_names if hasattr(self, n)]
        optimizer_GF = instantiate(self.hparams['optimizer'], params=itertools.chain(*netparams))
        
        optimizers = [optimizer_GF]
        schedulers = [{
            k: instantiate_scheduler(optimizer, v) if k=='scheduler' else v 
            for k,v in self.hparams['scheduler'].items()
        } for optimizer in optimizers]
        
        return optimizers, schedulers
    
    def validation_step(self, batch, batch_idx):
        stage = 'valid'
        outputs = self._step_forward_infer(batch, batch_idx)
        
        bs = self.image.size(0)  
        loss = self.criterionCls(outputs, self.label)

        if hasattr(self, 'metrics'):
            pp_outputs = self.convert_output(outputs)
            for k in self.metrics.keys():
                self.metrics[k](pp_outputs.float(), self.pp_label.float())
                if self.global_step == 0 and self.use_wandb:
                    for x in k.split('__'):
                        wandb.define_metric(f'metrics/valid_{x}', summary='max')
                    
        self.log(f'loss/{stage}', loss, batch_size=bs, on_step=True, on_epoch=True)
                    
        return loss
    
    def on_validation_epoch_end(self):
        if not hasattr(self, 'metrics'):
            return
    
        for k in self.metrics.keys():
            if self.metrics[k].get_buffer() is not None:
                mean_metric = self.metrics[k].aggregate()
                if isinstance(mean_metric, list):
                    kks = k.split('__')
                    for i in range(len(mean_metric)):
                        mmetric = mean_metric[i].item()
                        self.log(f'metrics/valid_{kks[i]}', mmetric)                    
                else:
                    mean_metric = mean_metric.item()
                    self.log(f'metrics/valid_{k}', mean_metric)
            self.metrics[k].reset()
            
    def predict_step(self, batch, batch_idx):
        outputs = self._step_forward_infer(batch, batch_idx)
        self.outputs = outputs
        self.pp_outputs = self.convert_output(outputs)
        return self.pp_outputs
    
    def test_step(self, batch, batch_idx):
        outputs = self._step_forward_infer(batch, batch_idx)
        self.outputs = outputs
        
        if hasattr(self, 'metrics'):
            pp_outputs = self.convert_output(outputs)
            for k in self.metrics.keys():
                self.metrics[k](pp_outputs.float(), self.pp_label.float())
        return None
    
    def on_test_epoch_end(self):
        if not hasattr(self, 'metrics'):
            return
    
        for k in self.metrics.keys():
            if self.metrics[k].get_buffer() is not None:
                mean_metric = self.metrics[k].aggregate()
                if isinstance(mean_metric, list):
                    kks = k.split('__')
                    for i in range(len(mean_metric)):
                        mmetric = mean_metric[i].item()
                        self.log(f'test_metrics/{kks[i]}', mmetric)                    
                else:
                    mean_metric = mean_metric.item()
                    self.log(f'test_metrics/{k}', mean_metric)
            self.metrics[k].reset()    

class ImageRegressionModel(ImageRegressionModel_Base): 
    def training_step(self, batch, batch_idx):
        stage = 'train'
        self._step_forward(batch, batch_idx)
        
        loss = 0
        
        # Labeled
        bs = self.image.size(0)                               
        # Classification loss: S(A) ~ Ya
        w0 = self.hparams['lambda_cls']
        if w0 > 0:            
            loss_S = self.criterionCls(self.image_cls, self.label)
            self.log('loss/cls', loss_S, batch_size=bs, on_step=True, on_epoch=True)
        else:
            loss_S = 0
        loss += loss_S * w0  

        # 
        #self.log(f'loss/{stage}', loss, batch_size=bs, on_step=True, on_epoch=True)

        return loss

class ImageRegressionModel_SigmoidOutput(ImageRegressionModel): 
    def convert_output(self, x):
        return torch.sigmoid(x)
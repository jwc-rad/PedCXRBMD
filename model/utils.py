from typing import Any, Callable, Dict, List

from hydra.utils import instantiate, get_class
from omegaconf import DictConfig, OmegaConf
from torch.optim import lr_scheduler

def instantiate_scheduler(optimizer, cfg: DictConfig):
    def add_optimizer(optimizer, cfg):
        if isinstance(cfg, Dict):
            for k, v in cfg.items():
                cfg[k] = add_optimizer(optimizer, v)
            if '_target_' in cfg:
                if issubclass(get_class(cfg.get('_target_')), lr_scheduler.LRScheduler):
                    cfg.update(dict(optimizer=optimizer))
            return cfg
        elif isinstance(cfg, List):
            return [add_optimizer(optimizer, x) for x in cfg]
        else:
            return cfg
    
    _cfg = OmegaConf.to_container(cfg, resolve=True)
    _cfg = add_optimizer(optimizer, _cfg)
    return instantiate(_cfg)
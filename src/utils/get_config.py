import inspect

from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def get_loss_from_config(config: dict, registry: dict) -> _Loss:
    name = config['name']
    kwargs = config.get('kwargs', {})

    if name not in registry:
        raise ValueError(f"Loss function '{name}' not found in registry.")
    
    criterion = registry[name]
    sig = inspect.signature(criterion.__init__)
    valid_args = {
        k: v for k, v in kwargs.items()
        if k in sig.parameters and k != 'self'
    }
    
    return criterion(**valid_args)


def get_optim_from_config(config: dict, registry: dict, model: nn.Module) -> Optimizer:
    name = config['name']
    kwargs = config.get('kwargs', {})

    if name not in registry:
        raise ValueError(f"Optimizer '{name}' not found in registry.")
    
    optimizer = registry[name]
    sig = inspect.signature(optimizer.__init__)
    valid_args = {
        k: v for k, v in kwargs.items()
        if k in sig.parameters and k != 'self'
    }

    return optimizer(model.parameters(), **valid_args)


def get_scheduler_from_config(config: dict, registry: dict, optimizer: Optimizer) -> _LRScheduler:
    name = config['name']
    kwargs = config.get('kwargs', {})

    if name not in registry:
        raise ValueError(f"Scheduler '{name}' not found in registry.")
    
    scheduler = registry[name]
    sig = inspect.signature(scheduler.__init__)
    valid_args = {
        k: v for k, v in kwargs.items()
        if k in sig.parameters and k != 'self'
    }

    return scheduler(optimizer, **valid_args)
from .ghost_pfld import *
from .pfld import *


__model_factory = {
    # image classification models
    'pfld': PFLDNet,
    'ghost_pfld': GhostPFLDNet,
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError('Unknown model: {}'.format(name))
    return __model_factory[name](*args, **kwargs)

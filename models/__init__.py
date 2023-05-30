from .resnet import resnet18
from .resnets import resnet18 as resnet18s
from .gresnets import resnet18 as gresnet18s

def get_model(config):
    if config.arch == "resnet18s":
        return resnet18s(num_classes=config.num_classes)
    if config.arch == "resnet18":
        return resnet18(num_classes=config.num_classes)
    if config.arch == "gresnet18s":
        return gresnet18s(num_classes=config.num_classes)
    raise ValueError("Unknown model: {}".format(config.arch))

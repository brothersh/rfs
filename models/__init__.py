from .convnet import convnet4
from .resnet import resnet12
from .resnet import seresnet12
from .wresnet import wrn_28_10

from .resnet import resnet50

model_pool = [
    'convnet4',
    'resnet12',
    'seresnet12',
    'wrn_28_10',
    'resnet50'
]

model_dict = {
    'wrn_28_10': wrn_28_10,
    'convnet4': convnet4,
    'resnet12': resnet12,
    'seresnet12': seresnet12,
    'resnet50': resnet50,
}

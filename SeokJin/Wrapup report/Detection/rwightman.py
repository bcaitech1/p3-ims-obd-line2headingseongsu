import sys
sys.path.append('/opt/ml/code/pytorch_image_models_master')
import torch.nn as nn
import timm
from ..builder import BACKBONES

@BACKBONES.register_module
class RWightman(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = timm.create_model(**kwargs)

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs

    def init_weights(self, pretrained: bool = True):
        pass

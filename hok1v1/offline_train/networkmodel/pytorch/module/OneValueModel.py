import torch
from torch import nn
import numpy as np
from train_eval_config.OneConfig import DimConfig
from train_eval_config.OneConfig import ModelConfig as Config


class OneValueModel(nn.Module):
    def __init__(self):
        """
        input s, get the feature after encoding
        """
        super(OneValueModel, self).__init__()

        feature_dim = Config.LSTM_UNIT_SIZE
        # feature_dim = Config.SERI_VEC_SPLIT_SHAPE[0][0]

        """
            network parameters
        """
        self.value = nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

        self.device = None

    def forward(self, state_feature):
        # input = state_feature.detach() ### bs x state_dim ###
        input = state_feature  ### bs x state_dim ###
        return self.value(input)

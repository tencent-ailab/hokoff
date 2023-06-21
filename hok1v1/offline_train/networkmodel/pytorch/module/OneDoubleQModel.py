import torch
from torch import nn
import numpy as np
from train_eval_config.OneConfig import DimConfig
from train_eval_config.OneConfig import ModelConfig as Config


class OneDoubleQModel(nn.Module):
    def __init__(self):
        """
        input s, get the feature after encoding
        """
        super(OneDoubleQModel, self).__init__()

        feature_dim = Config.LSTM_UNIT_SIZE
        # feature_dim = Config.SERI_VEC_SPLIT_SHAPE[0][0]
        label_size_list = Config.LABEL_SIZE_LIST
        action_dim = sum(label_size_list)

        """
            network parameters
        """
        self.q1 = nn.Sequential(nn.Linear(feature_dim + action_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        self.q2 = nn.Sequential(nn.Linear(feature_dim + action_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

        self.device = None

    def forward(self, state_feature, action):
        # input = torch.cat([state_feature.detach(), action], dim = 1) ### bs x (state_dim + action_dim) ###
        input = torch.cat([state_feature, action], dim=1)  ### bs x (state_dim + action_dim) ###
        q1, q2 = self.q1(input), self.q2(input)
        return q1, q2

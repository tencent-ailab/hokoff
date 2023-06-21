import torch as th  # in place of tensorflow
import torch.nn as nn  # for builtin modules including Linear, Conv2d, MultiheadAttention, LayerNorm, etc
from torch.nn import ModuleDict  # for layer naming when nn.Sequential is not viable
import numpy as np  # for some basic dimention computation, might be redundent

from math import ceil, floor
from collections import OrderedDict

# typing
from torch import Tensor, LongTensor
from typing import Dict, List, Tuple
from ctypes import Union

from train_eval_config.Config import Config
from train_eval_config.DimConfig import DimConfig
from networkmodel.pytorch.module.BaseModel import BaseModel as Model
import time

# class Mixer(nn.Module):
#     def __init__(self) -> None:
#         super(Mixer,self).__init__()


class BCLearner:
    def __init__(self, args):
        super(BCLearner, self).__init__()
        # feature configure parameter
        if 'ind' in args.run_prefix:
            from networkmodel.pytorch.module.IndModel import IndModel as Model

            self.ind = True
        else:
            from networkmodel.pytorch.module.BaseModel import BaseModel as Model

            self.ind = False
        self.args = args
        self.model_name = Config.NETWORK_NAME
        self.lstm_time_steps = args.lstm_time_steps
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.target_embedding_dim = Config.TARGET_EMBEDDING_DIM
        self.hero_data_split_shape = Config.HERO_DATA_SPLIT_SHAPE
        self.hero_seri_vec_split_shape = Config.HERO_SERI_VEC_SPLIT_SHAPE
        self.hero_feature_img_channel = Config.HERO_FEATURE_IMG_CHANNEL
        self.hero_label_size_list = Config.HERO_LABEL_SIZE_LIST
        self.hero_is_reinforce_task_list = Config.HERO_IS_REINFORCE_TASK_LIST

        self.learning_rate = args.lr
        self.var_beta = Config.BETA_START

        self.clip_param = Config.CLIP_PARAM
        self.restore_list = []
        self.min_policy = Config.MIN_POLICY
        self.embedding_trainable = False
        self.value_head_num = Config.VALUE_HEAD_NUM

        self.hero_num = 3
        self.hero_data_len = sum(Config.data_shapes[0])
        self.online_net = Model()
        self.optimizer = th.optim.Adam(params=self.online_net.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8)
        self.local_steps = 0
        self.update_target_net()

    def update_target_net(self):
        pass

    def compute_loss(self, data_dict):
        obs_s = data_dict['observation_s']  # torch.Size([64, 16, 3, 4586])
        legal_action_s = data_dict['legal_action_s']  # torch.Size([64, 16, 3, 161])
        action_s = data_dict['action_s']  # torch.Size([64, 16, 3, 5])
        sub_action_s = data_dict['sub_action_s']  # torch.Size([64, 16, 3, 5])
        local_q_list, _, _, _ = self.online_net(obs_s, only_inference=False)

        bc_error = 0
        local_q_taken_list = []
        for agent_idx in range(self.hero_num):
            agent_action = action_s[:, :, agent_idx : agent_idx + 1]  # bs,t,1,5
            agent_sub_action = sub_action_s[:, :, agent_idx : agent_idx + 1]  # bs,t,1,5
            agent_local_q_list = local_q_list[agent_idx]
            split_agent_legal_action = th.split(legal_action_s[:, :, agent_idx : agent_idx + 1], self.hero_label_size_list[0], dim=-1)
            for label_index, label_dim in enumerate(self.hero_label_size_list[agent_idx]):
                local_logits = agent_local_q_list[label_index] - (1 - split_agent_legal_action[label_index]) * 1e10
                softmax_prob = th.softmax(local_logits - th.max(local_logits, dim=-1, keepdim=True)[0], -1)
                chosen_label_agent_local_q = th.gather(
                    softmax_prob, dim=-1, index=agent_action[:, :, :, label_index : label_index + 1].long()
                )  # bs,t,1,1
                local_q_taken_list.append(chosen_label_agent_local_q * agent_sub_action[:, :, :, label_index : label_index + 1])
                log_p = th.log(chosen_label_agent_local_q + 0.00001) * agent_sub_action[:, :, :, label_index : label_index + 1]  # bs,t,1,1
                bc_error = bc_error - th.mean(log_p)
        loss = bc_error
        return loss, {'prob_taken': th.sum(th.stack(local_q_taken_list)) / th.sum(sub_action_s)}

    def to(self, device):
        self.online_net.to(device)
        # self.target_net.to(device)

    def state_dict(self):
        return self.online_net.state_dict()

    def load_state_dict(self, state_dict):
        self.online_net.load_state_dict(state_dict)
        # self.target_net.load_state_dict(state_dict)

    def parameters(self):
        return self.online_net.parameters()

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()

    def step(self, data_dict):
        before_step = time.time()
        self.optimizer.zero_grad()
        with th.cuda.amp.autocast():
            loss, info = self.compute_loss(data_dict)
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(list(self.online_net.parameters()), 10)
        self.optimizer.step()
        info['step_time'] = time.time() - before_step
        return loss, info

    def save_dict(self):
        save_dict = {
            "network_state_dict": self.online_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        return save_dict

    def load_save_dict(self, save_dict):
        self.online_net.load_state_dict(save_dict['network_state_dict'])
        self.optimizer.load_state_dict(save_dict['optimizer_state_dict'])
        self.update_target_net()

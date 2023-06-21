import torch as th
import torch.nn.functional as F
import torch.nn as nn
from train_eval_config.OneConfig import ModelConfig as Config
import time
import copy
import numpy as np

### 1v1 Base Encoder Model ###
from .module.OneBaseModel import OneBaseModel
from .module.OneDoubleQModel import OneDoubleQModel
from .module.OneValueModel import OneValueModel


class OneIQLLearner:
    def __init__(self, args):
        super(OneIQLLearner, self).__init__()

        self.args = args
        self.model_name = Config.NETWORK_NAME
        self.lstm_time_steps = args.lstm_time_steps
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE

        self.learning_rate = args.lr
        self.gamma = args.gamma

        self.state_dim = Config.SERI_VEC_SPLIT_SHAPE[0][0]
        self.label_size_list = Config.LABEL_SIZE_LIST

        self.batch_size = args.batch_size * args.lstm_time_steps

        ### online net ###
        self.online_net = OneBaseModel()
        self.online_double_q = OneDoubleQModel()
        self.online_value = OneValueModel()

        ### target net ###
        self._copy_target_net()
        self.tau = args.tau
        self.tau_iql = args.iql_tau
        self.beta = args.iql_beta
        self.max_advantage_clip = args.iql_max_advantage_clip

        self.mse = nn.MSELoss()

        print('#' * 50)
        print(len(list(self.online_net.parameters())) + len(list(self.online_double_q.parameters())) + len(list(self.online_value.parameters())))
        print('#' * 50)

        self.actor_optimizer = th.optim.Adam(params=self.online_net.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8)
        self.q_optimizer = th.optim.Adam(params=self.online_double_q.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8)
        self.value_optimizer = th.optim.Adam(params=self.online_value.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8)

        self.use_sub_action = args.use_sub_action

    def _copy_target_net(self):
        self.target_double_q = copy.deepcopy(self.online_double_q).requires_grad_(False)

    def update_target_net(self):
        ### update target q network ###
        for param, target_param in zip(self.online_double_q.parameters(), self.target_double_q.parameters()):
            target_param.data.copy_((1 - self.tau) * param.data + self.tau * target_param.data)

    def compute_loss(self, data_dict):
        obs = data_dict['observation'].squeeze()  # bs x 725
        next_obs = data_dict['next_observation'].squeeze()  # bs x 725
        legal_action = data_dict['legal_action'].squeeze()  # bs x 172
        action = data_dict['action'].squeeze()  # bs x 6
        reward = data_dict['reward'].squeeze().reshape([self.batch_size, -1])  # bs x 1
        ### for target q ###
        done = data_dict['done'].squeeze().reshape([self.batch_size, -1])  # bs x 1
        sub_action_mask = data_dict['sub_action'].squeeze()  # bs x 6

        obs_feature, logits = self.online_net(obs, only_inference=False)
        next_obs_feature, _ = self.online_net(next_obs, only_inference=False)

        """
            caculate target_q and target_value
        """
        value = self.online_value(obs_feature)
        with th.no_grad():
            split_action = th.split(action, [1] * len(self.label_size_list), dim=1)
            cur_l_onehot_actions = []
            for ii in range(len(split_action)):
                cur_l_onehot_actions.append(F.one_hot(split_action[ii].long().squeeze(), num_classes=Config.LABEL_SIZE_LIST[ii]))
            replay_onehot_action = th.cat(cur_l_onehot_actions, dim=1)
            target_q1, target_q2 = self.target_double_q(obs_feature, replay_onehot_action)
            target_q = th.min(target_q1, target_q2)

            value_s1 = self.online_value(next_obs_feature)
            target_value = reward + self.gamma * (1 - done) * value_s1

        """
            caculate advantage and value loss
        """
        advantage = target_q - value

        """
            caculate value loss
        """
        para_mul = (self.tau_iql - (advantage < 0).float()).abs()
        value_loss = (para_mul * (advantage**2)).mean()

        """
            caculate Q(s, a) and q loss
        """
        cur_q1, cur_q2 = self.online_double_q(obs_feature, replay_onehot_action)
        q_loss = 0.5 * self.mse(cur_q1, target_value) + 0.5 * self.mse(cur_q2, target_value)

        """
            caculate exp_advantage 
        """
        with th.no_grad():
            exp_advantage = self.beta * advantage
            exp_advantage = exp_advantage.exp().clamp(max=self.max_advantage_clip)

        """
            caculate log_prob --> log pi(a|s)
        """
        policy_action_log_probs = []
        spilit_legal_actions = th.split(legal_action, Config.LEGAL_ACTION_SIZE_LIST, dim=1)
        split_action = th.split(action, [1] * len(self.label_size_list), dim=1)
        for ii in range(len(logits)):
            legal_action_mask = spilit_legal_actions[ii].float()
            if ii == len(logits) - 1:
                # last dim need specific operation
                reshaped_next_legal_action = th.reshape(
                    spilit_legal_actions[ii], [self.batch_size, Config.LABEL_SIZE_LIST[0], Config.LABEL_SIZE_LIST[-1]]
                )  # bs,12,8
                one_hot_actions = th.reshape(
                    F.one_hot(split_action[0].long().squeeze(), num_classes=Config.LABEL_SIZE_LIST[0]),
                    [self.batch_size, Config.LABEL_SIZE_LIST[0], 1],
                )  # bs,12,1
                next_legal_action = th.sum(reshaped_next_legal_action * one_hot_actions, dim=1)  # bs, 8
                legal_action_mask = next_legal_action.float()
            masked_logits = logits[ii] * legal_action_mask - 1.0e15 * (1 - legal_action_mask)

            ### begin softmax ###
            masked_exp_logits = (masked_logits - masked_logits.max(1, keepdims=True).values).exp()
            masked_action_all_probs = masked_exp_logits / masked_exp_logits.sum(1, keepdims=True)
            masked_cur_action_prob = masked_action_all_probs * cur_l_onehot_actions[ii]

            if self.use_sub_action:
                policy_action_log_probs.append(th.log(masked_cur_action_prob.sum(1, keepdims=True) + 1e-5) * sub_action_mask[:, ii : ii + 1])
            else:
                policy_action_log_probs.append(th.log(masked_cur_action_prob.sum(1, keepdims=True) + 1e-5))

            cur_argmax = th.argmax(masked_logits, axis=1)[:, None]
            if ii == 0:
                first_action = th.reshape(cur_argmax, [-1])

        """
            caculate actor loss
        """
        policy_action_log_prob = sum(policy_action_log_probs)
        actor_loss = -exp_advantage * policy_action_log_prob  ### 128 x 1 ###
        actor_loss = actor_loss.mean()

        return (actor_loss, q_loss, value_loss), {
            'loss/q_loss': q_loss,
            'loss/actor_loss': actor_loss,
            'loss/value_loss': value_loss,
            'cur_Q': cur_q1.mean(),
            'value': value.mean(),
            'Q < V': np.count_nonzero((advantage < 0).cpu().data.numpy()) / self.batch_size,
            'advantage_mean': advantage.mean(),
            'log_prob': policy_action_log_prob.mean(),
            'tau': self.tau_iql,
        }

    def to(self, device):
        self.device = device
        self.online_net.device = device
        self.online_double_q.device = device
        self.online_value.device = device
        self.target_double_q.device = device
        self.online_net.to(device)
        self.online_double_q.to(device)
        self.online_value.to(device)
        self.target_double_q.to(device)

    # def state_dict(self):
    #     return self.online_net.state_dict()

    # def load_state_dict(self, state_dict):
    #     self.online_net.load_state_dict(state_dict)
    #     self._copy_target_net()

    # def parameters(self):
    #     return self.online_net.parameters()

    def train(self):
        self.online_net.train()
        self.online_double_q.train()
        self.online_value.train()

    def eval(self):
        self.online_net.eval()

    def step(self, data_dict):
        before_step = time.time()

        with th.cuda.amp.autocast():
            loss, info = self.compute_loss(data_dict)
        actor_loss, q_loss, value_loss = loss

        ### update value network ###
        self.value_optimizer.zero_grad()
        self.q_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        value_loss.backward(retain_graph=True)
        self.value_optimizer.step()

        ### update q network ###
        q_loss.backward(retain_graph=True)
        self.q_optimizer.step()

        ### update actor network ###
        actor_loss.backward()
        self.actor_optimizer.step()

        info['step_time'] = time.time() - before_step
        return actor_loss + q_loss + value_loss, info

    def save_dict(self):
        save_dict = {
            "network_state_dict": self.online_net.state_dict(),
            "optimizer_state_dict": self.actor_optimizer.state_dict(),
            "q_state_dict": self.online_double_q.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
            "value_state_dict": self.online_value.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
        }
        return save_dict

    def load_save_dict(self, save_dict):
        self.online_net.load_state_dict(save_dict['network_state_dict'])
        self.actor_optimizer.load_state_dict(save_dict['optimizer_state_dict'])
        self.online_double_q.load_state_dict(save_dict['q_state_dict'])
        self.q_optimizer.load_state_dict(save_dict['q_optimizer_state_dict'])
        self.online_value.load_state_dict(save_dict['value_state_dict'])
        self.value_optimizer.load_state_dict(save_dict['value_optimizer_state_dict'])
        self._copy_target_net()

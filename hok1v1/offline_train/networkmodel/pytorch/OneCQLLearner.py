import torch as th
import torch.nn.functional as F
import torch.nn as nn
from train_eval_config.OneConfig import ModelConfig as Config
import time
import copy

### 1v1 Base Encoder Model ###
from .module.OneBaseModel import OneBaseModel
from torch.optim.lr_scheduler import CosineAnnealingLR


class OneCQLLearner:
    def __init__(self, args):
        super(OneCQLLearner, self).__init__()

        self.args = args
        self.model_name = Config.NETWORK_NAME
        self.lstm_time_steps = args.lstm_time_steps
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE

        self.learning_rate = args.lr
        self.gamma = args.gamma

        self.state_dim = Config.SERI_VEC_SPLIT_SHAPE[0][0]
        self.label_size_list = Config.LABEL_SIZE_LIST

        self.batch_size = args.batch_size * args.lstm_time_steps

        self.online_net = OneBaseModel()
        ### target net ###
        self._copy_target_net()
        self.mse = nn.MSELoss()
        self.tau = args.tau

        self.cql_alpha = args.cql_alpha

        print('#' * 50)
        print(len(list(self.online_net.parameters())))
        print('#' * 50)
        self.optimizer = th.optim.Adam(params=self.online_net.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8)
        self.lr_schedule = CosineAnnealingLR(self.optimizer, args.max_steps, eta_min=3e-4)

        self.use_sub_action = args.use_sub_action

    def _copy_target_net(self):
        self.target_net = copy.deepcopy(self.online_net).requires_grad_(False)

    def update_target_net(self):
        for param, target_param in zip(self.online_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_((1 - self.tau) * param.data + self.tau * target_param.data)

    def compute_loss(self, data_dict):
        obs = data_dict['observation'].squeeze()  # bs x 725
        next_obs = data_dict['next_observation'].squeeze()  # bs x 725
        legal_action = data_dict['legal_action'].squeeze()  # bs x 172
        action = data_dict['action'].squeeze()  # bs x 6
        reward = data_dict['reward'].squeeze().reshape([self.batch_size, -1])  # bs x 1
        ### for target q ###
        next_legal_action = data_dict['next_legal_action'].squeeze()  # bs x 172
        done = data_dict['done'].squeeze().reshape([self.batch_size, -1])  # bs x 1
        sub_action_mask = data_dict['sub_action'].squeeze()  # bs x 6

        _, logits = self.online_net(obs, only_inference=False)
        _, next_logits = self.target_net(next_obs, only_inference=False)

        """
            caculate target Q(s', pi(s'))
        """
        split_action = th.split(action, [1] * len(self.label_size_list), dim=1)
        with th.no_grad():
            target_l_q = []
            spilit_legal_actions = th.split(next_legal_action, Config.LEGAL_ACTION_SIZE_LIST, dim=1)
            for ii in range(len(next_logits)):
                legal_action_mask = spilit_legal_actions[ii].float()
                if ii == len(next_logits) - 1:
                    # last dim need specific operation
                    reshaped_next_legal_action = th.reshape(
                        spilit_legal_actions[ii], [self.batch_size, Config.LABEL_SIZE_LIST[0], Config.LABEL_SIZE_LIST[-1]]
                    )  # bs,12,8
                    one_hot_actions = th.reshape(
                        F.one_hot(first_action, num_classes=Config.LABEL_SIZE_LIST[0]), [self.batch_size, Config.LABEL_SIZE_LIST[0], 1]
                    )  # bs,12,1
                    # one_hot_actions = th.reshape(F.one_hot(split_action[0].long().squeeze(), num_classes = Config.LABEL_SIZE_LIST[0]), [self.batch_size, Config.LABEL_SIZE_LIST[0], 1])#bs,12,1
                    next_legal_action = th.sum(reshaped_next_legal_action * one_hot_actions, dim=1)  # bs, 8
                    legal_action_mask = next_legal_action.float()
                masked_logits = next_logits[ii] * legal_action_mask - 1.0e15 * (1 - legal_action_mask)
                target_l_q.append(reward + self.gamma * (1 - done) * masked_logits.max(1, keepdims=True).values)

                cur_argmax = th.argmax(masked_logits, axis=1)[:, None]
                if ii == 0:
                    first_action = th.reshape(cur_argmax, [-1])

        """
            caculate Q(s, a)
        """
        split_action = th.split(action, [1] * len(self.label_size_list), dim=1)
        cur_l_q = []
        for ii in range(len(split_action)):
            cur_onehot_action = F.one_hot(split_action[ii].long().squeeze(), num_classes=Config.LABEL_SIZE_LIST[ii])
            cur_l_q.append((logits[ii] * cur_onehot_action).sum(1, keepdims=True))
        if self.use_sub_action:
            q_loss = sum([((cur_l_q[ii] - target_l_q[ii]) ** 2 * sub_action_mask[:, ii : ii + 1]).mean() for ii in range(len(cur_l_q))]) / len(
                cur_l_q
            )
        else:
            q_loss = sum([((cur_l_q[ii] - target_l_q[ii]) ** 2).mean() for ii in range(len(cur_l_q))]) / len(cur_l_q)
            # q_loss = sum([self.mse(cur_l_q[ii], target_l_q[ii]) for ii in range(len(cur_l_q))]) / len(cur_l_q)

        """
            cur_q_log_sum_exp
        """
        watch_exp = 0
        cur_q_log_sum_exp = []
        spilit_legal_actions = th.split(legal_action, Config.LEGAL_ACTION_SIZE_LIST, dim=1)
        for ii in range(len(logits)):
            legal_action_mask = spilit_legal_actions[ii].float()
            if ii == len(next_logits) - 1:
                # last dim need replay actions
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
            watch_exp = max(watch_exp, masked_logits.max())

            # cur_q_log_sum_exp.append( th.log( masked_logits.exp().sum(1, keepdims=True) + 1e-5 ) )
            cur_q_log_sum_exp.append(th.logsumexp(masked_logits, 1, keepdims=True))

        """
            CQL Loss
        """
        if self.use_sub_action:
            cql_loss = (
                self.cql_alpha
                * sum([((cur_q_log_sum_exp[ii] - cur_l_q[ii]) * sub_action_mask[:, ii : ii + 1]).mean() for ii in range(len(cur_l_q))])
                / len(cur_l_q)
            )
        else:
            cql_loss = self.cql_alpha * sum([(cur_q_log_sum_exp[ii] - cur_l_q[ii]).mean() for ii in range(len(cur_l_q))]) / len(cur_l_q)

        loss = q_loss + cql_loss

        return loss, {
            'q_loss': q_loss,
            'cql_loss': cql_loss,
            'cur_Q': cur_l_q[0].mean(),
            'logits_before_exp': watch_exp,
            'lr': self.lr_schedule.get_lr()[0],
        }

    def to(self, device):
        self.device = device
        self.online_net.device = device
        self.target_net.device = device
        self.online_net.to(device)
        self.target_net.to(device)

    def state_dict(self):
        return self.online_net.state_dict()

    def load_state_dict(self, state_dict):
        self.online_net.load_state_dict(state_dict)
        self._copy_target_net()

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
        grad_norm = th.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10)
        self.optimizer.step()
        self.lr_schedule.step()
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
        self._copy_target_net()

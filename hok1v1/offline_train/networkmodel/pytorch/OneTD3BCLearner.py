import torch as th
import torch.nn.functional as F
import torch.nn as nn
from train_eval_config.OneConfig import ModelConfig as Config
import time
import copy

### 1v1 Base Encoder Model ###
from .module.OneBaseModel import OneBaseModel
from .module.OneDoubleQModel import OneDoubleQModel


class OneTD3BCLearner:
    def __init__(self, args):
        super(OneTD3BCLearner, self).__init__()

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

        ### target net ###
        self._copy_target_net()
        self.tau = args.tau

        self.mse = nn.MSELoss()

        self.bc_constrain = args.bc_constrain
        self.td3bc_alpha = args.td3bc_alpha
        self.actor_update_freq = args.actor_update_freq
        self.train_step = 0

        print('#' * 50)
        print(len(list(self.online_net.parameters())) + len(list(self.online_double_q.parameters())))
        print('#' * 50)

        self.actor_optimizer = th.optim.Adam(params=self.online_net.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8)
        self.q_optimizer = th.optim.Adam(params=self.online_double_q.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8)

    def _copy_target_net(self):
        self.target_net = copy.deepcopy(self.online_net).requires_grad_(False)
        self.target_double_q = copy.deepcopy(self.online_double_q).requires_grad_(False)

    def update_target_net(self):
        ### update target actor network ###
        for param, target_param in zip(self.online_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_((1 - self.tau) * param.data + self.tau * target_param.data)

        ### update target q network ###
        for param, target_param in zip(self.online_double_q.parameters(), self.target_double_q.parameters()):
            target_param.data.copy_((1 - self.tau) * param.data + self.tau * target_param.data)

    def _gumbel_softmax(self, logits, temperature=0.5):
        gumbel_noise = -th.log(-th.log(th.rand_like(logits) + 1e-20))
        y = logits - th.max(logits, dim=-1, keepdim=True)[0] + gumbel_noise
        return th.softmax(y / temperature, dim=-1)

    def compute_loss(self, data_dict):
        obs = data_dict['observation'].squeeze()  # bs x 725
        next_obs = data_dict['next_observation'].squeeze()  # bs x 725
        legal_action = data_dict['legal_action'].squeeze()  # bs x 172
        action = data_dict['action'].squeeze()  # bs x 6
        reward = data_dict['reward'].squeeze().reshape([self.batch_size, -1])  # bs x 1
        ### for target q ###
        next_legal_action = data_dict['next_legal_action'].squeeze()  # bs x 172
        done = data_dict['done'].squeeze().reshape([self.batch_size, -1])  # bs x 1

        obs_feature, logits = self.online_net(obs, only_inference=False)
        next_obs_feature, next_logits = self.target_net(next_obs, only_inference=False)

        """
            calculate target Q(s', pi(s'))
        """
        with th.no_grad():
            # masked_logits_list = []
            onehot_action_list = []
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
                    next_legal_action = th.sum(reshaped_next_legal_action * one_hot_actions, dim=1)  # bs, 8
                    legal_action_mask = next_legal_action.float()
                masked_logits = next_logits[ii] * legal_action_mask - 1.0e15 * (1 - legal_action_mask)
                cur_argmax = th.argmax(masked_logits, axis=1)[:, None]

                onehot_action_list.append(F.one_hot(th.reshape(cur_argmax, [-1]), num_classes=Config.LABEL_SIZE_LIST[ii]))

                if ii == 0:
                    first_action = th.reshape(cur_argmax, [-1])
            # onehot_next_action = th.cat(masked_logits_list, dim=1)
            real_onehot_next_action = th.cat(onehot_action_list, dim=1)
            target_q1, target_q2 = self.target_double_q(next_obs_feature, real_onehot_next_action)
            target_q = reward + self.gamma * (1 - done) * th.min(target_q1, target_q2)

        """
            caculate Q(s, a)
        """
        split_action = th.split(action, [1] * len(self.label_size_list), dim=1)
        cur_l_onehot_actions = []
        for ii in range(len(split_action)):
            cur_l_onehot_actions.append(F.one_hot(split_action[ii].long().squeeze(), num_classes=Config.LABEL_SIZE_LIST[ii]))
        replay_onehot_action = th.cat(cur_l_onehot_actions, dim=1)
        cur_q1, cur_q2 = self.online_double_q(obs_feature, replay_onehot_action)
        q_loss = 0.5 * self.mse(cur_q1, target_q) + 0.5 * self.mse(cur_q2, target_q)

        """
            caculate actor_Q(s, pi(s))
        """
        policy_action_probs = []
        spilit_legal_actions = th.split(legal_action, Config.LEGAL_ACTION_SIZE_LIST, dim=1)
        for ii in range(len(logits)):
            legal_action_mask = spilit_legal_actions[ii].float()
            if ii == len(next_logits) - 1:
                # last dim need specific operation
                reshaped_next_legal_action = th.reshape(
                    spilit_legal_actions[ii], [self.batch_size, Config.LABEL_SIZE_LIST[0], Config.LABEL_SIZE_LIST[-1]]
                )  # bs,12,8
                one_hot_actions = th.reshape(
                    F.one_hot(first_action, num_classes=Config.LABEL_SIZE_LIST[0]), [self.batch_size, Config.LABEL_SIZE_LIST[0], 1]
                )  # bs,12,1
                next_legal_action = th.sum(reshaped_next_legal_action * one_hot_actions, dim=1)  # bs, 8
                legal_action_mask = next_legal_action.float()
            masked_logits = logits[ii] * legal_action_mask - 1.0e15 * (1 - legal_action_mask)

            ### begin softmax ###
            # masked_exp_logits = (masked_logits - masked_logits.max(1, keepdims=True).values).exp()
            # masked_next_probs = masked_exp_logits / masked_exp_logits.sum(1, keepdims=True)
            # policy_action_probs.append(masked_next_probs)
            policy_action_probs.append(self._gumbel_softmax(masked_logits))

            cur_argmax = th.argmax(masked_logits, axis=1)[:, None]
            if ii == 0:
                first_action = th.reshape(cur_argmax, [-1])
        policy_onehot_action = th.cat(policy_action_probs, dim=1)
        actor_q, _ = self.online_double_q(obs_feature, policy_onehot_action)
        lamda = self.td3bc_alpha / actor_q.abs().mean()
        actor_q_loss = -lamda.detach() * actor_q.mean()

        """
            caculate bc loss
        """
        bc_loss = 0
        split_action = th.split(action, [1] * len(self.label_size_list), dim=1)
        for ii in range(len(policy_action_probs)):
            onehot_cur_replay_action = F.one_hot(split_action[ii].long().squeeze(), num_classes=Config.LABEL_SIZE_LIST[ii])
            if ii != len(policy_action_probs) - 1:
                cur_action_prob = onehot_cur_replay_action * policy_action_probs[ii]
            else:
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

                ### begin softmax ###
                masked_logits = logits[ii] * legal_action_mask - 1.0e15 * (1 - legal_action_mask)
                masked_exp_logits = (masked_logits - masked_logits.max(1, keepdims=True).values).exp()
                masked_next_probs = masked_exp_logits / masked_exp_logits.sum(1, keepdims=True)
                cur_action_prob = masked_next_probs

            cur_action_log_prob = th.log(cur_action_prob.sum(1, keepdims=True) + 1e-5)
            bc_loss -= cur_action_log_prob
        bc_loss = self.bc_constrain * bc_loss.mean()

        actor_loss = actor_q_loss + bc_loss

        return (actor_loss, q_loss), {
            'q_loss': q_loss,
            'actor_loss': actor_loss,
            'bc_loss': bc_loss,
            'cur_Q': cur_q1.mean(),
            'train_step': self.train_step,
        }

    def to(self, device):
        self.device = device
        self.online_net.device = device
        self.target_net.device = device
        self.online_double_q.device = device
        self.target_double_q.device = device
        self.online_net.to(device)
        self.target_net.to(device)
        self.online_double_q.to(device)
        self.target_double_q.to(device)

    def state_dict(self):
        return self.online_net.state_dict()

    def load_state_dict(self, state_dict):
        self.online_net.load_state_dict(state_dict)
        self._copy_target_net()

    # def parameters(self):
    #     return self.online_net.parameters()

    def train(self):
        self.online_net.train()
        self.online_double_q.train()

    def eval(self):
        self.online_net.eval()

    def step(self, data_dict):
        before_step = time.time()

        self.train_step += 1

        with th.cuda.amp.autocast():
            loss, info = self.compute_loss(data_dict)
        actor_loss, q_loss = loss

        ### update q network ###
        self.q_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        q_loss.backward(retain_graph=True)
        self.q_optimizer.step()

        ### update actor network ###
        if self.train_step % self.actor_update_freq == 0:
            actor_loss.backward()
            self.actor_optimizer.step()

        info['step_time'] = time.time() - before_step
        return actor_loss + q_loss, info

    def save_dict(self):
        save_dict = {
            "network_state_dict": self.online_net.state_dict(),
            "optimizer_state_dict": self.actor_optimizer.state_dict(),
            "q_state_dict": self.online_double_q.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
        }
        return save_dict

    def load_save_dict(self, save_dict):
        self.online_net.load_state_dict(save_dict['network_state_dict'])
        self.actor_optimizer.load_state_dict(save_dict['optimizer_state_dict'])
        self.online_double_q.load_state_dict(save_dict['q_state_dict'])
        self.q_optimizer.load_state_dict(save_dict['q_optimizer_state_dict'])
        self._copy_target_net()

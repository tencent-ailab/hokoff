import torch as th
import torch.nn.functional as F
from train_eval_config.OneConfig import ModelConfig as Config
import time

### 1v1 Base Encoder Model ###
from .module.OneBaseModel import OneBaseModel


class OneBCLearner:
    def __init__(self, args):
        super(OneBCLearner, self).__init__()

        self.args = args
        self.model_name = Config.NETWORK_NAME
        self.lstm_time_steps = args.lstm_time_steps
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE

        self.learning_rate = args.lr

        self.state_dim = Config.SERI_VEC_SPLIT_SHAPE[0][0]
        self.label_size_list = Config.LABEL_SIZE_LIST

        self.batch_size = args.batch_size * args.lstm_time_steps
        self.online_net = OneBaseModel()
        print('#' * 50)
        print(len(list(self.online_net.parameters())))
        print('#' * 50)
        self.optimizer = th.optim.Adam(params=self.online_net.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8)

        self.use_sub_action = args.use_sub_action

    def update_target_net(self):
        pass

    def compute_loss(self, data_dict):
        obs = data_dict['observation'].squeeze()  # bs x 725
        legal_action = data_dict['legal_action'].squeeze()  # bs x 172
        action = data_dict['action'].squeeze()  # bs x 6
        sub_action_mask = data_dict['sub_action'].squeeze()  # bs x 6

        _, logits = self.online_net(obs, only_inference=False)
        masked_logits_list = []
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
            masked_logits = th.exp(masked_logits - masked_logits.max(1, keepdims=True).values)
            masked_probs = masked_logits / (masked_logits.sum(1, keepdims=True) + 1e-3)
            masked_logits_list.append(masked_probs)

        bc_loss = 0
        split_action = th.split(action, [1] * len(self.label_size_list), dim=1)
        for ii in range(len(masked_logits_list)):
            onehot_cur_replay_action = F.one_hot(split_action[ii].long().squeeze(), num_classes=Config.LABEL_SIZE_LIST[ii])
            cur_action_prob = onehot_cur_replay_action * masked_logits_list[ii]
            cur_action_log_prob = th.log(cur_action_prob.sum(1, keepdims=True) + 1e-5)  ### bs x 1 ###
            if self.use_sub_action:
                cur_action_log_prob *= sub_action_mask[:, ii : ii + 1]
            bc_loss -= cur_action_log_prob
        loss = bc_loss.mean()

        return loss, {}

    def to(self, device):
        self.device = device
        self.online_net.device = device
        self.online_net.to(device)

    def state_dict(self):
        return self.online_net.state_dict()

    def load_state_dict(self, state_dict):
        self.online_net.load_state_dict(state_dict)

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

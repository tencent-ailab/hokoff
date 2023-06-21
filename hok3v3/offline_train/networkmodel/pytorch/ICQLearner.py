import torch as th  # in place of tensorflow

from train_eval_config.Config import Config
from networkmodel.pytorch.module.OFFPGCritic import OffPGCritic
import time


class ICQLearner:
    def __init__(self, args):
        super(ICQLearner, self).__init__()
        # feature configure parameter
        self.args = args
        if 'ind' in args.run_prefix:
            from networkmodel.pytorch.module.IndModel import IndModel as Model

            self.ind = True
        else:
            from networkmodel.pytorch.module.BaseModel import BaseModel as Model

            self.ind = False
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
        self.target_net = Model()
        self.local_steps = 0
        self.local_critic = [
            OffPGCritic(self.hero_label_size_list[0][ii], state_shape=64 + 192 + self.hero_num) for ii in range(len(self.hero_label_size_list[0]))
        ]
        # self.double_local_critic = [OffPGCritic(self.hero_label_size_list[0][ii],state_shape=64+192+self.hero_num) for ii in range(len(self.hero_label_size_list[0]))]
        self.local_target_critic = [
            OffPGCritic(self.hero_label_size_list[0][ii], state_shape=64 + 192 + self.hero_num) for ii in range(len(self.hero_label_size_list[0]))
        ]
        # self.double_local_target_critic = [OffPGCritic(self.hero_label_size_list[0][ii],state_shape=64+192+self.hero_num) for ii in range(len(self.hero_label_size_list[0]))]

        self.update_target_net()
        self.critic_param = []
        for i in range(len(self.local_critic)):
            self.critic_param += list(self.local_critic[i].parameters())
            # self.critic_param+=list(self.double_local_critic[i].parameters())
        self.policy_optimizer = th.optim.Adam(params=list(self.online_net.parameters()), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8)
        self.critic_optimizer = th.optim.Adam(params=self.critic_param, lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8)

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
        for i in range(len(self.local_critic)):
            self.local_target_critic[i].load_state_dict(self.local_critic[i].state_dict())
            # self.double_local_target_critic[i].load_state_dict(self.double_local_critic[i].state_dict())

    def soft_update(self):
        for target_param, param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(target_param.data * self.args.soft_update_tau + param.data * (1 - self.args.soft_update_tau))
        for i in range(len(self.local_critic)):
            for target_param, param in zip(self.local_target_critic[i].parameters(), self.local_critic[i].parameters()):
                target_param.data.copy_(target_param.data * self.args.soft_update_tau + param.data * (1 - self.args.soft_update_tau))

    def _train_critic(self, data_dict):
        obs = data_dict['observation_s']
        next_obs = data_dict['next_observation_s']
        real_next_obs = th.concat([obs[:, 1:], next_obs], dim=1)  # bs,t,na,d
        done = th.unsqueeze(data_dict['done'], dim=-1)  # bs,t,1,1,
        reward = data_dict['reward_s']  # bs,t,na,1,
        sub_action_s = data_dict['sub_action_s']  # torch.Size([bs, t, 3, 5])

        policy_list, _, shared_encoding, individual_encoding = self.online_net(obs, only_inference=False)
        # critic_out_list,raw_critic_out_list,double_critic_out_list,double_raw_critic_out_list = self.critic_out(data_dict,shared_encoding,individual_encoding)
        critic_out_list, raw_critic_out_list = self.critic_out(data_dict, shared_encoding, individual_encoding)
        with th.no_grad():
            tar_shared_encoding, tar_individual_encoding = self.target_net.encoder(real_next_obs)
            # target_critic_out_list,_,double_target_critic_out_list,_ = self.critic_out(data_dict,tar_shared_encoding,tar_individual_encoding,target=True)
            target_critic_out_list, _ = self.critic_out(data_dict, tar_shared_encoding, tar_individual_encoding, target=True)
        critic_loss = 0.0
        for agent_idx in range(self.hero_num):
            agent_is_target_critic_out_list = []
            agent_sub_action = sub_action_s[:, :, agent_idx : agent_idx + 1]  # bs,t,1,5
            for label_index, label_dim in enumerate(self.hero_label_size_list[agent_idx]):
                # if th.sum(agent_sub_action[:,:,:,label_index:label_index+1])<1:
                #     continue
                beta = 1000
                advantage_q = th.nn.functional.softmax(
                    (target_critic_out_list[agent_idx][label_index] - th.max(target_critic_out_list[agent_idx][label_index], 0, keepdim=True)[0])
                    / beta,
                    dim=0,
                )
                is_target_critic = len(advantage_q) * advantage_q * target_critic_out_list[agent_idx][label_index]
                # double_advantage_q= th.nn.functional.softmax((double_target_critic_out_list[agent_idx][label_index]-th.max(double_target_critic_out_list[agent_idx][label_index],0,keepdim=True)[0])/beta,dim=0)
                # double_is_target_critic = len(double_advantage_q)*double_advantage_q*double_target_critic_out_list[agent_idx][label_index]
                # real_target = reward[:,:,agent_idx:agent_idx+1]+self.args.gamma*(1-done)*th.min(is_target_critic,double_is_target_critic)
                real_target = reward[:, :, agent_idx : agent_idx + 1] + self.args.gamma * (1 - done) * is_target_critic

                # critic_loss=critic_loss + th.mean(((critic_out_list[agent_idx][label_index]-real_target.detach())*agent_sub_action[:,:,:,label_index:label_index+1])**2*0.5)
                critic_loss = critic_loss + th.mean(((critic_out_list[agent_idx][label_index] - real_target.detach())) ** 2 * 0.5)
                # critic_loss=critic_loss + th.mean(((double_critic_out_list[agent_idx][label_index]-real_target.detach())*agent_sub_action[:,:,:,label_index:label_index+1])**2*0.5)
        critic_loss = critic_loss / (self.hero_num * len(self.hero_label_size_list[0]))
        return critic_loss, policy_list, critic_out_list, raw_critic_out_list  # ,double_critic_out_list,double_raw_critic_out_list

    def critic_out(self, data_dict, shared_encoding, individual_encoding, target=False):
        if target:
            actions = data_dict['action_s']
            next_actions = data_dict['next_action_s']
            real_actions = th.concat([actions[:, 1:], next_actions], dim=1)  # bs,t,na,d
        else:
            real_actions = data_dict['action_s']
        if not self.ind:
            state = [th.concat([shared_encoding, individual_encoding[ii]], dim=-1) for ii in range(self.hero_num)]  # bs,t,1,64+192
        else:
            state = individual_encoding  # bs,t,1,256
        done = th.unsqueeze(data_dict['done'], dim=-1)  # bs,t,1,1,
        critic_out_list = []
        raw_critic_out_list = []
        double_critic_out_list = []
        double_raw_critic_out_list = []
        state_agent_one_hot_list = []
        for agent_idx in range(self.hero_num):
            agent_one_hot = th.nn.functional.one_hot(th.ones_like(done).long() * agent_idx, num_classes=self.hero_num).squeeze(-2)  # bs,t,1,3
            state_agent_one_hot_list.append(th.concat([state[agent_idx], agent_one_hot], dim=-1))
            agent_critic_out_list = []
            raw_agent_critic_out_list = []
            double_agent_critic_out_list = []
            double_raw_agent_critic_out_list = []
            for label_index, label_dim in enumerate(self.hero_label_size_list[agent_idx]):
                if target:
                    label_agent_critic_out = th.detach(self.local_target_critic[label_index](state_agent_one_hot_list[-1]))  # bs,t,1,d
                    # double_label_agent_critic_out = th.detach(self.double_local_target_critic[label_index](state_agent_one_hot_list[-1]))#bs,t,1,d
                else:
                    label_agent_critic_out = self.local_critic[label_index](state_agent_one_hot_list[-1])  # bs,t,1,d
                    # double_label_agent_critic_out = self.double_local_critic[label_index](state_agent_one_hot_list[-1])#bs,t,1,d
                chosen_label_agent_critic_out = th.gather(
                    label_agent_critic_out, dim=-1, index=real_actions[:, :, agent_idx : agent_idx + 1, label_index : label_index + 1].long()
                )  # bs,t,1,1
                # double_chosen_label_agent_critic_out = th.gather(double_label_agent_critic_out,dim=-1,index=real_actions[:,:,agent_idx:agent_idx+1,label_index:label_index+1].long())#bs,t,1,1
                agent_critic_out_list.append(chosen_label_agent_critic_out)
                raw_agent_critic_out_list.append(label_agent_critic_out)
                # double_agent_critic_out_list.append(double_chosen_label_agent_critic_out)
                # double_raw_agent_critic_out_list.append(double_label_agent_critic_out)
            critic_out_list.append(agent_critic_out_list)
            raw_critic_out_list.append(raw_agent_critic_out_list)
            double_critic_out_list.append(double_agent_critic_out_list)
            double_raw_critic_out_list.append(double_raw_agent_critic_out_list)
        return critic_out_list, raw_critic_out_list  # ,double_critic_out_list,double_raw_critic_out_list

    def compute_loss(self, data_dict):
        # critic_loss,policy_list,critic_out_list,raw_critic_out_list,double_critic_out_list,double_raw_critic_out_list = self._train_critic(data_dict)
        critic_loss, policy_list, critic_out_list, raw_critic_out_list = self._train_critic(data_dict)
        legal_action_s = data_dict['legal_action_s']  # torch.Size([64, 16, 3, 161])
        action_s = data_dict['action_s']  # torch.Size([64, 16, 3, 5])
        sub_action_s = data_dict['sub_action_s']  # torch.Size([64, 16, 3, 5])

        local_q_taken_list = []
        local_prob_taken_list = []
        policy_loss = 0
        for agent_idx in range(self.hero_num):
            agent_action = action_s[:, :, agent_idx : agent_idx + 1]  # bs,t,1,5
            agent_sub_action = sub_action_s[:, :, agent_idx : agent_idx + 1]  # bs,t,1,5
            split_agent_legal_action = th.split(legal_action_s[:, :, agent_idx : agent_idx + 1], self.hero_label_size_list[0], dim=-1)
            for label_index, label_dim in enumerate(self.hero_label_size_list[agent_idx]):
                # if th.sum(agent_sub_action[:,:,:,label_index:label_index+1])<1:
                #     continue
                agent_label_logits = policy_list[agent_idx][label_index]
                agent_label_logits = (
                    agent_label_logits * split_agent_legal_action[label_index] - (1.0 - split_agent_legal_action[label_index]) * 10**4
                )
                max_agent_label_logits = th.max(agent_label_logits, dim=-1, keepdim=True)[0]
                numerator = th.exp(agent_label_logits - max_agent_label_logits)
                agent_label_prob = numerator / numerator.sum(-1, keepdim=True)
                agent_label_prob[split_agent_legal_action[label_index] == 0] = 0  # bs,t,1,d
                agent_label_prob_taken = th.gather(
                    agent_label_prob, dim=-1, index=agent_action[:, :, :, label_index : label_index + 1].long()
                )  # bs,t,1,1
                log_agent_label_prob_taken = th.log(agent_label_prob_taken + 0.0001)

                baseline = th.sum(agent_label_prob * raw_critic_out_list[agent_idx][label_index], dim=-1).unsqueeze(-1).detach()  # bs,t,1,1
                advantages = critic_out_list[agent_idx][label_index] - baseline
                # double_baseline = th.sum(agent_label_prob*double_raw_critic_out_list[agent_idx][label_index],dim=-1).unsqueeze(-1).detach()#bs,t,1,1
                # double_advantages = double_critic_out_list[agent_idx][label_index]-double_baseline
                # advantages = th.min(advantages,double_advantages)

                local_q_taken_list.append(critic_out_list[agent_idx][label_index])
                local_prob_taken_list.append(agent_label_prob_taken)
                beta = 0.1
                advantages = th.exp((advantages - th.max(advantages, dim=0, keepdim=True)[0]) / beta) / th.exp(
                    (advantages - th.max(advantages, dim=0, keepdim=True)[0]) / beta
                ).sum(dim=0, keepdim=True)
                policy_loss = policy_loss - th.mean(
                    (len(advantages) * advantages).detach() * log_agent_label_prob_taken
                )  # *agent_sub_action[:,:,:,label_index:label_index+1])
                # policy_loss = policy_loss - th.mean((len(advantages)*advantages).detach()*log_agent_label_prob_taken)
        policy_loss = policy_loss / (self.hero_num * len(self.hero_label_size_list[0]))
        return policy_loss + critic_loss, {
            'policy_loss': policy_loss,
            'critic_loss': critic_loss,
            'local_q_taken': th.mean(th.stack(local_q_taken_list)),
            'prob_taken': th.mean(th.stack(local_prob_taken_list)),
        }

    def to(self, device):
        self.online_net.to(device)
        self.target_net.to(device)
        for i in range(len(self.local_critic)):
            self.local_critic[i].to(device)
            self.local_target_critic[i].to(device)
            # self.double_local_critic[i].to(device)
            # self.double_local_target_critic[i].to(device)

    def train(self):
        self.online_net.train()
        for i in range(len(self.local_critic)):
            self.local_critic[i].train()
            # self.double_local_critic[i].train()

    def eval(self):
        self.online_net.eval()
        for i in range(len(self.local_critic)):
            self.local_critic[i].eval()
            # self.double_local_critic[i].eval()

    def step(self, data_dict):
        before_step = time.time()
        with th.cuda.amp.autocast():
            loss, info = self.compute_loss(data_dict)
        # with th.autograd.detect_anomaly():

        self.critic_optimizer.zero_grad()
        info['critic_loss'].backward(retain_graph=True)
        grad_norm = th.nn.utils.clip_grad_norm_(list(self.critic_param), 10)
        if th.isnan(grad_norm).any():
            print('critic nan!!!!')
            exit(0)
        self.critic_optimizer.step()

        info['policy_loss'].backward()
        grad_norm = th.nn.utils.clip_grad_norm_(list(self.online_net.parameters()), 10)
        if th.isnan(grad_norm).any():
            print('policy nan!!!!')
            exit(0)
        self.policy_optimizer.step()
        self.policy_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        info['step_time'] = time.time() - before_step
        return loss, info

    def save_dict(self):
        save_dict = {
            "network_state_dict": self.online_net.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            # "local_critic_state_dict" : self.local_critic.state_dict(),
        }
        for i in range(len(self.local_critic)):
            save_dict['local_critic_{}_state_dict'.format(i)] = self.local_critic[i].state_dict()
            # save_dict['double_local_critic_{}_state_dict'.format(i)] = self.double_local_critic[i].state_dict()
        return save_dict

    def load_save_dict(self, save_dict):
        self.online_net.load_state_dict(save_dict['network_state_dict'])
        self.policy_optimizer.load_state_dict(save_dict['policy_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(save_dict['critic_optimizer_state_dict'])
        for i in range(len(self.local_critic)):
            self.local_critic[i].load_state_dict(save_dict['local_critic_{}_state_dict'.format(i)])
        #    self.double_local_critic[i].load_state_dict(save_dict['double_local_critic_{}_state_dict'.format(i)])
        self.update_target_net()

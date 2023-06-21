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
from networkmodel.pytorch.module.Mixer import QMixer
from networkmodel.pytorch.module.Critic import DoubleMLPNetwork
import time


class OMARLearner:
    def __init__(self, args):
        super(OMARLearner, self).__init__()
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
            DoubleMLPNetwork(64 + 192 + self.hero_num, action_dim=self.hero_label_size_list[0][ii]) for ii in range(len(self.hero_label_size_list[0]))
        ]
        self.local_target_critic = [
            DoubleMLPNetwork(64 + 192 + self.hero_num, action_dim=self.hero_label_size_list[0][ii]) for ii in range(len(self.hero_label_size_list[0]))
        ]
        self.update_target_net()
        self.critic_param = []
        for i in range(len(self.local_critic)):
            self.critic_param += list(self.local_critic[i].parameters())
        self.policy_optimizer = th.optim.Adam(params=list(self.online_net.parameters()), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8)
        self.critic_optimizer = th.optim.Adam(params=self.critic_param, lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8)

        self.omar_coe = args.omar_coe
        # self.omar_iters=2
        # self.omar_num_samples = 5

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
        for i in range(len(self.local_critic)):
            self.local_target_critic[i].load_state_dict(self.local_critic[i].state_dict())

    def _train_critic(self, data_dict):
        obs = data_dict['observation_s']
        next_obs = data_dict['next_observation_s']
        real_next_obs = th.concat([obs[:, 1:], next_obs], dim=1)  # bs,t,na,d
        done = th.unsqueeze(data_dict['done'], dim=-1)  # bs,t,1,1,
        reward = data_dict['reward_s']  # bs,t,na,1,
        sub_action_s = data_dict['sub_action_s']  # torch.Size([64, 16, 3, 5])
        legal_action_s = data_dict['legal_action_s']
        actions = data_dict['action_s']

        tar_policy_list, _, tar_shared_encoding, tar_individual_encoding = self.target_net(real_next_obs, only_inference=False)
        tar_state_agent_one_hot_list = self.critic_out(data_dict, tar_shared_encoding, tar_individual_encoding, target=True)
        policy_list, _, shared_encoding, individual_encoding = self.online_net(obs, only_inference=False)
        state_agent_one_hot_list = self.critic_out(data_dict, shared_encoding, individual_encoding)
        td_loss = 0.0
        cql_loss = 0.0
        local_q_1_list = []
        for agent_idx in range(self.hero_num):
            agent_sub_action = sub_action_s[:, :, agent_idx : agent_idx + 1]  # bs,t,1,5
            hero_legal_action = legal_action_s[:, :, agent_idx : agent_idx + 1]  # bs,t,1,161
            split_hero_legal_action = th.split(hero_legal_action, self.hero_label_size_list[0], dim=-1)
            tmp_local_q_1_list = []
            for label_index, label_dim in enumerate(self.hero_label_size_list[agent_idx]):
                tar_agent_logits = tar_policy_list[agent_idx][label_index]  # bs,t,1,d
                label_split_hero_legal_action = split_hero_legal_action[label_index]  # bs,t,1,d
                masked_tar_agent_logits = tar_agent_logits * label_split_hero_legal_action - 10**10 * (1 - label_split_hero_legal_action)
                target_local_q_1, target_local_q_2 = self.local_target_critic[label_index](tar_state_agent_one_hot_list[agent_idx])  # bs,t,1,ad
                masked_tar_policy_actions = th.argmax(masked_tar_agent_logits, dim=-1, keepdim=True).long()  # bs,t,1,1
                masked_chosen_target_local_q_1 = th.gather(target_local_q_1, dim=-1, index=masked_tar_policy_actions)
                masked_chosen_target_local_q_2 = th.gather(target_local_q_2, dim=-1, index=masked_tar_policy_actions)

                real_target = reward[:, :, agent_idx : agent_idx + 1] + self.args.gamma * (1 - done) * th.min(
                    masked_chosen_target_local_q_1, masked_chosen_target_local_q_2
                )  # bs,t,1,1
                local_q_1, local_q_2 = self.local_critic[label_index](state_agent_one_hot_list[agent_idx])  # bs,t,1,ad
                tmp_local_q_1_list.append(local_q_1)
                masked_chosen_local_q_1 = th.gather(
                    local_q_1, dim=-1, index=actions[:, :, agent_idx : agent_idx + 1, label_index : label_index + 1].long()
                )
                masked_chosen_local_q_2 = th.gather(
                    local_q_2, dim=-1, index=actions[:, :, agent_idx : agent_idx + 1, label_index : label_index + 1].long()
                )
                # td_loss=td_loss+th.mean(((critic_out_list[agent_idx][label_index]-real_target.detach())*agent_sub_action[:,:,:,label_index:label_index+1])**2*0.5)
                # td_loss=td_loss+th.mean(((masked_chosen_local_q_1-real_target.detach()))**2*0.5)
                # td_loss=td_loss+th.mean(((masked_chosen_local_q_2-real_target.detach()))**2*0.5)
                td_loss = td_loss + th.mean(
                    ((masked_chosen_local_q_1 - real_target.detach()) * agent_sub_action[:, :, :, label_index : label_index + 1]) ** 2 * 0.5
                )
                td_loss = td_loss + th.mean(
                    ((masked_chosen_local_q_2 - real_target.detach()) * agent_sub_action[:, :, :, label_index : label_index + 1]) ** 2 * 0.5
                )

                masked_agent_label_local_q_1 = local_q_1 * split_hero_legal_action[label_index] - 10**10 * (
                    1 - split_hero_legal_action[label_index]
                )  # bs,t,1,d
                masked_agent_label_local_q_2 = local_q_2 * split_hero_legal_action[label_index] - 10**10 * (
                    1 - split_hero_legal_action[label_index]
                )  # bs,t,1,d
                negative_sampling_1 = th.logsumexp(masked_agent_label_local_q_1, dim=-1, keepdim=True)  # bs,t,1,1
                negative_sampling_2 = th.logsumexp(masked_agent_label_local_q_2, dim=-1, keepdim=True)  # bs,t,1,1
                cql_loss += th.mean((negative_sampling_1 - masked_chosen_local_q_1) * agent_sub_action[:, :, :, label_index : label_index + 1])
                cql_loss += th.mean((negative_sampling_2 - masked_chosen_local_q_2) * agent_sub_action[:, :, :, label_index : label_index + 1])
            local_q_1_list.append(tmp_local_q_1_list)
        critic_loss = (td_loss + self.args.cql_alpha * cql_loss) / (self.hero_num * len(self.hero_label_size_list[0]))

        # self.critic_optimizer.zero_grad()
        return critic_loss, policy_list, shared_encoding, individual_encoding, state_agent_one_hot_list, local_q_1_list

    def critic_out(self, data_dict, shared_encoding, individual_encoding, target=False):
        if target:
            obs = data_dict['observation_s']
            next_obs = data_dict['next_observation_s']
            real_obs = th.concat([obs[:, 1:], next_obs], dim=1)  # bs,t,na,d
            actions = data_dict['action_s']
            next_actions = data_dict['next_action_s']
            real_actions = th.concat([actions[:, 1:], next_actions], dim=1)  # bs,t,na,d
        else:
            real_obs = data_dict['observation_s']
            real_actions = data_dict['action_s']
        if not self.ind:
            state = [th.concat([shared_encoding, individual_encoding[ii]], dim=-1) for ii in range(self.hero_num)]  # bs,t,1,64+192
        else:
            state = individual_encoding  # bs,t,1,256
        done = th.unsqueeze(data_dict['done'], dim=-1)  # bs,t,1,1,
        critic_out_list = []
        raw_critic_out_list = []
        state_agent_one_hot_list = []
        for agent_idx in range(self.hero_num):
            agent_one_hot = th.nn.functional.one_hot(th.ones_like(done).long() * agent_idx, num_classes=self.hero_num).squeeze(-2)  # bs,t,1,3
            state_agent_one_hot_list.append(th.concat([state[agent_idx], agent_one_hot], dim=-1))
            # agent_critic_out_list = []
            # raw_agent_critic_out_list = []
            # for label_index, label_dim in enumerate(self.hero_label_size_list[agent_idx]):
            #     if target:
            #         label_agent_critic_out = th.detach(self.local_target_critic[label_index](state_agent_one_hot_list[-1]))#bs,t,1,d
            #     else:
            #         label_agent_critic_out = self.local_critic[label_index](state_agent_one_hot_list[-1])#bs,t,1,d
            #     chosen_label_agent_critic_out = th.gather(label_agent_critic_out,dim=-1,index=real_actions[:,:,agent_idx:agent_idx+1,label_index:label_index+1].long())#bs,t,1,1
            #     agent_critic_out_list.append(chosen_label_agent_critic_out)
            #     raw_agent_critic_out_list.append(label_agent_critic_out)
            # critic_out_list.append(agent_critic_out_list)
            # raw_critic_out_list.append(raw_agent_critic_out_list)
        return state_agent_one_hot_list

    def _gumbel_softmax(self, logits, temperature=0.5):
        gumbel_noise = -th.log(-th.log(th.rand_like(logits) + 1e-20))
        y = logits - th.max(logits, dim=-1, keepdim=True)[0] + gumbel_noise
        return th.softmax(y / temperature, dim=-1)

    def compute_loss(self, data_dict):
        critic_loss, policy_list, shared_encoding, individual_encoding, state_agent_one_hot_list, local_q_1_list = self._train_critic(data_dict)
        obs_s = data_dict['observation_s']  # torch.Size([64, 16, 3, 4586])
        legal_action_s = data_dict['legal_action_s']  # torch.Size([64, 16, 3, 161])
        action_s = data_dict['action_s']  # torch.Size([64, 16, 3, 5])
        sub_action_s = data_dict['sub_action_s']  # torch.Size([64, 16, 3, 5])
        bs, max_t, n_agents, _ = obs_s.shape

        td_error = 0
        cql_error = 0
        local_q_taken_list = []
        local_prob_taken_list = []
        policy_loss = 0
        for agent_idx in range(self.hero_num):
            agent_action = action_s[:, :, agent_idx : agent_idx + 1]  # bs,t,1,5
            agent_sub_action = sub_action_s[:, :, agent_idx : agent_idx + 1]  # bs,t,1,5
            split_agent_legal_action = th.split(legal_action_s[:, :, agent_idx : agent_idx + 1], self.hero_label_size_list[0], dim=-1)
            for label_index, label_dim in enumerate(self.hero_label_size_list[agent_idx]):
                agent_label_logits = policy_list[agent_idx][label_index]
                agent_label_logits = agent_label_logits - 10**4 * (1.0 - split_agent_legal_action[label_index])
                numerator = th.exp((agent_label_logits - th.max(agent_label_logits, dim=-1, keepdim=True)[0]))
                agent_label_prob = numerator / numerator.sum(dim=-1, keepdim=True)
                agent_label_prob[split_agent_legal_action[label_index] == 0] = 0  # bs,t,1,d
                # local_q_1,local_q_2 = self.local_critic[label_index](state_agent_one_hot_list[agent_idx])#bs,t,1,1
                local_q_1 = local_q_1_list[agent_idx][label_index]
                local_q_1_taken = th.gather(local_q_1, dim=-1, index=agent_action[:, :, :, label_index : label_index + 1].long())
                baseline = th.sum(local_q_1 * agent_label_prob, dim=-1, keepdim=True)
                advantage = local_q_1_taken - baseline
                log_prob_taken = th.log(
                    th.gather(agent_label_prob, dim=-1, index=agent_action[:, :, :, label_index : label_index + 1].long()) + 0.00001
                )
                policy_loss += -(1 - self.omar_coe) * th.mean(
                    (log_prob_taken * advantage.detach()) * agent_sub_action[:, :, :, label_index : label_index + 1]
                )
                # #############omar####################
                # action_dim = label_dim
                # self.omar_mu = th.cuda.FloatTensor(bs,max_t,1, 1).zero_() + label_dim/2
                # self.omar_sigma = th.cuda.FloatTensor(bs,max_t,1, 1).zero_() + label_dim/2
                # repeat_avail_action = th.repeat_interleave(split_agent_legal_action[label_index].unsqueeze(-2),repeats=self.omar_num_samples,dim=-2)#bs,ts,1,nsample,ad
                # for iter_idx in range(self.omar_iters):
                #     # print(self.omar_mu.max(),self.omar_sigma.max())
                #     dist = th.distributions.Normal(self.omar_mu, self.omar_sigma)

                #     cem_sampled_acs = dist.sample((self.omar_num_samples,)).permute(1,2,3,0,4).clamp(0, action_dim-1)
                #     cem_sampled_acs = th.div(cem_sampled_acs+0.5,1,rounding_mode='trunc').long()#discretize
                #     #bs,ts,na,nsample,1
                #     cem_sampled_avail = th.gather(repeat_avail_action,dim=-1,index=cem_sampled_acs)#bs,ts,na,nsample,1

                #     repeat_q_vals = th.repeat_interleave(local_q_1.unsqueeze(-2),repeats=self.omar_num_samples,dim=-2)#bs,ts,1,nsample,ad
                #     all_pred_qvals = th.gather(repeat_q_vals,dim=-1,index=cem_sampled_acs)#bs,ts,1,nsample,1
                #     all_pred_qvals = all_pred_qvals-1e4*(1.0-cem_sampled_avail)
                #     if th.min(th.max(local_q_1, -1, keepdim=True)[0]).item()<-1e9:
                #         continue

                #     updated_mu = self.compute_softmax_acs(all_pred_qvals, cem_sampled_acs)
                #     self.omar_mu = updated_mu#bs,ts,na,1

                #     updated_sigma = th.sqrt(th.mean(((cem_sampled_acs - updated_mu.unsqueeze(-2))*cem_sampled_avail) ** 2, -2))
                #     self.omar_sigma = updated_sigma+0.01#bs,ts,na,1
                # dist = th.distributions.Normal(self.omar_mu, self.omar_sigma)
                # cem_sampled_acs = dist.sample((self.omar_num_samples,)).permute(1,2,3,0,4).clamp(0, action_dim-1)
                # cem_sampled_acs = th.div(cem_sampled_acs+0.5,1,rounding_mode='trunc').long()#discretize
                # #bs,ts,na,nsample,1
                # cem_sampled_avail = th.gather(repeat_avail_action,dim=-1,index=cem_sampled_acs)#bs,ts,na,nsample,1

                # repeat_q_vals = th.repeat_interleave(local_q_1.unsqueeze(-2),repeats=self.omar_num_samples,dim=-2)#bs,ts,1,nsample,ad
                # all_pred_qvals = th.gather(repeat_q_vals,dim=-1,index=cem_sampled_acs)#bs,ts,na,nsample,1
                # all_pred_qvals = all_pred_qvals-1e4*(1.0-cem_sampled_avail)
                # top_qvals, top_inds = th.topk(all_pred_qvals, 1, dim=-2)#bs,ts,na,1,1
                # top_acs = th.gather(cem_sampled_acs, -2, top_inds)#bs,ts,na,1,1
                # curr_pol_actions = agent_label_prob.argmax(-1,keepdim=True)#bs,ts,na,1

                # cem_qvals = top_qvals.squeeze(-1)#bs,ts,na,1
                # pol_qvals = th.gather(local_q_1, dim=3, index=curr_pol_actions)#bs,ts,na,1
                # cem_acs = top_acs.squeeze(-1)#bs,ts,na,1
                # pol_acs = curr_pol_actions#bs,ts,na,1

                # candidate_qvals = th.cat([pol_qvals, cem_qvals], -1)#bs,ts,na,2
                # candidate_acs = th.cat([pol_acs, cem_acs], -1)#bs,ts,na,2

                # max_qvals, max_inds = th.max(candidate_qvals, -1, keepdim=True)#bs,ts,na,1

                # max_acs = th.gather(candidate_acs, -1, max_inds)#bs,ts,na,1
                # one_hot_max_acs = th.nn.functional.one_hot(max_acs,num_classes=action_dim).float()#bs,ts,na,ad
                max_acs = th.argmax(local_q_1, dim=-1, keepdim=True)  # bs,ts,na,1
                one_hot_max_acs = th.nn.functional.one_hot(max_acs, num_classes=label_dim).float()  # bs,ts,na,ad

                omar_loss = th.nn.functional.mse_loss(
                    agent_label_prob.view(-1, label_dim), one_hot_max_acs.view(-1, label_dim).detach(), reduce=False
                )
                policy_loss += self.omar_coe * th.mean(
                    omar_loss.reshape(bs, max_t, 1, label_dim) * agent_sub_action[:, :, :, label_index : label_index + 1]
                )

                local_q_taken_list.append(local_q_1_taken)
                local_prob_taken_list.append(agent_label_prob.max(-1, keepdim=True)[0])
                ####################################
        # policy_loss = self.omar_coe*omar_loss+(1-self.omar_coe)*coma_loss
        policy_loss = policy_loss / (self.hero_num * len(self.hero_label_size_list[0]))

        return policy_loss + critic_loss, {
            'policy_loss': policy_loss,
            'critic_loss': critic_loss,
            'local_q_taken': th.mean(th.stack(local_q_taken_list)),
            'prob_taken': th.mean(th.stack(local_prob_taken_list)),
        }

    def compute_softmax_acs(self, q_vals, acs):
        max_q_vals = th.max(q_vals, -2, keepdim=True)[0]  # bs,ts,na,1,1
        norm_q_vals = q_vals - max_q_vals
        e_beta_normQ = th.exp(norm_q_vals)  # bs,ts,na,nsample,1
        a_mult_e = acs * e_beta_normQ  # bs,ts,na,nsample,1
        numerators = a_mult_e
        denominators = e_beta_normQ

        sum_numerators = th.sum(numerators, -2)
        sum_denominators = th.sum(denominators, -2)

        softmax_acs = sum_numerators / sum_denominators  # bs,ts,na,1

        return softmax_acs

    def to(self, device):
        self.online_net.to(device)
        self.target_net.to(device)
        # if self.global_mixer is not None:
        #     # for ii in range(len(self.local_mixer)):
        #     #     self.local_mixer[ii].to(device)
        #     #     self.target_local_mixer[ii].to(device)
        #     self.local_mixer.to(device)
        #     self.target_local_mixer.to(device)
        #     self.target_global_mixer.to(device)
        #     self.global_mixer.to(device)
        for i in range(len(self.local_critic)):
            self.local_critic[i].to(device)
            self.local_target_critic[i].to(device)

    # def state_dict(self):
    #     return self.online_net.state_dict()
    # def load_state_dict(self,state_dict):
    #     self.online_net.load_state_dict(state_dict)
    #     self.target_net.load_state_dict(state_dict)
    # def parameters(self):
    #     return self.online_net.parameters()
    def train(self):
        self.online_net.train()
        for i in range(len(self.local_critic)):
            self.local_critic[i].train()

    def eval(self):
        self.online_net.eval()
        for i in range(len(self.local_critic)):
            self.local_critic[i].eval()

    def step(self, data_dict):
        before_step = time.time()
        # self.policy_optimizer.zero_grad()
        with th.cuda.amp.autocast():
            loss, info = self.compute_loss(data_dict)

        self.critic_optimizer.zero_grad()
        info['critic_loss'].backward(retain_graph=True)
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_param, 10)
        if th.isnan(grad_norm).any():
            print('critic_grad_norm nan')
            exit(0)
        self.critic_optimizer.step()

        info['policy_loss'].backward()
        grad_norm = th.nn.utils.clip_grad_norm_(list(self.online_net.parameters()), 10)
        info['policy_grad_norm'] = grad_norm
        if th.isnan(grad_norm).any():
            print('policy_grad_norm nan')
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
        return save_dict

    def load_save_dict(self, save_dict):
        self.online_net.load_state_dict(save_dict['network_state_dict'])
        self.policy_optimizer.load_state_dict(save_dict['policy_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(save_dict['critic_optimizer_state_dict'])
        for i in range(len(self.local_critic)):
            self.local_critic[i].load_state_dict(save_dict['local_critic_{}_state_dict'.format(i)])
        self.update_target_net()

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GlobalMixer(nn.Module):
    def __init__(self, n_agents, n_actions, local_state_shape, state_shape, mixing_embed_dim=128):
        super(GlobalMixer, self).__init__()

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.local_state_dim = int(np.prod(local_state_shape))
        self.state_dim = int(np.prod(state_shape))

        self.embed_dim = mixing_embed_dim

        self.local_hyper_w_1 = nn.Sequential(
            nn.Linear(self.local_state_dim, self.embed_dim), nn.ReLU(inplace=True), nn.Linear(self.embed_dim, self.n_actions)
        )
        # V(s) instead of a bias for the last layers
        self.local_V = nn.Sequential(nn.Linear(self.local_state_dim, self.embed_dim), nn.ReLU(inplace=True), nn.Linear(self.embed_dim, 1))

        self.global_hyper_w_1 = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim), nn.ReLU(inplace=True), nn.Linear(self.embed_dim, self.n_agents)
        )
        # V(s) instead of a bias for the last layers
        self.global_V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim), nn.ReLU(inplace=True), nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, shared_encoding, individual_encoding, ind=False):
        bs, t, na, ad = agent_qs.shape
        state_agent_one_hot_list = []
        for agent_idx in range(self.n_agents):
            agent_one_hot = th.nn.functional.one_hot(th.ones_like(agent_qs[:, :, 0:1, 0:1]).long() * agent_idx, num_classes=self.n_agents).squeeze(
                -2
            )  # bs,t,1,3
            if not ind:
                state_agent_one_hot_list.append(th.concat([shared_encoding, individual_encoding[agent_idx], agent_one_hot], dim=-1))
            else:
                state_agent_one_hot_list.append(th.concat([individual_encoding[agent_idx], agent_one_hot], dim=-1))
        local_states = th.concat(state_agent_one_hot_list, dim=-2)  # bs,t,3,dim
        if not ind:
            states = th.concat([shared_encoding] + list(individual_encoding), dim=-1)  # bs,t,1,dim
        else:
            states = th.concat(list(individual_encoding), dim=-1)

        states = states.reshape(-1, self.state_dim).detach()
        local_states = local_states.reshape(-1, self.local_state_dim).detach()
        agent_qs = agent_qs.reshape(-1, 1, self.n_actions)
        # First layer for each agent
        w1 = self.local_hyper_w_1(local_states).abs()
        w1 = w1.view(-1, self.n_actions, 1)
        v = self.local_V(local_states).view(-1, 1, 1)
        hidden = th.bmm(agent_qs, w1) + v  # -1, 1, 1
        local_q_tot = hidden.reshape(-1, 1, self.n_agents)

        # Second layer for global
        global_w1 = self.global_hyper_w_1(states).abs()
        global_w1 = global_w1.view(-1, self.n_agents, 1)
        global_v = self.global_V(states).view(-1, 1, 1)
        # Compute final output
        global_y = th.bmm(local_q_tot, global_w1) + global_v
        # Reshape and return
        q_tot = global_y.view(bs, t, 1, 1)
        agent_w1 = w1.view(bs, t, self.n_agents, self.n_actions)
        norm_agent_w1 = agent_w1 / th.sum(agent_w1, dim=-1, keepdim=True)
        reshaped_global_w1 = global_w1.view(bs, t, self.n_agents, 1)
        norm_global_w1 = reshaped_global_w1 / th.sum(reshaped_global_w1, dim=-2, keepdim=True)
        k = norm_agent_w1 * norm_global_w1  # bs,t,na,action

        return q_tot, k

    # def k(self, states):
    #     bs = states.size(0)
    #     w1 = th.abs(self.local_hyper_w_1(states))
    #     w_final = th.abs(self.local_hyper_w_final(states))
    #     w1 = w1.view(-1, self.n_actions, self.embed_dim)
    #     w_final = w_final.view(-1, self.embed_dim, 1)
    #     local_k = th.bmm(w1,w_final).view(bs, -1, self.n_actions)
    #     local_k = local_k / th.sum(local_k, dim=2, keepdim=True)#bs,1,actions

    #     return k

    # def b(self, states):
    #     bs = states.size(0)
    #     w_final = th.abs(self.hyper_w_final(states))
    #     w_final = w_final.view(-1, self.embed_dim, 1)
    #     b1 = self.hyper_b_1(states)
    #     b1 = b1.view(-1, 1, self.embed_dim)
    #     v = self.V(states).view(-1, 1, 1)
    #     b = th.bmm(b1, w_final) + v
    #     return b

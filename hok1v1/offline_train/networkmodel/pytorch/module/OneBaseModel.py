import torch
from torch import nn
import numpy as np
from train_eval_config.OneConfig import DimConfig
from train_eval_config.OneConfig import ModelConfig as Config


class OneBaseModel(nn.Module):
    def __init__(self):
        """
        input s, get the feature after encoding
        """
        super(OneBaseModel, self).__init__()

        state_dim = Config.SERI_VEC_SPLIT_SHAPE[0][0]
        label_size_list = Config.LABEL_SIZE_LIST
        """
            network parameters
        """
        # self.net_hero_main = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(DimConfig.DIM_OF_HERO_MAIN[ii], 64),
        #         nn.ReLU(),
        #         nn.Linear(64, 32),
        #         nn.ReLU(),
        #         nn.Linear(32, 16)
        #     ) for ii in range(len(DimConfig.DIM_OF_HERO_MAIN))
        # ])
        # self.net_hero_emy = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(DimConfig.DIM_OF_HERO_EMY[ii], 512),
        #         nn.ReLU(),
        #         nn.Linear(512, 256),
        #         nn.ReLU(),
        #         nn.Linear(256, 128)
        #     ) for ii in range(len(DimConfig.DIM_OF_HERO_EMY))
        # ])
        # self.net_hero_frd = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(DimConfig.DIM_OF_HERO_FRD[ii], 512),
        #         nn.ReLU(),
        #         nn.Linear(512, 256),
        #         nn.ReLU(),
        #         nn.Linear(256, 128)
        #     ) for ii in range(len(DimConfig.DIM_OF_HERO_FRD))
        # ])
        # self.net_soldier1 = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(DimConfig.DIM_OF_SOLDIER_1_10[ii], 64),
        #         nn.ReLU(),
        #         nn.Linear(64, 64),
        #         nn.ReLU(),
        #         nn.Linear(64, 32)
        #     ) for ii in range(len(DimConfig.DIM_OF_SOLDIER_1_10))
        # ])
        # self.net_soldier2 = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(DimConfig.DIM_OF_SOLDIER_11_20[ii], 64),
        #         nn.ReLU(),
        #         nn.Linear(64, 64),
        #         nn.ReLU(),
        #         nn.Linear(64, 32)
        #     ) for ii in range(len(DimConfig.DIM_OF_SOLDIER_11_20))
        # ])
        # self.net_organ1 = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(DimConfig.DIM_OF_ORGAN_1_2[ii], 64),
        #         nn.ReLU(),
        #         nn.Linear(64, 64),
        #         nn.ReLU(),
        #         nn.Linear(64, 32)
        #     ) for ii in range(len(DimConfig.DIM_OF_ORGAN_1_2))
        # ])
        # self.net_organ2 = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(DimConfig.DIM_OF_ORGAN_3_4[ii], 64),
        #         nn.ReLU(),
        #         nn.Linear(64, 64),
        #         nn.ReLU(),
        #         nn.Linear(64, 32)
        #     ) for ii in range(len(DimConfig.DIM_OF_ORGAN_3_4))
        # ])

        self.public_hero_main = nn.Sequential(nn.Linear(DimConfig.DIM_OF_HERO_MAIN[0], 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.net_hero_main = nn.Sequential(nn.Linear(32, 16))

        self.public_hero = nn.Sequential(nn.Linear(DimConfig.DIM_OF_HERO_EMY[0], 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.net_hero_emy = nn.Sequential(nn.Linear(256, 128))
        self.net_hero_frd = nn.Sequential(nn.Linear(256, 128))

        self.public_soldier = nn.Sequential(nn.Linear(DimConfig.DIM_OF_SOLDIER_1_10[0], 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.net_soldier1 = nn.Sequential(nn.Linear(64, 32))
        self.net_soldier2 = nn.Sequential(nn.Linear(64, 32))

        self.public_organ = nn.Sequential(nn.Linear(DimConfig.DIM_OF_ORGAN_1_2[0], 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.net_organ1 = nn.Sequential(nn.Linear(64, 32))
        self.net_organ2 = nn.Sequential(nn.Linear(64, 32))

        self.pulic_fc = nn.Sequential(nn.Linear(16 + 128 * 2 + 32 * 4 + 25, 512), nn.ReLU())
        self.action_fc = nn.ModuleList(
            [nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, label_size_list[ii])) for ii in range(len(label_size_list))]
        )
        self.target_embed_fc_for_pulic_result = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, Config.TARGET_EMBED_DIM))
        self.target_embed_fc_for_target_embed = nn.Linear(Config.TARGET_EMBED_DIM, Config.TARGET_EMBED_DIM)

        self.state_dim = state_dim
        self.label_size_list = label_size_list

        ### get split feature dim ###
        self.hero_dim = (
            int(np.sum(DimConfig.DIM_OF_HERO_FRD)) + int(np.sum(DimConfig.DIM_OF_HERO_EMY)) + int(np.sum(DimConfig.DIM_OF_HERO_MAIN))
        )  # 484
        self.soldier_dim = int(np.sum(DimConfig.DIM_OF_SOLDIER_1_10)) + int(np.sum(DimConfig.DIM_OF_SOLDIER_11_20))  # 144
        self.organ_dim = int(np.sum(DimConfig.DIM_OF_ORGAN_1_2)) + int(np.sum(DimConfig.DIM_OF_ORGAN_3_4))  # 72
        self.global_info_dim = int(np.sum(DimConfig.DIM_OF_GLOBAL_INFO))  # 25
        self.target_embed_dim = Config.TARGET_EMBED_DIM

        self.lstm = nn.LSTM(512, 512, batch_first=True, num_layers=1)
        self.lstm_steps = 16

        self.device = None

    def forward(self, input, only_inference=False, lstm_cell=None, lstm_hidden=None):
        """
        :param input: bs x 725
        """
        self.batch_size = input.shape[0] if len(input.shape) != 1 else 1  ### for test ###
        input = torch.reshape(input, (self.batch_size, -1))  ### bs, 725 ###

        """
            feature_vec_split
        """
        feature_vec_split_list = torch.split(input, [self.hero_dim, self.soldier_dim, self.organ_dim, self.global_info_dim], dim=1)
        hero_vec_list = torch.split(
            feature_vec_split_list[0],
            [
                int(np.sum(DimConfig.DIM_OF_HERO_FRD)),
                int(np.sum(DimConfig.DIM_OF_HERO_EMY)),
                int(np.sum(DimConfig.DIM_OF_HERO_MAIN)),
            ],
            dim=1,
        )
        soldier_vec_list = torch.split(
            feature_vec_split_list[1],
            [
                int(np.sum(DimConfig.DIM_OF_SOLDIER_1_10)),
                int(np.sum(DimConfig.DIM_OF_SOLDIER_11_20)),
            ],
            dim=1,
        )
        organ_vec_list = torch.split(
            feature_vec_split_list[2],
            [
                int(np.sum(DimConfig.DIM_OF_ORGAN_1_2)),
                int(np.sum(DimConfig.DIM_OF_ORGAN_3_4)),
            ],
            dim=1,
        )
        global_info_list = feature_vec_split_list[3]

        soldier_1_10 = torch.split(soldier_vec_list[0], DimConfig.DIM_OF_SOLDIER_1_10, dim=1)
        soldier_11_20 = torch.split(soldier_vec_list[1], DimConfig.DIM_OF_SOLDIER_11_20, dim=1)
        organ_1_2 = torch.split(organ_vec_list[0], DimConfig.DIM_OF_ORGAN_1_2, dim=1)
        organ_3_4 = torch.split(organ_vec_list[1], DimConfig.DIM_OF_ORGAN_3_4, dim=1)
        hero_frd = torch.split(hero_vec_list[0], DimConfig.DIM_OF_HERO_FRD, dim=1)
        hero_emy = torch.split(hero_vec_list[1], DimConfig.DIM_OF_HERO_EMY, dim=1)
        hero_main = torch.split(hero_vec_list[2], DimConfig.DIM_OF_HERO_MAIN, dim=1)  ### ??? ###
        global_info = global_info_list

        """
            target embedding 
        """
        tar_embed_list = []
        # non_target_embedding
        tar_embed_list.append(0.1 * torch.ones([self.batch_size, self.target_embed_dim]).to(self.device))

        """
            hero_main
        """
        for index in range(len(hero_main)):
            fc3_hero_result = self.net_hero_main(self.public_hero_main(hero_main[index]))
        hero_main_concat_result = fc3_hero_result

        """
            hero_frd
        """
        hero_frd_result_list = []
        for index in range(len(hero_frd)):
            fc3_hero_result = self.net_hero_frd(self.public_hero(hero_frd[index]))
            #  frd_hero_embedding
            _, split_1 = torch.split(fc3_hero_result, [96, self.target_embed_dim], dim=1)
            tar_embed_list.append(split_1)
            hero_frd_result_list.append(fc3_hero_result)
        hero_frd_concat_result = torch.cat(hero_frd_result_list, dim=1)

        """
            hero_emy
        """
        hero_emy_result_list = []
        for index in range(len(hero_emy)):
            fc3_hero_result = self.net_hero_emy(self.public_hero(hero_emy[index]))
            # emy_hero_embedding
            _, split_1 = torch.split(fc3_hero_result, [96, self.target_embed_dim], dim=1)
            tar_embed_list.append(split_1)
            hero_emy_result_list.append(fc3_hero_result)
        hero_emy_concat_result = torch.cat(hero_emy_result_list, dim=1)

        """
            soldier 1 - 10
        """
        soldier_1_result_list = []
        for index in range(len(soldier_1_10)):
            fc3_soldier_result = self.net_soldier1(self.public_soldier(soldier_1_10[index]))
            soldier_1_result_list.append(fc3_soldier_result)
        soldier_1_concat_result = torch.cat(soldier_1_result_list, dim=1).reshape([self.batch_size, len(soldier_1_10), -1]).max(dim=1).values

        """
            soldier 11 - 20
        """
        soldier_2_result_list = []
        for index in range(len(soldier_11_20)):
            fc3_soldier_result = self.net_soldier2(self.public_soldier(soldier_11_20[index]))
            #  emy soldier embedding
            tar_embed_list.append(fc3_soldier_result)
            soldier_2_result_list.append(fc3_soldier_result)
        soldier_2_concat_result = torch.cat(soldier_2_result_list, dim=1).reshape([self.batch_size, len(soldier_11_20), -1]).max(dim=1).values

        """
            organ frd
        """
        organ_1_result_list = []
        for index in range(len(organ_1_2)):
            fc3_organ_result = self.net_organ1(self.public_organ(organ_1_2[index]))
            organ_1_result_list.append(fc3_organ_result)
        organ_1_concat_result = torch.cat(organ_1_result_list, dim=1).reshape([self.batch_size, len(organ_1_2), -1]).max(dim=1).values

        """
            organ_emy
        """
        organ_2_result_list = []
        for index in range(len(organ_3_4)):
            fc3_organ_result = self.net_organ2(self.public_organ(organ_3_4[index]))
            organ_2_result_list.append(fc3_organ_result)
        organ_2_concat_result = torch.cat(organ_2_result_list, dim=1).reshape([self.batch_size, len(organ_3_4), -1]).max(dim=1).values
        ### emy target embedding ###
        tar_embed_list.append(organ_2_concat_result)

        """
            public_concat
        """
        concat_result = torch.cat(
            [
                soldier_1_concat_result,
                soldier_2_concat_result,
                organ_1_concat_result,
                organ_2_concat_result,
                hero_main_concat_result,
                hero_frd_concat_result,
                hero_emy_concat_result,
                global_info,
            ],
            dim=1,
        )

        """
            public fc
        """
        fc_pulic_result_for_lstm = self.pulic_fc(concat_result)

        """
            public lstm
        """
        if not only_inference:
            fc_pulic_result_for_lstm = fc_pulic_result_for_lstm.reshape([self.batch_size // self.lstm_steps, self.lstm_steps, -1])
            fc_pulic_result, _ = self.lstm(fc_pulic_result_for_lstm)
            fc_pulic_result = fc_pulic_result.reshape([self.batch_size, -1])
        else:
            fc_pulic_result_for_lstm = fc_pulic_result_for_lstm.reshape([1, 1, -1])
            fc_pulic_result, (cell, hidden) = self.lstm(fc_pulic_result_for_lstm, (lstm_cell, lstm_hidden))
            fc_pulic_result = fc_pulic_result.reshape([1, -1])

        """
            action layer
        """
        result_list = []
        for index in range(0, len(self.label_size_list) - 1):
            fc2_label_result = self.action_fc[index](fc_pulic_result)
            result_list.append(fc2_label_result)

        """
            target attention
        """
        target_embed_fc = self.target_embed_fc_for_pulic_result(fc_pulic_result).reshape([-1, self.target_embed_dim, 1])  ### bs x 32 x 1
        tar_embedding = torch.stack(tar_embed_list, dim=1)  ### bs x 8 x 32 ###
        ulti_tar_embedding = self.target_embed_fc_for_target_embed(tar_embedding)  ### bs x 8 x 32 ###
        fc3_label_result = torch.bmm(ulti_tar_embedding, target_embed_fc).squeeze()  ### bs x 8 ###
        fc3_label_result = fc3_label_result.reshape(self.batch_size, -1)
        result_list.append(fc3_label_result)

        if only_inference:
            result_list.append(cell)
            result_list.append(hidden)

        # return input, result_list
        return fc_pulic_result, result_list

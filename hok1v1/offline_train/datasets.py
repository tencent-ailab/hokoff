import h5py
import numpy as np
from concurrent import futures
import random
from torch.utils import data
import torch as torch
import time
from train_eval_config.OneConfig import ModelConfig as Config


class Datasets(object):
    def __init__(self, replay_dirs, batch_size, lstm_steps, device, train_step_per_buffer, num_workers, max_step, dataset_name) -> None:
        self.replay_dirs = replay_dirs
        self.batch_size = batch_size
        self.lstm_steps = lstm_steps
        self.device = device
        self.train_step = 0
        self.max_step = max_step
        self.change_step = 50000
        self.first_half = True
        self.dataset_name = dataset_name

        # all_keys = [
        #     'observation',
        #     'action',
        #     'reward',
        #     'done',
        #     'legal_action'
        #     'sub_action'
        # ]
        self.data_split = [
            Config.SERI_VEC_SPLIT_SHAPE[0][0],
            len(Config.LABEL_SIZE_LIST),
            1,
            1,
            sum(Config.LEGAL_ACTION_SIZE_LIST),
            len(Config.LABEL_SIZE_LIST),
        ]
        self.done_index = Config.SERI_VEC_SPLIT_SHAPE[0][0] + len(Config.LABEL_SIZE_LIST) + 1 + 1 - 1
        self.load_dataset()

    def load_dataset(self):
        print('Begin Load All Data...')
        self.all_data_f = h5py.File(self.replay_dirs + '/' + self.dataset_name + '/all_data/all_data.hdf5', 'r')
        total_shape = self.all_data_f['datas'].shape[0]
        if self.train_step == 0:
            load_index = list(range(total_shape // 2))
        else:
            del self.all_data  ### first clear ###
            load_index = list(range(total_shape // 2, total_shape))
            time.sleep(5)
        if self.first_half:
            load_index = list(range(total_shape // 2))
        if 'sub_task' in self.replay_dirs:
            load_index = list(range(total_shape))
        self.all_data = torch.tensor(self.all_data_f['datas'][load_index])
        self.all_data_f.close()
        print('Load All Data Over...')

        self.data_length = self.all_data.shape[0]
        self.data_index = range(self.data_length - self.lstm_steps)

        self.first_half = not self.first_half

    def next_batch(self):
        index = random.sample(self.data_index, self.batch_size)  ### 128 ###
        final_index = []
        for ii in range(len(index)):  ### revise the chosen index ###
            if torch.sum(self.all_data[index[ii] : index[ii] + self.lstm_steps, self.done_index]) > 0:
                while self.all_data[index[ii] + self.lstm_steps - 1, self.done_index] == 0:
                    index[ii] -= 1
            for kk in range(self.lstm_steps):
                final_index.append(index[ii] + kk)
        final_next_index = (1 + np.array(final_index)).clip(max=self.data_length - 1)

        cur_data = torch.split(self.all_data[final_index], self.data_split, dim=1)
        cur_next_data = torch.split(self.all_data[final_next_index], self.data_split, dim=1)

        self.train_step += 1
        # if self.train_step == (self.max_step // 2 ): self.load_dataset()
        if self.train_step % self.change_step == 0:
            self.load_dataset()

        return {
            'observation': cur_data[0].to(self.device),
            'action': cur_data[1].to(self.device),
            'reward': cur_data[2].to(self.device),
            'next_observation': cur_next_data[0].to(self.device),
            'done': cur_data[3].to(self.device),
            'legal_action': cur_data[4].to(self.device),
            'next_legal_action': cur_next_data[4].to(self.device),
            'sub_action': cur_data[5].to(self.device),
        }

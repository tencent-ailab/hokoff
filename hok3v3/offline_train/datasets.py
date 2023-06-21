import h5py
import numpy as np
import random
import os
from torch.utils import data
import torch as th
import threading
import multiprocessing

ctx = multiprocessing.get_context("spawn")


class H5Dataset(data.Dataset):
    def __init__(self, h5file_dir) -> None:
        self.file_dir = h5file_dir
        self.data = {}
        with h5py.File(self.file_dir, 'r') as file:
            self.dataset_len = len(file['done'])
            self.keys = list(file.keys())

    def __getitem__(self, index):
        chosen_data = {}
        if len(self.data.keys()) == 0:
            f = h5py.File(self.file_dir, 'r')
            for key in self.keys:
                self.data[key] = f.get(key)
        for key in self.keys:
            chosen_data[key] = self.data[key][index].astype(np.float32)
        return chosen_data

    def __len__(self):
        return self.dataset_len


class Datasets(object):
    def __init__(self, replay_dirs, batch_size, lstm_steps, device, train_step_per_buffer=1000, num_workers=1) -> None:
        self.replay_dirs = replay_dirs
        self.batch_size = batch_size
        self.lstm_steps = lstm_steps
        self.train_step_per_buffer = train_step_per_buffer
        self.device = device
        self.data_files = []
        self.data_files_flag = {}
        for name in os.listdir(replay_dirs):
            if 'hdf5' in name:
                self.data_files.append(name)
                self.data_files_flag[name] = 0
        self.q = ctx.Queue(20)

        self.h5py_blocks = 10
        self.num_workers = num_workers
        self.data_q = ctx.Queue(self.h5py_blocks * self.num_workers)
        for ii in range(self.num_workers):
            load_worker = ctx.Process(target=self.single_reset, args=(ii,))
            load_worker.daemon = True
            load_worker.start()
        for i in range(1):
            p = ctx.Process(target=self.single_worker, args=(i,))
            p.daemon = True
            p.start()

        self.train_step = 0

    def single_reset(self, k):
        num_buffer_files = len(self.data_files)
        buffer_idx_list = [i * num_buffer_files // self.num_workers for i in range(self.num_workers)] + [num_buffer_files]
        train_step_now = 0
        name_now = np.random.choice(self.data_files[buffer_idx_list[k] : buffer_idx_list[k + 1]], 1, replace=False)[0]
        h5dataset = H5Dataset(os.path.join(self.replay_dirs, name_now))
        h5dataset_len = len(h5dataset)
        base_idx_list = [i * h5dataset_len // self.h5py_blocks for i in range(self.h5py_blocks)] + [h5dataset_len]
        random_idx_list = list(range(self.h5py_blocks))
        random.shuffle(random_idx_list)
        for i in random_idx_list:
            self.data_q.put(h5dataset[np.array(range(base_idx_list[i], base_idx_list[i + 1]))])
        print('dataset reset:', name_now)
        # return total_datas,train_step_now,name_now

    def single_worker(self, k):
        batch_size = self.batch_size
        train_step_now = 0
        total_datas_list = []
        while True:
            if (not self.data_q.empty()) or len(total_datas_list) == 0:
                total_datas = self.data_q.get()
                total_datas_list.append(total_datas)
                total_datas_list = total_datas_list[-self.h5py_blocks * self.num_workers :]
            else:
                total_datas = total_datas_list[np.random.choice(len(total_datas_list), 1)[0]]
            max_sample_idx = total_datas['done'].shape[0]
            # print(max_sample_idx,base_idx_list,random_choice)
            batch_idx = np.random.choice(max_sample_idx, batch_size, replace=False)
            total_done = total_datas['done']
            new_batch_idx = []
            next_new_batch_idx = []
            # if self.lstm_steps>1:
            for ii in range(batch_size):
                invalid_idx = -1
                for jj in range(self.lstm_steps):
                    if batch_idx[ii] + jj + 1 >= max_sample_idx:
                        invalid_idx = jj + 1
                        break
                    if total_done[batch_idx[ii] + jj]:
                        invalid_idx = jj + 1
                        break
                if not invalid_idx == -1:
                    batch_idx[ii] -= self.lstm_steps - invalid_idx
                for jj in range(self.lstm_steps):
                    new_batch_idx.append(batch_idx[ii] + jj)
                if batch_idx[ii] + self.lstm_steps >= max_sample_idx:
                    next_new_batch_idx.append(batch_idx[ii] + self.lstm_steps - 1)
                else:
                    next_new_batch_idx.append(batch_idx[ii] + self.lstm_steps)

            batch_data = {}
            for key in total_datas.keys():
                batch_data[key] = total_datas[key][new_batch_idx].reshape(
                    [batch_size, self.lstm_steps] + list(total_datas[key].shape[1:])
                )  # .to(dtype=th.float32, device=self.device)
                batch_data[key] = th.from_numpy(batch_data[key]).to(dtype=th.float32, device=self.device)
                batch_data['next_' + key] = total_datas[key][next_new_batch_idx].reshape(
                    [batch_size, 1] + list(total_datas[key].shape[1:])
                )  # .to(dtype=th.float32, device=self.device)
                batch_data['next_' + key] = th.from_numpy(batch_data['next_' + key]).to(dtype=th.float32, device=self.device)
            self.q.put(batch_data)
            train_step_now += 1

    def next_batch(self):
        self.train_step += 1
        if self.train_step > self.train_step_per_buffer:
            self.train_step = 0
            for ii in range(self.num_workers):
                load_worker = ctx.Process(target=self.single_reset, args=(ii,))
                load_worker.daemon = True
                load_worker.start()
        batch_data = self.q.get()
        return batch_data

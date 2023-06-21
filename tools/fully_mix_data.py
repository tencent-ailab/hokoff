#!/bin/sh
import argparse
import os
from os.path import dirname, abspath
import numpy as np
import h5py
import random


def parse_args():
    """parsing input command line parameters"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Worker Parser')
    parser.add_argument('--mixed_path', default='', type=str, help="Commands list file location.")
    parser.add_argument('--source_path_list', default='', type=str, help="Commands list file location.")
    # parser.add_argument('--total_num',default=50,dest='tn',type=int,help='how many episodes')

    return parser.parse_args()


args = parse_args()
source_path_list = args.source_path_list.split(',')
source_files_list = []
max_files = 0
episodes_per_file = 0

if not os.path.exists(args.mixed_path):
    os.mkdir(args.mixed_path)

for path in source_path_list:
    files = []
    for file in os.listdir(path):
        if '.hdf5' in file:
            files.append(file)
            f = h5py.File(path + '/' + file, 'r')
            episodes_per_file = max(episodes_per_file, np.sum(f['done']))
    source_files_list.append(files)
    max_files = max(max_files, len(files))

for ii in range(max_files):
    full_episodes_list = []
    for jj in range(len(source_files_list)):
        if len(source_files_list[jj]) >= ii + 1:
            tmp_dict = {}
            f = h5py.File(source_path_list[jj] + '/' + source_files_list[jj][ii], 'r')
            done_index = np.where(f['done'])[0][:-1] + 1
            print(source_path_list[jj] + '/' + source_files_list[jj][ii], np.sum(f['done']), done_index)
            tmp_dict_list = [{}] * (len(done_index) + 1)
            for key in f.keys():
                tmp_list = np.split(f[key], done_index)
                for kk in range(len(tmp_list)):
                    tmp_dict_list[kk][key] = tmp_list[kk]
            full_episodes_list += tmp_dict_list
            f.close()
    random.shuffle(full_episodes_list)
    print('total:', len(full_episodes_list))
    # exit(0)
    episodes_list = full_episodes_list[:episodes_per_file]
    dataset = h5py.File(args.mixed_path + '/' + str(ii) + '_0.hdf5', 'a')
    for episode in episodes_list:
        for key in episode.keys():
            if key not in dataset.keys():
                dataset.create_dataset(
                    key,
                    data=np.array(episode[key]),
                    compression="gzip",
                    maxshape=(None, *(list(episode[key].shape)[1:])),
                    chunks=True,
                )
            else:
                dataset[key].resize((dataset[key].shape[0] + episode[key].shape[0]), axis=0)
                dataset[key][-episode[key].shape[0] :] = np.array(episode[key])

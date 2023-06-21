import h5py
import numpy as np
import os

import argparse

parser = argparse.ArgumentParser(description='Offline 3v3 Train')
parser.add_argument("--root_path", type=str, default="/datasets/3v3version1/level-2-0", help="The root path of the offline information.")
parser.add_argument("--cpu_num", type=int, default=50, help="Cpu num --> Count of total files.")
parser.add_argument("--eval_num", type=int, default=50, help="Eval num in every file...")
args = parser.parse_args()

if __name__ == "__main__":
    max_num = args.cpu_num
    eval_num = args.eval_num
    for i in range(max_num):
        try:
            f = h5py.File(os.path.join(args.root_path, str(i) + '_0.hdf5'), 'r')
            print(i, np.sum(f['done']), eval_num == np.sum(f['done']))
            if eval_num != np.sum(f['done']):  ### remove all wrong datas ###
                if os.path.exists(os.path.join(args.root_path, str(i) + '_0.hdf5')):
                    os.remove(os.path.join(args.root_path, str(i) + '_0.hdf5'))
                    os.remove(os.path.join(args.root_path, str(i) + '_1.hdf5'))
        except:
            if os.path.exists(os.path.join(args.root_path, str(i) + '_0.hdf5')):
                os.remove(os.path.join(args.root_path, str(i) + '_0.hdf5'))
                os.remove(os.path.join(args.root_path, str(i) + '_1.hdf5'))
            print(i, 'wrong')
    count = 0
    for i in range(max_num - 1, -1, -1):
        if os.path.exists(os.path.join(args.root_path, str(i) + '_0.hdf5')):
            os.rename(os.path.join(args.root_path, str(i) + '_0.hdf5'), os.path.join(args.root_path, str(i + count) + '_0.hdf5'))
            os.rename(os.path.join(args.root_path, str(i) + '_1.hdf5'), os.path.join(args.root_path, str(i + count) + '_1.hdf5'))

        else:
            count += 1
    print('Remove Total Count = {}'.format(count))

import h5py
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt

# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['xtick.direction'] = 'in'  # in; out; inout
plt.rcParams['ytick.direction'] = 'in'

# sns.violinplot(x=['first', 'first', 'second', 'second', 'third', 'third'], y=[1,2,3,99,21,33], fontproperties = font_pro)
# plt.plot()
# plt.savefig('/code/tools/test.png', dpi=500, bbox_inches='tight')
# plt.close()

# exit()

# SERI_VEC_SPLIT_SHAPE = [(725,), (84,)]
# LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 8]
# done_index = SERI_VEC_SPLIT_SHAPE[0][0] + len(LABEL_SIZE_LIST) + 1 + 1 - 1

all_all_replay_dirs = [
    #     ['/datasets/3v3version2/norm_poor', '/datasets/3v3version2/norm_medium', '/datasets/3v3version2/norm_expert', '/datasets/3v3version2/norm_mixed'],
    #     ['/datasets/3v3version2/hard_poor', '/datasets/3v3version2/hard_medium', '/datasets/3v3version2/hard_expert', '/datasets/3v3version2/hard_mixed'],
    #     ['/datasets/3v3version2/norm_multi_level', '/datasets/3v3version2/hard_multi_level'],
    #     ['/datasets/3v3version2/norm_general', '/datasets/3v3version2/hard_general'],
    #     ['/datasets/3v3version2/norm_multi_ally', '/datasets/3v3version2/norm_multi_oppo', '/datasets/3v3version2/norm_multi_ally_oppo'],
    #     ['/datasets/3v3version2/norm_stupid_partner', '/datasets/3v3version2/norm_expert_partner', '/datasets/3v3version2/norm_mixed_partner'],
    ['/datasets/3v3version2/gain_gold_medium', '/datasets/3v3version2/gain_gold_expert', '/datasets/3v3version2/gain_gold_mixed'],
]

for all_replay_dirs in all_all_replay_dirs:

    # all_replay_dirs = ['/datasets/3v3version2/hard_poor', '/datasets/3v3version2/hard_medium', '/datasets/3v3version2/hard_expert', '/datasets/3v3version2/hard_mixed']
    # all_replay_dirs = ['/datasets/3v3version2/norm_poor', '/datasets/3v3version2/norm_medium']
    x = [s[s.rindex('/') + 1 :] for s in all_replay_dirs]

    # reward_index = done_index - 1
    gamma = 1.0
    all_dataset_ret = []
    all_dataset_x = []

    propotion_rate = 0.0

    for index, replay_dirs in enumerate(all_replay_dirs):
        all_ret = []
        file_list = os.listdir(replay_dirs)
        for file in file_list:
            if '.hdf5' in file:
                f = h5py.File(replay_dirs + '/' + file, 'r')
                done = np.squeeze(f['done'])  # bs
                reward = np.squeeze(np.mean(f['reward_s'], axis=1))  # bs
                ii = done.shape[0] - 1
                while ii > 0:
                    if done[ii]:
                        cur_ret = reward[ii]
                        while ii - 1 >= 0 and done[ii - 1] == 0:
                            cur_ret = reward[ii - 1] + gamma * cur_ret
                            ii -= 1
                        ii -= 1
                        all_ret.append(cur_ret)

        all_ret.sort()

        if propotion_rate != 0:
            all_ret = all_ret[int(propotion_rate * len(all_ret)) : -int(propotion_rate * len(all_ret))]

        all_dataset_ret += all_ret
        all_dataset_x += [x[index]] * len(all_ret)

    # if 'sub_task' in replay_dirs:
    #     all_dataset_ret = np.array(all_dataset_ret)
    #     all_dataset_ret = 1 - (all_dataset_ret - all_dataset_ret.min()) / (all_dataset_ret.max() - all_dataset_ret.min() + 1e-5)
    # else:
    all_dataset_ret = np.array(all_dataset_ret)
    # if 'gain_gold' in replay_dirs:
    #     all_dataset_ret = (all_dataset_ret - all_dataset_ret.min()) / (all_dataset_ret.max() - all_dataset_ret.min() + 1e-5)

    sns.violinplot(x=all_dataset_x, y=all_dataset_ret)
    plt.ylabel('Episode Return')
    plt.plot()
    plt.savefig('/code/tools/' + '_'.join(x) + '_gamma{}_remove_{}.png'.format(gamma, propotion_rate), dpi=500, bbox_inches='tight')
    plt.close()

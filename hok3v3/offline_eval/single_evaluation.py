import numpy as np
import sys
import re

sys.path.append('.')
import os.path as osp
import datetime
import os

# from actor import Actor
# from agent import Agent
from torch.utils.tensorboard import SummaryWriter
import time
import argparse
import pandas as pd

gc_server_addr = "localhost:23432"
ai_server_ip = '127.0.0.1'


def count_result(root_path, run, train_step):
    wr_dir = osp.join(root_path, run, "win_rate_details")
    all_wr_txt = os.listdir(wr_dir)
    wr_dic = {}
    for wr_txt in all_wr_txt:
        if wr_txt.startswith('offline_win_rate_'):
            with open(osp.join(wr_dir, wr_txt), 'r') as f:
                lines = f.readlines()
            tmp_wr = -1
            for line in lines:
                try:
                    model_name = re.findall('model_iter: \d+', line)[0].split(' ')[1]
                    wr = float(re.findall('win_rate: \d+\.?\d*', line)[0][10:])
                    if int(model_name) == train_step:
                        # if model_name in wr_dic.keys():
                        #     wr_dic[model_name].append(wr)
                        # else:
                        #     wr_dic[model_name] = [wr]
                        tmp_wr = wr
                except:
                    print("Parse error for line:", line)
            if tmp_wr != -1:
                if train_step in wr_dic.keys():
                    wr_dic[train_step].append(tmp_wr)
                else:
                    wr_dic[train_step] = [tmp_wr]
    model_list = []
    win_rate_list = []
    if len(list(wr_dic.keys())) > 0:
        file_path = osp.join(root_path, run, 'eval', 'win_rate_all.txt')
        with open(file_path, 'w') as f:
            for k, v in wr_dic.items():
                print(k, 'win_rate:', sum(v) / len(v), file=f)
                model_list.append(k)
                win_rate_list.append(sum(v) / len(v))
        print("Win rate saved to", file_path)
    else:
        print("Find no win rate txt!")

    return model_list, win_rate_list


def evaluate(root_path, run_prefix, levels, eval_num, cpu_num=4, final_test=False, tensorflow_oppo=False, max_steps=500000, dataset_name='level-0-0'):

    eval_log_len = 0
    model_step_list = []
    model_pool_path = os.path.join(root_path, run_prefix)
    if os.path.exists(os.path.join(model_pool_path, 'eval', 'eval.log')):
        os.remove(os.path.join(model_pool_path, 'eval', 'eval.log'))
    train_step_count = -1
    tb_summary_list = SummaryWriter(log_dir=os.path.join(root_path, run_prefix, 'eval'))
    final_win_rate_list = []
    final_model_num = []
    done_flag = 0
    while True:
        if done_flag:
            break
        model_step_list = []
        model_pool_path = os.path.join(root_path, run_prefix)
        for file_name in os.listdir(model_pool_path):
            if '_model' in file_name:
                model_step = int(file_name.split('_')[0])
                model_step_list.append(model_step)
        model_step_list.sort()
        if len(model_step_list) == 0:
            time.sleep(0.1)
            continue

        if final_test:
            if model_step_list[-1] < max_steps:
                break
            new_model_step_list = []
            for i in range(len(model_step_list)):
                if model_step_list[i] % 1000 == 0:
                    new_model_step_list.append(model_step_list[i])
                    new_model_step_list = new_model_step_list[-3:]
            for i in range(len(new_model_step_list)):
                if new_model_step_list[i] > train_step_count:
                    train_step_count = new_model_step_list[i]
                    break
        else:
            if model_step_list[-1] > train_step_count:
                train_step_count = model_step_list[-1]
            else:
                if train_step_count >= max_steps:
                    done_flag = 1
                continue
        start_time = start_time = time.time()
        current_path = os.path.dirname(os.path.abspath(__file__))
        os.system(
            "sh "
            + current_path
            + "/scripts/start_eval.sh {} {} {} {} {} {} {} {} {} >> {} 2>&1 &".format(
                levels,
                eval_num,
                cpu_num,
                train_step_count,
                run_prefix,
                int(run_prefix.split('_')[-1]),
                root_path,
                tensorflow_oppo,
                dataset_name,
                os.path.join(model_pool_path, 'eval', 'eval.log'),
            )
        )
        start_time = time.time()
        while True:
            if not os.path.exists(os.path.join(model_pool_path, 'eval', 'eval.log')):
                time.sleep(0.1)
                continue
            with open(os.path.join(model_pool_path, 'eval', 'eval.log'), 'r') as f:
                lines = f.readlines()
            if len(lines) > eval_log_len:
                if 'All actors are done' in lines[-1]:
                    eval_log_len = len(lines)
                    break
            if time.time() - start_time > 60 * 7 * eval_num:
                break
        model_list, win_rate_list = count_result(root_path, run_prefix, train_step_count)
        print(
            f"======evaluate exp {run_prefix}: model {train_step_count} level: {levels} win rate: {0 if len(win_rate_list)==0 else np.mean(win_rate_list)} total evaluation time: {(time.time()-start_time)/60} min========="
        )
        if len(win_rate_list) > 0:
            tb_summary_list.add_scalar('eval/win_rate', win_rate_list[0], train_step_count)
            final_win_rate_list.append(win_rate_list[0])
            final_model_num.append(train_step_count)
        else:
            # final_win_rate_list.append(-1)
            final_model_num.append(train_step_count)
        if final_test and train_step_count >= max_steps:
            if len(final_win_rate_list) == 0:
                final_win_rate_list = [-1]
            save_win_rate_to_excel(root_path, [run_prefix], [levels], [dataset_name], [np.mean(final_win_rate_list)], [train_step_count])
            break


def save_win_rate_to_excel(root_path, run_prefix_list, levels_list, dataset_name_list, win_rate_list, model_num_list):
    excel_dir = os.path.join(root_path, 'final_win_rate.xlsx')
    data = {}
    data['run_prefix'] = np.array(run_prefix_list)
    data['levels'] = np.array(levels_list)
    data['dataset'] = np.array(dataset_name_list)
    data['final_win_rate'] = np.array(win_rate_list)
    data['final_model_num'] = np.array(model_num_list)
    sheet_name = 'test'
    if not os.path.exists(excel_dir):
        print('Create excel file:{}'.format(excel_dir))
        writer = pd.ExcelWriter(excel_dir)
        add_data_df = pd.DataFrame(data, index=list(range(data['run_prefix'].shape[0])))
        add_data_df.to_excel(writer, sheet_name=sheet_name)
        writer.save()
    else:
        print('add data to excel file:{}'.format(excel_dir))

        data_df = pd.read_excel(excel_dir, sheet_name=sheet_name, index_col=False, header=0)
        data['run_prefix'] = np.concatenate([data_df.values[:, 1], data['run_prefix']], axis=-1)
        data['levels'] = np.concatenate([data_df.values[:, 2], data['levels']], axis=-1)
        data['dataset'] = np.concatenate([data_df.values[:, 3], data['dataset']], axis=-1)
        data['final_win_rate'] = np.concatenate([data_df.values[:, 4], data['final_win_rate']], axis=-1)
        data['final_model_num'] = np.concatenate([data_df.values[:, 5], data['final_model_num']], axis=-1)
        add_data_df = pd.DataFrame(data, index=list(range(data['run_prefix'].shape[0])))
        writer = pd.ExcelWriter(excel_dir)
        add_data_df.to_excel(writer, sheet_name=sheet_name)
        print(add_data_df.values)
        # data_df.append(add_data_df,ignore_index=True)
        # print(data_df.values)
        # add_data_df.to_excel(writer,sheet_name=sheet_name)
        writer.save()

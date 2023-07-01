import os

os.environ['dataop'] = os.path.join(os.path.dirname(__file__), 'lib')
from train_eval_framework.config_control import ConfigControl
from train_eval_framework.log_manager import LogManager
import argparse
import numpy as np
import torch as th
import random

parser = argparse.ArgumentParser(description='Offline 3v3 Train')
parser.add_argument("--root_path", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)),"offline_logs"), help="The root path of the offline information.")
parser.add_argument("--replay_dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)),"/datasets/3v3version1/"), help="The root path of the datasets.")
parser.add_argument("--dataset_name", type=str, default="norm_medium", help="dataset name.")
parser.add_argument("--run_prefix", type=str, default="run_indbc_0", help="The run prefix of the offline exp.")
parser.add_argument("--levels", type=str, default="0", help="The levels of the opponents.")
parser.add_argument("--cql_alpha", type=float, default=10.0, help="alpha")
parser.add_argument("--omar_coe", type=float, default=0.5, help="omar coe")
parser.add_argument("--gamma", type=float, default=0.99, help="gamma")
parser.add_argument("--batch_size", type=int, default=512, help="batch_size")
parser.add_argument("--lr", type=float, default=1e-4, help="lr")
parser.add_argument("--lstm_time_steps", type=int, default=1, help="lstm_time_steps")
parser.add_argument("--use_lstm", action='store_true', help="use lstm")
parser.add_argument("--cpu_num", type=int, default=1, help="cpu_num")
parser.add_argument("--eval_num", type=int, default=0, help="eval_num")
parser.add_argument("--thread_num", type=int, default=4, help="thread_num")
parser.add_argument("--target_update_freq", type=int, default=2000, help="target_update_freq")
parser.add_argument("--soft_update_tau", type=float, default=0.995, help="soft_update_rate")
parser.add_argument("--max_steps", type=int, default=500000, help="max_steps")
parser.add_argument("--train_step_per_buffer", type=int, default=1000, help="max_steps")
parser.add_argument("--buffer_num_workers", type=int, default=2, help="buffer_num_workers")
args = parser.parse_args()
th.set_num_threads(args.thread_num)
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "train_eval_framework", "common.conf")
    config_manager = ConfigControl(config_path)
    config_manager.batch_size = args.batch_size
    config_manager.save_model_dir = os.path.join(args.root_path, args.run_prefix)
    config_manager.max_steps = args.max_steps
    os.makedirs(config_manager.save_model_dir, exist_ok=True)
    os.makedirs(args.replay_dir, exist_ok=True)

    if not os.path.exists(os.path.join(args.replay_dir, args.dataset_name)):
        try:
            if not os.path.exists(args.replay_dir):
                os.makedirs(args.replay_dir)
            print('Downloading pre-collected {}'.format(args.dataset_name))
            os.system(
                'cd {}; wget  https://kaiwu-assets-1258344700.cos.ap-shanghai.myqcloud.com/paper/hok-offline/3v3/{}.zip'.format(
                    args.replay_dir, args.dataset_name
                )
            )
            os.system('cd {}; unzip {}.zip'.format(args.replay_dir, args.dataset_name))
            os.system('cd {}; rm {}.zip'.format(args.replay_dir, args.dataset_name))
        except:
            print('There is no pre-collected dataset named {}!!!!'.format(args.dataset_name))
            exit(0)
    else:
        print('Dataset {} exists'.format(args.dataset_name))

    if 'mixed' in args.replay_dir:
        args.train_step_per_buffer = 500
    if 'gain_gold' in args.dataset_name:
        args.max_steps = 100000

    from benchmark import Benchmark
    from networkmodel.pytorch import REGISTRY

    network = REGISTRY[args.run_prefix.split('_')[1]](args=args)
    bench = Benchmark(args, network, config_manager, LogManager)
    bench.run()

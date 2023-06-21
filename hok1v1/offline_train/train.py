import os

os.environ['dataop'] = os.path.join(os.path.dirname(__file__), 'lib')
from train_eval_framework.config_control import ConfigControl
from train_eval_framework.log_manager import LogManager
import argparse
import numpy as np
import torch as th
import random

parser = argparse.ArgumentParser(description='Offline 1v1 Train')
parser.add_argument("--root_path", type=str, default="/code/offline_logs", help="The root path of the offline information.")
parser.add_argument("--replay_dir", type=str, default="/datasets/version5/", help="The root path of the datasets.")
parser.add_argument("--dataset_name", type=str, default="norm_medium", help="The name of the dataset.")
parser.add_argument("--run_prefix", type=str, default="run_1v1bc", help="The run prefix of the offline exp.")
parser.add_argument("--levels", type=str, default="1,1", help="The levels of the agents.")
parser.add_argument("--gamma", type=float, default=0.99, help="gamma")
parser.add_argument("--batch_size", type=int, default=128, help="batch_size")
parser.add_argument("--lr", type=float, default=3e-4, help="lr")
parser.add_argument("--lstm_time_steps", type=int, default=16, help="lstm_time_steps")
parser.add_argument("--cpu_num", type=int, default=1, help="cpu_num")
parser.add_argument("--eval_num", type=int, default=0, help="eval_num")
parser.add_argument("--thread_num", type=int, default=4, help="thread_num")
parser.add_argument("--max_steps", type=int, default=500000, help="max_steps")
parser.add_argument("--train_step_per_buffer", type=int, default=1000, help="max_steps")
parser.add_argument("--buffer_num_workers", type=int, default=1, help="buffer_num_workers")

parser.add_argument("--target_update_freq", type=int, default=1, help="target update freq")
parser.add_argument("--cql_alpha", type=float, default=10.0, help="cql alpha")
parser.add_argument("--bc_constrain", type=float, default=1.0, help="td3bc alpha")
parser.add_argument("--tau", type=float, default=0.995, help="target net update")
parser.add_argument("--actor_update_freq", type=int, default=1, help="actor update frequency")
parser.add_argument("--td3bc_alpha", type=float, default=2.5, help="td3bc alpha")
parser.add_argument("--iql_tau", type=float, default=0.7, help="iql tau")
parser.add_argument("--iql_beta", type=float, default=3.0, help="iql beta")
parser.add_argument("--iql_max_advantage_clip", type=float, default=100.0, help="iql max advantage clip")
parser.add_argument("--use_sub_action", type=int, default=1, help="sub action mask")
parser.add_argument("--use_lstm", type=int, default=1, help="whether to use lstm")
parser.add_argument("--sac_alpha", type=float, default=1.0, help="sac alpha")

args = parser.parse_args()
th.set_num_threads(args.thread_num)

if __name__ == "__main__":
    seed = random.randint(0, 100000)
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    if not os.path.exists(os.path.join(args.replay_dir, args.dataset_name)):
        try:
            if not os.path.exists(args.replay_dir):
                os.makedirs(args.replay_dir)
            print('Downloading pre-collected {}'.format(args.dataset_name))
            os.system(
                'cd {}; wget  https://kaiwu-assets-1258344700.cos.ap-shanghai.myqcloud.com/paper/hok-offline/1v1/{}.zip'.format(
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

    config_path = os.path.join(os.path.dirname(__file__), "train_eval_framework", "common.conf")

    config_manager = ConfigControl(config_path)
    config_manager.batch_size = args.batch_size
    config_manager.save_model_dir = os.path.join(args.root_path, args.run_prefix)
    config_manager.max_steps = args.max_steps
    os.makedirs(config_manager.save_model_dir, exist_ok=True)
    os.makedirs(config_manager.train_dir, exist_ok=True)
    os.makedirs(config_manager.send_model_dir, exist_ok=True)

    from benchmark import Benchmark
    from networkmodel.pytorch import REGISTRY

    network = REGISTRY[args.run_prefix.split('_')[1]](args=args)

    bench = Benchmark(args, network, config_manager, LogManager)
    bench.run()

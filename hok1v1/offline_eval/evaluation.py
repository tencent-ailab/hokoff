import numpy as np

from threading import Thread

# from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

import re

# sys.path.pop()
sys.path.append('.')
# from model_config import ModelConfig
import threading

# from model import Model
# from networkmodel.pytorch.BaseModel import BaseModel
# import torch
# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)
import os.path as osp
import datetime
import os

# from actor import Actor
# from agent import Agent
from torch.utils.tensorboard import SummaryWriter
import time
import argparse
from offline_eval.single_evaluation import evaluate as single_evaluate

parser = argparse.ArgumentParser(description='Offline 3v3 Train')

parser.add_argument("--root_path", type=str, default="/code/offline_logs", help="The root path of the offline information.")
parser.add_argument("--run_prefix", type=str, default="run_10086", help="The run prefix of the offline exp.")
parser.add_argument("--levels", type=str, default="0,0", help="The levels of the agents.")
# parser.add_argument("--train_step",type=int,default=0,help='trainning step')
parser.add_argument("--cpu_num", type=int, default=1, help="cpu_num")
parser.add_argument("--eval_num", type=int, default=1, help="eval_num")
parser.add_argument("--final_test", type=int, default=0, help="eval_num")
parser.add_argument("--tensorflow_oppo", type=int, default=1, help="use tensorflow eval")
parser.add_argument("--max_steps", type=int, default=500000, help="eval_num")
parser.add_argument("--dataset_name", type=str, default='level-0-0', help="eval_num")
args = parser.parse_args()


gc_server_addr = "localhost:23432"
ai_server_ip = '127.0.0.1'
if __name__ == "__main__":
    
    root_path = args.root_path
    run_prefix = args.run_prefix
    levels = args.levels
    cpu_num = args.cpu_num
    eval_num = args.eval_num
    run_list = run_prefix.split(';')
    levels_list = levels.split(';')
    max_steps = args.max_steps
    assert len(run_list) == len(levels_list)

    single_evaluate(root_path, run_prefix, levels, eval_num, cpu_num, args.final_test, args.tensorflow_oppo, max_steps, args.dataset_name)

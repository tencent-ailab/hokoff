import sys
from hok.hok3v3.gamecore_client import GameCoreClient as Environment
from absl import app as absl_app
from absl import flags
import random
sys.path.append('.')
from baselinemodel.model_config import ModelConfig
from baselinemodel.model import Model
from baselinemodel.tensorflowmodel import Model as tfmodel
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import os
from actor import Actor

# from hok.gamecore.kinghonour.agent import Agent as Agent
from agent import Agent

# from hok.algorithms.model.sample_manager import SampleManager as SampleManager
from utils.ensure_path_exist import ensure_path_exist
from utils.log_win_rate import log_win_rate

FLAGS = flags.FLAGS

flags.DEFINE_integer("actor_id", 0, "actor id")
flags.DEFINE_integer("i", 0, "seed")
flags.DEFINE_string("gc_server_addr", "localhost:23432", "address of gamecore server")
flags.DEFINE_string("ai_server_ip", "localhost", "host of ai_server")
flags.DEFINE_integer("thread_num", 1, "thread_num")
flags.DEFINE_string("dataset_path", "", "dataset save path")
flags.DEFINE_string("agent_models", "", "agent_model_list")
flags.DEFINE_integer("eval_number", -1, "battle number for evaluation")
# flags.DEFINE_integer("max_episode", -1, "max number for run episode")
flags.DEFINE_string("run_prefix", 'run_10086', "run_prefix")
flags.DEFINE_integer("train_step", 0, "train_step")
flags.DEFINE_string("levels", '0', "max number for run episode")
# flags.DEFINE_string("monitor_server_addr", "127.0.0.1:8086", "monitor server addr")
flags.DEFINE_string("offline_log_path", "/code/offline_logs", "log path for offline information")
flags.DEFINE_bool("tensorflow_oppo", 0, "use tensorflow oppo")
flags.DEFINE_string("dataset_name", 'level-0-0', "use tensorflow oppo")
AGENT_NUM = 2
import time
import torch as th

# gamecore as lib
def gc_as_lib(argv):
    # TODO: used for different process
    thread_id = 0
    actor_id = FLAGS.thread_num * FLAGS.i + thread_id
    agents = []

    eval_number = FLAGS.eval_number
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if int(FLAGS.levels) == -1:
        enemy_model = 'common_ai'
    else:
        if FLAGS.tensorflow_oppo:
            enemy_model = root_dir + '/baselines/tensorflow/level-' + FLAGS.levels + '/code/actor/model/init'
        else:
            enemy_model = root_dir + '/baselines/pytorch/level-' + FLAGS.levels + '/code/actor/model/init'
    load_models = [FLAGS.agent_models, enemy_model]
    print('!!!', load_models)
    for i, m in enumerate(load_models):
        if m == "common_ai":
            load_models[i] = None
    eval_mode = True
    if any(FLAGS.dataset_path):
        dataset_path = [FLAGS.dataset_path[:-5] + "_" + str(i) + ".hdf5" for i in range(AGENT_NUM)]
        for path in dataset_path:
            ensure_path_exist(path[: path.rfind('/')])
    else:
        dataset_path = ["" for i in range(AGENT_NUM)]
    from networkmodel.pytorch.module import INFER_REGISTRY

    agents.append(
        Agent(
            INFER_REGISTRY[FLAGS.run_prefix.split('_')[1]](),
            keep_latest=True,
            local_mode=eval_mode,
            backend=ModelConfig.backend,
            dataset=dataset_path[0],
        )
    )
    agents.append(
        Agent(
            tfmodel(ModelConfig) if FLAGS.tensorflow_oppo else Model(ModelConfig),
            keep_latest=False,
            local_mode=eval_mode,
            backend='tensorflow' if FLAGS.tensorflow_oppo else "pytorch",
            dataset=dataset_path[1],
        )
    )
    win_list = []
    extra_add_seed = 0
    start_time = time.time()
    repeat_time = 0
    while len(win_list) == 0:
        env = Environment(host=FLAGS.ai_server_ip, seed=FLAGS.actor_id + extra_add_seed, gc_server=FLAGS.gc_server_addr)
        if 'gain_gold' in FLAGS.dataset_name:
            sub_task = 'gain_gold'
        else:
            sub_task = None
        actor = Actor(id=FLAGS.actor_id + extra_add_seed, agents=agents, monitor_logger=None, sub_task=sub_task)
        # actor.set_sample_managers(sample_manager)
        actor.set_env(env)
        win_list = actor.run(eval_mode=eval_mode, eval_number=eval_number, load_models=load_models, dataset_name=FLAGS.dataset_name)
        extra_add_seed += 10000
        repeat_time += 1
        if repeat_time == 3 and len(win_list) == 0:
            win_list = [-1]
            break

    win_rate = sum(win_list) / len(win_list)
    if win_rate != -1:
        log_win_rate(os.path.join(FLAGS.offline_log_path, FLAGS.run_prefix, 'win_rate_details'), actor_id, FLAGS.train_step, win_rate)


if __name__ == '__main__':
    absl_app.run(gc_as_lib)

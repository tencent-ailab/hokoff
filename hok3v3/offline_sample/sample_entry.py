import sys

sys.path.append('.')
# sys.path.append('./lib')
from absl import app as absl_app
from absl import flags
from baselinemodel.model_config import ModelConfig
from baselinemodel.tensorflowmodel import Model as tfModel
from baselinemodel.model import Model as thModel
import torch
import os
from actor import Actor
from agent import Agent
from utils.ensure_path_exist import ensure_path_exist

FLAGS = flags.FLAGS

flags.DEFINE_integer("actor_id", 0, "actor id")
# flags.DEFINE_integer("i", 0, "seed")
flags.DEFINE_string("gc_server_addr", "localhost:23432", "address of gamecore server")
flags.DEFINE_string("ai_server_ip", "localhost", "host of ai_server")
flags.DEFINE_integer("thread_num", 1, "thread_num")
flags.DEFINE_string("dataset_path", "", "dataset save path")
flags.DEFINE_string("agent_models", "", "agent_model_list")
flags.DEFINE_integer("eval_number", -1, "battle number for evaluation")
# flags.DEFINE_integer("max_episode", -1, "max number for run episode")
# flags.DEFINE_string("monitor_server_addr", "127.0.0.1:8086", "monitor server addr")
# flags.DEFINE_string("offline_log_path", "/code/offline/offline_logs", "log path for offline information")
flags.DEFINE_string("backend", "pytorch", "log path for offline information")
flags.DEFINE_string("dataset_name", "tmplevel-0-0", "log path for offline information")
AGENT_NUM = 2
import time
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
# gamecore as lib
def gc_as_lib(argv):
    print('backend:', FLAGS.backend)
    from hok.hok3v3.gamecore_client import GameCoreClient as Environment
    
    agents = []

    eval_number = FLAGS.eval_number
    load_models = FLAGS.agent_models.split(',')
    print('!!!', load_models)
    for i, m in enumerate(load_models):
        if m == "common_ai":
            load_models[i] = None
    eval_mode = True

    if any(FLAGS.dataset_path):
        dataset_path = [FLAGS.dataset_path[:-5] + "_" + str(i) + ".hdf5" for i in range(AGENT_NUM)]
        for path in dataset_path:
            ensure_path_exist(path[: path.rfind('/')])
        if 'gain_gold' in FLAGS.dataset_name:
            dataset_path[1] = ""
    else:
        dataset_path = ["" for i in range(AGENT_NUM)]

    if ',' not in FLAGS.backend:
        backend_list = [FLAGS.backend] * 2
    else:
        backend_list = FLAGS.backend.split(',')

    assert backend_list[0] == 'tensorflow' or backend_list[0] == 'pytorch'
    # if load_models
    agents.append(
        Agent(
            tfModel(ModelConfig) if backend_list[0] == 'tensorflow' else thModel(ModelConfig),
            keep_latest=True,
            local_mode=eval_mode,
            backend=backend_list[0],
            dataset=dataset_path[0],
        )
    )
    assert backend_list[1] == 'tensorflow' or backend_list[1] == 'pytorch'
    agents.append(
        Agent(
            tfModel(ModelConfig) if backend_list[1] == 'tensorflow' else thModel(ModelConfig),
            keep_latest=False,
            local_mode=eval_mode,
            backend=backend_list[1],
            dataset=dataset_path[1],
        )
    )

    env = Environment(host=FLAGS.ai_server_ip, seed=FLAGS.actor_id, gc_server=FLAGS.gc_server_addr)
    # env = None
    if 'gain_gold' in FLAGS.dataset_name:
        sub_task = 'gain_gold'
    else:
        sub_task = None
    actor = Actor(id=FLAGS.actor_id, agents=agents, monitor_logger=None, sub_task=sub_task, sample_data=True)
    # actor.set_sample_managers(sample_manager)
    actor.set_env(env)
    actor.run(eval_mode=eval_mode, eval_number=eval_number, load_models=load_models, dataset_name=FLAGS.dataset_name)


if __name__ == '__main__':
    absl_app.run(gc_as_lib)

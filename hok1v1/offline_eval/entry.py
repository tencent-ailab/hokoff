import sys

sys.path.append(".")

import random
import re

from absl import app as absl_app
from absl import flags

from agent import Agent as Agent
from actor import Actor
from baselinemodel.model import Model as BaselineModel
from networkmodel.pytorch.module.OneBaseModel import OneBaseModel as Model
from config.config import Config
from utils.ensure_path_exist import ensure_path_exist
from utils.log_win_rate import log_win_rate

FLAGS = flags.FLAGS

flags.DEFINE_integer("actor_id", 0, "actor id")
flags.DEFINE_integer("i", 0, "seed")
flags.DEFINE_integer("max_step", 500, "max step of one round")
flags.DEFINE_integer("train_step", 0, "max step of one round")
flags.DEFINE_string("mem_pool_addr", "localhost:35200", "address of memory pool")
flags.DEFINE_string("model_pool_addr", "localhost:10016", "address of model pool")
flags.DEFINE_string("gamecore_ip", "localhost", "address of gamecore")
flags.DEFINE_integer("thread_num", 1, "thread_num")
flags.DEFINE_boolean("use_lstm", 1, "if use_lstm")

flags.DEFINE_string("agent_models", "", "agent_model_list")
flags.DEFINE_string("dataset_path", "", "dataset save path")
flags.DEFINE_string("levels", "1,1", "levels")

flags.DEFINE_integer("eval_number", -1, "battle number for evaluation")

flags.DEFINE_string("gamecore_path", "~/.hok", "installation path of gamecore")
flags.DEFINE_string("game_log_path", "/code/logs/game_log", "log path for game information")
flags.DEFINE_string("offline_log_path", "/code/offline_logs", "log path for offline information")
flags.DEFINE_string("run_prefix", "run_indbc_0", "log path for offline information")
flags.DEFINE_string("dataset_name", 'level-0-0', "dataset name")

MAP_SIZE = 100
AGENT_NUM = 2
import torch as th

#  gamecore as lib
def gc_as_lib(argv):
    from hok.hok1v1 import HoK1v1

    thread_id = 0
    actor_id = FLAGS.thread_num * FLAGS.i + thread_id
    seed = FLAGS.actor_id
    th.set_num_threads(FLAGS.thread_num)
    agents = []
    game_id_init = "None"
    main_agent = random.randint(0, 1)

    eval_number = FLAGS.eval_number
    if FLAGS.levels.split(',')[1] == -1:
        enemy_model = 'common_ai'
        enemy_level = -1
    else:
        enemy_level = FLAGS.levels.split(',')[1]
        enemy_model = '../baselines/tensorflow/level-' + FLAGS.levels.split(',')[1] + '/algorithms/checkpoint'

    load_models = [FLAGS.agent_models, enemy_model]

    print('Enemy level:', enemy_level)
    print(load_models)
    for i, m in enumerate(load_models):
        if m == "common_ai":
            load_models[i] = None
    eval_mode = eval_number > 0

    import os
    gc_server_addr = os.getenv("GAMECORE_SERVER_ADDR")
    ai_server_addr = os.getenv("AI_SERVER_ADDR")
    if gc_server_addr is None or len(gc_server_addr) == 0 or "127.0.0.1" in gc_server_addr:
        # local gc server
        gc_server_addr = "127.0.0.1:23333"
        ai_server_addr = "127.0.0.1"
        remote_mode = 1
    else:
        # remote gc server
        remote_mode = 2
    remote_param = {
        "remote_mode": remote_mode,
        "gc_server_addr": gc_server_addr,
        "ai_server_addr": ai_server_addr,
    }
    gc_mode = os.getenv("GC_MODE")
    if gc_mode == "local":
        remote_param = None
    env = HoK1v1.load_game(
        runtime_id=seed,
        gamecore_path=FLAGS.gamecore_path,
        game_log_path=FLAGS.game_log_path,
        eval_mode=True,
        config_path="config.dat",
        remote_param=remote_param,
    )
    if any(FLAGS.dataset_path):
        dataset_path = [FLAGS.dataset_path[:-5] + "_" + str(i) + ".hdf5" for i in range(AGENT_NUM)]
        for path in dataset_path:
            ensure_path_exist(path[:-8])
    else:
        dataset_path = ["" for i in range(AGENT_NUM)]
    agents.append(Agent(Model, keep_latest=1, dataset=dataset_path, backend='pytorch'))
    agents.append(Agent(BaselineModel, keep_latest=0, dataset=dataset_path, backend='tensorflow'))

    actor = Actor(id=seed, agents=agents, gpu_ip=FLAGS.mem_pool_addr.split(":")[0])
    actor.set_env(env)
    
    import os 
    cur_dir_name = os.path.dirname(os.path.realpath(__file__))
    cur_dir_dir_name = os.path.dirname(cur_dir_name)
    if 'sub_task' in FLAGS.dataset_name:
        actor.sub_task = True
    if 'multi_hero_oppo' in FLAGS.dataset_name:
        env_config_path = '{}/offline_eval/hero_config/multi_hero_oppo_config/hero_config_1.json,{}/offline_eval/hero_config/multi_hero_oppo_config/hero_config_2.json'.format(cur_dir_name, cur_dir_name)
    elif 'multi_oppo' in FLAGS.dataset_name:
        env_config_path = '{}/hero_config/multi_oppo_config/hero_config_1.json,{}/hero_config/multi_oppo_config/hero_config_2.json'.format(cur_dir_name, cur_dir_name)
    elif 'multi_hero' in FLAGS.dataset_name:
        env_config_path = '{}/hero_config/multi_hero_config/hero_config_1.json,{}/hero_config/multi_hero_config/hero_config_2.json'.format(cur_dir_name, cur_dir_name)
    elif 'hard_expert' in FLAGS.dataset_name:
        env_config_path = '{},{}'.format(
            '{}/baselines/tensorflow/level-7/hero_config.json'.format(cur_dir_dir_name), load_models[1][: -len('algorithms/checkpoint')] + 'hero_config.json'
        )
    else:
        env_config_path = '{},{}'.format(
            load_models[1][: -len('algorithms/checkpoint')] + 'hero_config.json', load_models[1][: -len('algorithms/checkpoint')] + 'hero_config.json'
        )

    offline_win_rate = actor.run(eval_mode=eval_mode, eval_number=eval_number, load_models=load_models, env_config_path=env_config_path)
    log_win_rate(os.path.join(FLAGS.offline_log_path, FLAGS.run_prefix, 'win_rate_details'), actor_id, FLAGS.train_step, offline_win_rate)


if __name__ == "__main__":
    absl_app.run(gc_as_lib)

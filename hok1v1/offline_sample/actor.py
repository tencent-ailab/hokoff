# -*- coding: utf-8 -*-
"""
    KingHonour Data production process
"""
import os
import traceback
import time
import logging
import json

from collections import deque
import numpy as np
from config.config import Config
from framework.common_log import CommonLogger
from framework.common_log import g_log_time
from framework.common_func import log_time_func

import random
from config.common_config import ModelConfig

IS_TRAIN = Config.IS_TRAIN
LOG = CommonLogger.get_logger()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
OS_ENV = os.environ
IS_DEV = OS_ENV.get("IS_DEV")


class Actor:
    """
    used for sample logic
        run 1 episode
        save sample in sample manager
    """

    # ALL_CONFIG_DICT = {
    #     "luban": [{"hero": "luban", "skill": "frenzy"} for _ in range(2)],
    #     "miyue": [{"hero": "miyue", "skill": "frenzy"} for _ in range(2)],
    #     "lvbu": [{"hero": "lvbu", "skill": "flash"} for _ in range(2)],
    #     "libai": [{"hero": "libai", "skill": "flash"} for _ in range(2)],
    #     "makeboluo": [{"hero": "makeboluo", "skill": "stun"} for _ in range(2)],
    #     "direnjie": [{"hero": "direnjie", "skill": "frenzy"} for _ in range(2)],
    #     "guanyu": [{"hero": "guanyu", "skill": "sprint"} for _ in range(2)],
    #     "diaochan": [{"hero": "diaochan", "skill": "purity"} for _ in range(2)],
    #     "luna": [{"hero": "luna", "skill": "intimidate"} for _ in range(2)],
    #     "hanxin": [{"hero": "hanxin", "skill": "flash"} for _ in range(2)],
    #     "huamulan": [{"hero": "huamulan", "skill": "flash"} for _ in range(2)],
    #     "buzhihuowu": [{"hero": "buzhihuowu", "skill": "execute"} for _ in range(2)],
    #     "jvyoujing": [{"hero": "jvyoujing", "skill": "flash"} for _ in range(2)],
    #     "houyi": [{"hero": "houyi", "skill": "frenzy"} for _ in range(2)],
    #     "zhongkui": [{"hero": "zhongkui", "skill": "stun"} for _ in range(2)],
    #     "ganjiangmoye": [{"hero": "ganjiangmoye", "skill": "flash"} for _ in range(2)],
    #     "kai": [{"hero": "kai", "skill": "intimidate"} for _ in range(2)],
    #     "gongsunli": [{"hero": "gongsunli", "skill": "frenzy"} for _ in range(2)],
    #     "peiqinhu": [{"hero": "peiqinhu", "skill": "flash"} for _ in range(2)],
    #     "shangguanwaner": [
    #         {"hero": "shangguanwaner", "skill": "heal"} for _ in range(2)
    #     ],
    # }
    HERO_DICT = {
        "luban": 112,
        "miyue": 121,
        "lvbu": 123,
        "libai": 131,
        "makeboluo": 132,
        "direnjie": 133,
        "guanyu": 140,
        "diaochan": 141,
        "luna": 146,
        "hanxin": 150,
        "huamulan": 154,
        "buzhihuowu": 157,
        "jvyoujing": 163,
        "houyi": 169,
        "zhongkui": 175,
        "ganjiangmoye": 182,
        "kai": 193,
        "gongsunli": 199,
        "peiqinhu": 502,
        "shangguanwaner": 513,
    }

    # def __init__(self, id, type):
    def __init__(self, id, agents, max_episode: int = 0, env=None, gpu_ip="127.0.0.1"):
        self.m_config_id = id
        self.m_task_uuid = Config.TASK_UUID
        self.m_episode_info = deque(maxlen=100)
        self.env = env
        self._max_episode = max_episode

        self.m_run_step = 0
        self.m_best_reward = 0

        self._last_print_time = time.time()
        self._episode_num = 0
        CommonLogger.set_config(self.m_config_id)
        self.agents = agents
        self.render = None

        self.sub_task = False
        self.sub_task_epsilon = 0.0
        self.sub_task_score = [604.157 * 3, 960.840 * 3]

    def set_env(self, environment):
        self.env = environment

    def set_agents(self, agents):
        self.agents = agents

    def _get_common_ai(self, eval, load_models):
        use_common_ai = [False] * len(self.agents)
        for i, agent in enumerate(self.agents):
            if eval:
                if load_models is None or len(load_models) < 2:
                    if not agent.keep_latest:
                        use_common_ai[i] = True
                elif load_models[i] is None:
                    use_common_ai[i] = True

        return use_common_ai

    def _reload_agents(self, eval=False, load_models=None):
        for i, agent in enumerate(self.agents):
            LOG.debug("reset agent {}".format(i))
            if load_models is None or len(load_models) < 2:
                agent.reset("common_ai")
            else:
                if load_models[i] is None:
                    agent.reset("common_ai")
                else:
                    agent.reset("network", model_path=load_models[i])

    def _run_episode(self, env_config, eval=False, load_models=None, eval_info=""):
        for item in g_log_time.items():
            g_log_time[item[0]] = []
        done = False
        log_time_func("reset")
        log_time_func("one_episode")
        LOG.debug("reset env")
        LOG.info(env_config)
        use_common_ai = self._get_common_ai(eval, load_models)

        # ATTENTION: agent.reset() loads models from local file which cost a lot of time.
        #            Before upload your code, please check your code to avoid ANY time-wasting
        #            operations between env.reset() and env.close_game(). Any TIMEOUT in a round
        #            of game will cause undefined errors.

        # reload agent models
        self._reload_agents(eval, load_models)
        render = self.render if eval else None
        # restart a new game
        # reward :[dead,ep_rate,exp,hp_point,kill,last_hit,money,tower_hp_point,reward_sum]
        print('-----------------before reseting env----------------------')
        print('env_config', env_config)
        print('use_common_ai', use_common_ai)
        print('eval', eval)
        print('render', render)
        # use_common_ai=[True,True]

        while True:
            _, r, d, state_dict = self.env.reset(env_config, use_common_ai=use_common_ai, eval=eval)
            if state_dict[0]['frame_no'] <= 10:
                break
            else:
                self.env.close_game()

        print('-----------------after reseting env1----------------------')
        if state_dict[0] is None:
            game_id = state_dict[1]["game_id"]
        else:
            game_id = state_dict[0]["game_id"]

        # update agents' game information
        for i, agent in enumerate(self.agents):
            player_id = self.env.player_list[i]
            camp = self.env.player_camp.get(player_id)
            agent.set_game_info(camp, player_id)

        # reset mem pool and models
        rewards = [[], []]
        step = 0
        log_time_func("reset", end=True)
        game_info = {}
        episode_infos = [{"h_act_num": 0} for _ in self.agents]

        while not done:
            log_time_func("one_frame")
            actions = []
            log_time_func("predict_process")
            for i, agent in enumerate(self.agents):
                if use_common_ai[i]:
                    actions.append(None)
                    rewards[i].append(0.0)
                    continue

                action, d_action, sample = agent.process(state_dict[i])
                if eval:
                    action = d_action

                actions.append(action)
                if action[0] == 10:
                    episode_infos[i]["h_act_num"] += 1
                rewards[i].append(sample["reward"])

            log_time_func("predict_process", end=True)

            log_time_func("step")

            if self.sub_task:
                for ii in range(len(actions)):
                    if not self.agents[ii].keep_latest:
                        actions[ii][0] = 1
                        continue
                    if self.agents[ii].save_h5_sample:  ### self.agents[ii].keep_latest == True
                        ### epsilon greedy ###
                        if random.uniform(0, 1) < self.sub_task_epsilon:
                            cur_legal_action = state_dict[ii]['legal_action']  ### 172, ###
                            label_split_size = [sum(ModelConfig.LABEL_SIZE_LIST[: index + 1]) for index in range(len(ModelConfig.LABEL_SIZE_LIST))]
                            legal_actions = np.split(cur_legal_action, label_split_size[:-1])
                            random_action = []
                            for index, every_legal in enumerate(legal_actions):
                                if index == len(legal_actions) - 1:
                                    which_button = random_action[0]
                                    every_legal = every_legal.reshape([12, 8])[which_button]
                                every_legal /= np.sum(every_legal)
                                random_action.append(np.argmax(np.random.multinomial(1, every_legal, size=1)))
                            actions[ii] = random_action

            _, r, d, state_dict = self.env.step(actions)

            log_time_func("step", end=True)

            req_pbs = self.env.cur_req_pb
            if req_pbs[0] is None:
                req_pb = req_pbs[1]
            else:
                req_pb = req_pbs[0]
            LOG.debug("step: {}, frame_no: {}, reward: {}, {}".format(step, req_pb.frame_no, r[0], r[1]))
            step += 1
            done = d[0] or d[1]

            ### FOR Sub Task ###
            if self.sub_task:
                for organ in req_pb.organ_list:
                    if str(organ.type) == 'ActorType.ACTOR_CRYSTAL' and organ.hp == 0:
                        done = True
                        break
            for i, agent in enumerate(self.agents):
                if agent.save_h5_sample:
                    agent._sample_process_for_saver_sp({'reward': state_dict[i]['reward']}, done=done)
            log_time_func("one_frame", end=True)

        self.env.close_game()

        game_info["length"] = req_pb.frame_no
        loss_camp = -1
        camp_hp = {}
        all_camp_list = []
        for organ in req_pb.organ_list:
            if organ.type == 24:
                if organ.hp <= 0:
                    loss_camp = organ.camp
                camp_hp[organ.camp] = organ.hp
                all_camp_list.append(organ.camp)
            if organ.type in [21, 24]:
                LOG.info("Tower {} in camp {}, hp: {}".format(organ.type, organ.camp, organ.hp))

        for i, agent in enumerate(self.agents):
            if use_common_ai[i]:
                continue
            for hero_state in req_pbs[i].hero_list:
                if agent.player_id == hero_state.runtime_id:
                    episode_infos[i]["money_per_frame"] = hero_state.moneyCnt / game_info["length"]
                    episode_infos[i]["kill"] = hero_state.killCnt
                    episode_infos[i]["death"] = hero_state.deadCnt
                    episode_infos[i]["hurt_per_frame"] = hero_state.totalHurt / game_info["length"]
                    episode_infos[i]["hurtH_per_frame"] = hero_state.totalHurtToHero / game_info["length"]
                    episode_infos[i]["hurtBH_per_frame"] = hero_state.totalBeHurtByHero / game_info["length"]
                    episode_infos[i]["hero_id"] = env_config['heroes'][i][0]['hero_id']
                    episode_infos[i]["totalHurtToHero"] = hero_state.totalHurtToHero
                    break
            if loss_camp == -1:
                episode_infos[i]["win"] = 0
            else:
                episode_infos[i]["win"] = -1 if agent.hero_camp == loss_camp else 1

            if agent.offline_agent:
                offline_win = max(episode_infos[i]['win'], 0)  # note: -1 is set to 0
                LOG.info(f"offline agent win:{episode_infos[i]['win']}")

            # print("rewards {} :".format(i),rewards[i])
            episode_infos[i]["reward"] = np.sum(rewards[i])
            episode_infos[i]["h_act_rate"] = episode_infos[i]["h_act_num"] / step

        log_time_func("one_episode", end=True)
        # print game information
        self._print_info(game_id, game_info, episode_infos, eval, eval_info, common_ai=use_common_ai, load_models=load_models)

        if self.sub_task:
            offline_win = 100 * (1 - (game_info["length"] - self.sub_task_score[0]) / (self.sub_task_score[1] - self.sub_task_score[0]))

        return offline_win

    def _print_info(self, game_id, game_info, episode_infos, eval, eval_info="", common_ai=None, load_models=None):
        if common_ai is None:
            common_ai = [False] * len(self.agents)
        if eval and len(eval_info) > 0:
            LOG.info("eval_info: %s" % eval_info)
        LOG.info("=" * 50)
        LOG.info("game_id : %s" % game_id)
        for item in g_log_time.items():
            if len(item) <= 1 or len(item[1]) == 0 or len(item[0]) == 0:
                continue
            mean = np.mean(item[1])
            max = np.max(item[1])
            sum = np.sum(item[1])
            LOG.info("%s | sum: %s mean:%s max:%s times:%s" % (item[0], sum, mean, max, len(item[1])))
            g_log_time[item[0]] = []
        LOG.info("=" * 50)
        for i, agent in enumerate(self.agents):
            if common_ai[i]:
                continue
            LOG.info(
                "Agent is_main:{}, model:{}, type:{}, camp:{},reward:{:.3f}, win:{}, win_{}:{},h_act_rate:{}".format(
                    agent.keep_latest and eval,
                    load_models[i],
                    agent.agent_type,
                    agent.hero_camp,
                    episode_infos[i]["reward"],
                    episode_infos[i]["win"],
                    episode_infos[i]["hero_id"],
                    episode_infos[i]["win"],
                    1.0,
                )
            )
            LOG.info(
                "Agent is_main:{}, money_per_frame:{:.2f}, kill:{}, death:{}, hurt_pf:{:.2f}".format(
                    agent.keep_latest and eval,
                    episode_infos[i]["money_per_frame"],
                    episode_infos[i]["kill"],
                    episode_infos[i]["death"],
                    episode_infos[i]["hurt_per_frame"],
                )
            )

        LOG.info("game info length:{}".format(game_info["length"]))

        LOG.info("=" * 50)

    def run(self, eval_mode=True, eval_number=-1, load_models=None, env_config_path=None):

        self._last_print_time = time.time()
        self._episode_num = 0

        if eval_mode:
            if load_models is None:
                raise "load_models is None! "
            LOG.info("eval_mode start...")
            agent_0, agent_1 = 0, 1
            cur_models = [load_models[agent_0], load_models[agent_1]]
            cur_eval_cnt = 1
            swap = False

        last_clean = time.time()

        # support multi heroes
        hero_data_list = []
        hero_config_0, hero_config_1 = env_config_path.split(",")
        with open(hero_config_0, 'r') as hero_file0:
            hero_data_list.append(json.load(hero_file0))
        with open(hero_config_1, 'r') as hero_file1:
            hero_data_list.append(json.load(hero_file1))

        if np.random.random() < 0.5:  ### change left && right side ###
            cur_models.reverse()
            self.agents.reverse()
            hero_data_list.reverse()

        offline_agent_win_list = []
        camp1_index = 0
        camp2_index = 0
        while True:
            # heroes' names
            heroes0 = list(hero_data_list[0].keys())  ### two sides' hero names ###
            heroes1 = list(hero_data_list[1].keys())
            # 英雄池大小
            hero_num0 = len(hero_data_list[0])
            hero_num1 = len(hero_data_list[1])
            hero_name1 = heroes0[camp1_index]  ### hero-from-camp1 vs hero-from-camp2 ###
            hero_name2 = heroes1[camp2_index]
            # config_dicts = [
            #     {"hero": hero_name1, "skill": hero_data_list[0][hero_name1]},
            #     {"hero": hero_name2, "skill": hero_data_list[1][hero_name2]},
            # ]
            from hok.hok1v1.camp import HERO_DICT
            first_id, second_id = HERO_DICT[hero_name1], HERO_DICT[hero_name2]
            config_dicts = {
                "mode": "1v1",
                "heroes": [
                    [{"hero_id": first_id, "skill_id": 80115, "symbol": [1512, 1512]}],
                    [{"hero_id": second_id, "skill_id": 80115, "symbol": [1512, 1512]}],
                ],
            }

            camp1_index += 1
            if camp1_index % hero_num0 == 0:  ### make all heros from two sides battle ###
                camp1_index = 0
                camp2_index = (camp2_index + 1) % hero_num1

            print('config_dicts:', config_dicts)
            try:
                # provide a init eval value at the first episode
                print("cur models", cur_models)
                if swap:
                    eval_info = "{} vs {}, {}/{}".format(agent_1, agent_0, cur_eval_cnt, eval_number)
                else:
                    eval_info = "{} vs {}, {}/{}".format(agent_0, agent_1, cur_eval_cnt, eval_number)
                offline_agent_win_list.append(self._run_episode(config_dicts, True, load_models=cur_models, eval_info=eval_info))
                # swap camp every-episode swap #
                cur_models.reverse()
                self.agents.reverse()
                hero_data_list.reverse()
                camp1_index, camp2_index = camp2_index, camp1_index

                swap = not swap

                self._episode_num += 1
            except Exception as e:  # pylint: disable=broad-except
                LOG.error(e)
                traceback.print_exc()

            if eval_mode:
                # update eval agents and eval cnt
                cur_eval_cnt += 1
                if cur_eval_cnt > eval_number:
                    cur_eval_cnt = 1

                    agent_1 += 1
                    if agent_1 >= len(load_models):
                        agent_0 += 1
                        agent_1 = agent_0 + 1

                    if agent_1 >= len(load_models):
                        # eval end
                        break

                    cur_models = [load_models[agent_0], load_models[agent_1]]
            else:
                # In training process, clean game_log every 1 hour.
                # feel free to DIY it by yourself.
                now = time.time()
                if self.m_config_id == 0 and now - last_clean > 3600:
                    LOG.info("Clean the game_log automatically.")
                    os.system('find /logs/cpu_log/game_log -mmin +60 -name "*" -exec rm -rfv {} \;')
                    last_clean = now

            if 0 < self._max_episode <= self._episode_num:
                break

        win_rate = np.mean(offline_agent_win_list)
        LOG.info("====================win rate====================")
        LOG.info(f"The winning rate of the offline_agent is {win_rate}")

        for agent in self.agents:
            agent.close()
            print('close ip!')

        return win_rate

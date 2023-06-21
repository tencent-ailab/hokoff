import numpy as np
import random

from eval_framework.predictor.utils import cvt_tensor_to_infer_input, cvt_tensor_to_infer_output

from hok.gamecore.kinghonour.frame_state_convert import convert_frame_state_pb, Object
from eval_framework.common_func import log_time

# from train_eval_config.hok_config import Config
import eval_framework.logging as LOG
import h5py
import os
import torch

_g_check_point_prefix = "checkpoints_"
_g_rand_max = 10000
_g_model_update_ratio = 0.8
# LOG = CommonLogger.get_logger()


def z_axis_symmetric(obj):
    if not isinstance(obj, Object):
        return

    for k, v in obj.__dict__.items():
        if isinstance(v, list):
            for i in v:
                z_axis_symmetric(i)
        else:
            z_axis_symmetric(v)

        if k == "x":
            obj.x = -obj.x


def cvt_infer_list_to_numpy_list(infer_list):
    data_list = [infer.data for infer in infer_list]
    return data_list


class RandomAgent:
    def process(self, feature, legal_action):
        action = [random.randint(0, 2) - 1, random.randint(0, 2) - 1]
        value = [0.0]
        neg_log_pi = [0]
        return action, value, neg_log_pi


class Agent:
    def __init__(self, model_cls, keep_latest=False, local_mode=False, rule_only=False, backend="pytorch", dataset=None):
        self.model = model_cls
        self.backend = backend
        self.rule_only = rule_only
        if self.backend == "tensorflow":
            from eval_framework.predictor.predictor.local_predictor import (
                LocalCkptPredictor as LocalPredictor,
            )

            self.graph = self.model.build_infer_graph()
            self._predictor = LocalPredictor(self.graph)
        else:
            from eval_framework.predictor.predictor.local_torch_predictor import (
                LocalTorchPredictor,
            )

            self._predictor = LocalTorchPredictor(self.model)

        self.model_version = ""
        self.is_latest_model: bool = False
        self.keep_latest = keep_latest
        self.model_list = []

        self.lstm_unit_size = self.model.lstm_unit_size

        self.lstm_hidden = None
        self.lstm_cell = None

        # self.agent_type = "common_ai"
        # self.player_id = 0
        self.last_model_path = None

        # self.agent_type = "network"
        self.agent_type = "network"
        if not any(dataset):
            self.save_h5_sample = False
            self.dataset_name = None
            self.dataset = None
            self.tmp_dataset = None
        else:
            self.save_h5_sample = True
            self.dataset_name = dataset
            self.dataset = h5py.File(dataset, "a")
            self.tmp_dataset_name = self.dataset_name[: dataset.rfind('/') + 1] + 'tmp_' + self.dataset_name[dataset.rfind('/') + 1 :]
            if os.path.exists(self.tmp_dataset_name):
                os.remove(self.tmp_dataset_name)
            self.tmp_dataset = h5py.File(dataset, "a")

    # def reset(self, hero_camp, player_id, agent_type=None, model_path=None):
    # def reset(self, agent_type=None, model_path=None):
    def reset(self, agent_type=None, model_path=None):
        # self.hero_camp = hero_camp
        # self.player_id = player_id
        # reset lstm input
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])

        if agent_type is not None:
            if self.keep_latest:
                self.agent_type = "network"
            else:
                self.agent_type = agent_type

        # for test without model pool

        if model_path is None:
            while True:
                try:
                    if self.keep_latest:
                        self._get_latest_model()
                    else:
                        self._get_random_model()
                    return
                except Exception as e:
                    LOG.error(e)
                    LOG.error("get_model error, try again...")
        elif self.rule_only:
            self._get_random_model()
        else:
            ret = self._predictor.load_model(model_path)
        if self.dataset is None:
            self.save_h5_sample = False
        else:
            self.save_h5_sample = True
            if len(self.tmp_dataset.keys()) == 0:
                self.dataset.close()
                self.dataset = h5py.File(self.dataset_name, "a")
                self.tmp_dataset.close()
                self.tmp_dataset = h5py.File(self.tmp_dataset_name, 'a')
            else:
                for key in self.tmp_dataset.keys():
                    if key not in self.dataset.keys():
                        self.dataset.create_dataset(
                            key,
                            data=np.array(self.tmp_dataset[key]),
                            compression="gzip",
                            maxshape=(None, *(list(self.tmp_dataset[key].shape)[1:])),
                            chunks=True,
                        )
                    else:
                        self.dataset[key].resize((self.dataset[key].shape[0] + self.tmp_dataset[key].shape[0]), axis=0)
                        self.dataset[key][-self.tmp_dataset[key].shape[0] :] = np.array(self.tmp_dataset[key])
                self.dataset.close()
                self.dataset = h5py.File(self.dataset_name, "a")
                self.tmp_dataset.close()
                if os.path.exists(self.tmp_dataset_name):
                    os.remove(self.tmp_dataset_name)
                self.tmp_dataset = h5py.File(self.tmp_dataset_name, 'a')

    def is_common_ai(self):
        if self.agent_type == "common_ai":
            return True
        else:
            return False

    @log_time("aiprocess_process")
    def predict_process(self, state_dict, req_pb):
        hero_data_list = []
        runtime_ids = []
        for hero_idx in range(len(state_dict)):
            feature = state_dict[hero_idx]["feature"]
            hero_data_list.append(feature)
            runtime_ids.append(state_dict[hero_idx]["hero_runtime_id"])

        frame_state = convert_frame_state_pb(req_pb.frame_state)
        if self.backend == "tensorflow":
            pred_ret, lstm_info = self._predict_process(hero_data_list, frame_state, runtime_ids)
        else:
            pred_ret, lstm_info = self._predict_process_torch(hero_data_list, frame_state, runtime_ids)

        return pred_ret, lstm_info

    def _predict_process(self, hero_data_list, frame_state, runtime_ids):
        # TODO: add a switch for controlling sample strategy.
        # put data to input
        input_list = cvt_tensor_to_infer_input(self.model.get_input_tensors())
        input_list[0].set_data(np.array(hero_data_list[0]))
        input_list[1].set_data(np.array(hero_data_list[1]))
        input_list[2].set_data(np.array(hero_data_list[2]))
        input_list[3].set_data(self.lstm_cell)
        input_list[4].set_data(self.lstm_hidden)

        output_list = cvt_tensor_to_infer_output(self.model.get_output_tensors())
        output_list = self._predictor.inference(input_list=input_list, output_list=output_list)
        # cvt output data
        np_output = cvt_infer_list_to_numpy_list(output_list)

        prob_h0, prob_h1, prob_h2, self.lstm_cell, self.lstm_hidden = np_output[:5]
        prob = []
        prob.append(prob_h0)
        prob.append(prob_h1)
        prob.append(prob_h2)
        lstm_info = (self.lstm_cell, self.lstm_hidden)

        return prob, lstm_info

    def _predict_process_torch(self, hero_data_list, frame_state, runtime_ids):
        # TODO: add a switch for controlling sample strategy.
        # put data to input
        input_list = []
        input_list.append(np.array(hero_data_list[0]))
        input_list.append(np.array(hero_data_list[1]))
        input_list.append(np.array(hero_data_list[2]))
        input_list.append(self.lstm_cell)
        input_list.append(self.lstm_hidden)

        if (not self.keep_latest) or self.save_h5_sample:
            output_list = self._predictor.inference(input_list)
        else:
            torch_inputs = [torch.from_numpy(nparr).to(torch.float32) for nparr in input_list]
            self.model.eval()
            with torch.no_grad():
                output_list = self.model(torch_inputs, only_inference=True)
        np_output_list = []
        for output in output_list:
            np_output_list.append(output.numpy())

        prob_h0, prob_h1, prob_h2, self.lstm_cell, self.lstm_hidden = np_output_list[:5]
        prob = []
        prob.append(prob_h0)
        prob.append(prob_h1)
        prob.append(prob_h2)
        lstm_info = (self.lstm_cell, self.lstm_hidden)

        return prob, lstm_info

    def _get_h5file_keys(self, h5file):
        keys = []

        def visitor(name, item):
            if isinstance(item, h5py.Dataset):
                keys.append(name)

        h5file.visititems(visitor)
        return keys

    def _sample_process_for_saver_sp(self, sample_dict):
        keys_in_h5 = self._get_h5file_keys(self.tmp_dataset)
        # if not 'observation_s' in keys_in_h5:
        #     self.tmp_dataset.create_dataset(
        #         "observation_s",
        #         data=[np.stack(sample_dict["vc_feature_s"])],
        #         compression="gzip",
        #         maxshape=(None, 3,len(sample_dict["vc_feature_s"][0])),
        #         chunks=True,
        #     )
        # else:
        # self.tmp_dataset['observation_s'].resize((self.tmp_dataset['observation_s'].shape[0] + 1), axis=0)
        # self.tmp_dataset['observation_s'][-1] = np.stack(sample_dict["vc_feature_s"])
        if 'reward_s' in sample_dict.keys():
            if not 'reward_s' in keys_in_h5:
                self.tmp_dataset.create_dataset(
                    "reward_s",
                    data=[np.expand_dims(np.array(sample_dict["reward_s"]), -1)],
                    compression="gzip",
                    maxshape=(None, 3, 1),
                    chunks=True,
                )
            else:
                self.tmp_dataset['reward_s'].resize((self.tmp_dataset['reward_s'].shape[0] + 1), axis=0)
                self.tmp_dataset['reward_s'][-1] = np.expand_dims(np.array(sample_dict["reward_s"]), -1)
        if 'done' in sample_dict.keys():
            if not 'done' in keys_in_h5:
                self.tmp_dataset.create_dataset(
                    "done",
                    data=[[sample_dict['done']]],
                    compression="gzip",
                    maxshape=(None, 1),
                    chunks=True,
                )
            else:
                self.tmp_dataset['done'].resize((self.tmp_dataset['done'].shape[0] + 1), axis=0)
                self.tmp_dataset['done'][-1] = [sample_dict['done']]

    def _sample_process_for_saver(self, sample_dict):
        keys = ("frame_no", "vec_feature_s", "legal_action_s", "action_s", "sub_action_s")
        keys_in_h5 = self._get_h5file_keys(self.tmp_dataset)
        if 'frame_no' not in keys_in_h5:
            self.tmp_dataset.create_dataset(
                "frame_no",
                data=[[sample_dict["frame_no"]]],
                compression="gzip",
                maxshape=(None, 1),
                chunks=True,
            )
            self.tmp_dataset.create_dataset(
                "observation_s",
                data=[np.stack(sample_dict["vec_feature_s"]).astype(np.float32)],
                compression="gzip",
                maxshape=(None, 3, len(sample_dict["vec_feature_s"][0])),
                chunks=True,
            )
            self.tmp_dataset.create_dataset(
                "legal_action_s",
                data=[np.stack(sample_dict["legal_action_s"])],
                compression="gzip",
                maxshape=(None, 3, len(sample_dict["legal_action_s"][0])),
                chunks=True,
            )
            self.tmp_dataset.create_dataset(
                "action_s",
                data=[np.stack(sample_dict["action_s"])],
                compression="gzip",
                maxshape=(None, 3, len(sample_dict["action_s"][0])),
                chunks=True,
            )
            # self.tmp_dataset.create_dataset(
            #     "reward_s",
            #     data=[np.expand_dims(np.array(sample_dict["reward_s"]),-1)],
            #     compression="gzip",
            #     maxshape=(None, 3,1),
            #     chunks=True,
            # )
            self.tmp_dataset.create_dataset(
                "sub_action_s",
                data=[np.stack(sample_dict["sub_action_s"])],
                compression="gzip",
                maxshape=(None, 3, len(sample_dict["sub_action_s"][0])),
                chunks=True,
            )

        else:
            # reward_init = 0
            # if 'reward_s' not in keys_in_h5:
            #     self.tmp_dataset.create_dataset(
            #         "reward_s",
            #         data=[np.expand_dims(np.array(sample_dict["reward_s"]),-1)],
            #         compression="gzip",
            #         maxshape=(None, 3,1),
            #         chunks=True,
            #     )
            #     reward_init=1

            for key, value in sample_dict.items():
                if key in keys:
                    key_dataset = key
                    if key_dataset == "vec_feature_s":
                        key_dataset = "observation_s"

                    self.tmp_dataset[key_dataset].resize((self.tmp_dataset[key_dataset].shape[0] + 1), axis=0)
                    if isinstance(value, list):
                        # if key_dataset=='reward_s' and not reward_init:
                        #     self.tmp_dataset[key_dataset][-1]=np.expand_dims(np.array(value),-1)
                        # else:
                        self.tmp_dataset[key_dataset][-1] = np.stack(value, 0)
                    else:
                        self.tmp_dataset[key_dataset][-1] = [value]

    def close(self):
        if self.dataset is not None:
            if len(self.tmp_dataset.keys()) > 0:
                for key in self.tmp_dataset.keys():
                    if key not in self.dataset.keys():
                        self.dataset.create_dataset(
                            key,
                            data=np.array(self.tmp_dataset[key]),
                            compression="gzip",
                            maxshape=(None, *(list(self.tmp_dataset[key].shape)[1:])),
                            chunks=True,
                        )
                    else:
                        self.dataset[key].resize((self.dataset[key].shape[0] + self.tmp_dataset[key].shape[0]), axis=0)
                        self.dataset[key][-self.tmp_dataset[key].shape[0] :] = np.array(self.tmp_dataset[key])
                # self.dataset.close()
                # self.tmp_dataset.close()
                # os.remove(self.tmp_dataset_name)
            self.save_h5_sample = True
            self.dataset.close()
            self.tmp_dataset.close()
            os.remove(self.tmp_dataset_name)

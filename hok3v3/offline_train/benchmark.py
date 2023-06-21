# -*- coding: utf-8 -*-
import time

# try:
#     import horovod.torch as hvd
#     has_hvd = True
# except:
has_hvd = False
import numpy as np
import torch
import os
import threading
from datasets import Datasets
from torch.utils.tensorboard import SummaryWriter
import sys
import torch as th

class Benchmark(object):
    def __init__(self, args, network, config_manager, LogManagerClass):
        self.args = args

        self.log_manager = LogManagerClass(backend="pytorch")
        self.log_manager.print_info("init starting, backend=pytorch")
        self.config_manager = config_manager
        # self.model_manager = ModelManagerClass(self.config_manager.push_to_modelpool)

        self.rank = 0
        self.rank_size = 1
        self.local_rank = 0
        self.local_size = 1
        # self.rank = hvd.rank() if has_hvd else 0
        # self.rank_size = hvd.size() if has_hvd else 1
        # self.local_rank = hvd.local_rank() if has_hvd else 0
        # self.local_size = hvd.local_size() if has_hvd else 1
        self.is_chief_rank = self.rank == 0

        device_idx = self.local_rank
        if torch.cuda.is_available():
            self.log_manager.print_info("Cuda is available and being used")
            self.device = torch.device("cuda", device_idx)
            torch.cuda.set_device(device_idx)
        else:
            self.log_manager.print_info("Cuda not available. Using CPU instead")
            self.device = torch.device("cpu", device_idx)

        self.net = network
        self.net.to(self.device)
        self.dataset = Datasets(
            os.path.join(args.replay_dir, args.dataset_name),
            self.args.batch_size,
            self.net.lstm_time_steps,
            device=self.device,
            train_step_per_buffer=args.train_step_per_buffer,
            num_workers=args.buffer_num_workers,
        )

        self.local_step = 0
        self.warmup_steps = 0
        self.step_train_times = list()
        self.skip_update_times = 0
        self.tb_writer = SummaryWriter(log_dir=os.path.join(args.root_path, args.run_prefix, 'train'))

        self._init_model()
        self.log_manager.print_info("init finished")
        if self.args.eval_num > 0:  # better don't use, will disturb the efficiency
            # t1 = threading.Thread(target=evaluate, args=[self.args.root_path,self.args.run_prefix,self.args.levels,self.args.eval_num, self.args.cpu_num])
            # t1.setDaemon(True)
            # t1.start()
            evaluate_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            os.system(
                'nohup python {}/evaluation.py --root_path={} --run_prefix={} --levels={} --cpu_num={} --eval_num={} --dataset_name={}'.format(
                    evaluate_dir_path, args.root_path, args.run_prefix, args.levels, args.cpu_num, args.eval_num, args.dataset_name
                )
            )

    def _init_model(self):

        # only load checkpoint on master node and then broadcast
        max_model_step = 0
        if os.path.exists(self.config_manager.save_model_dir):
            for file_name in os.listdir(self.config_manager.save_model_dir):
                if '_model' in file_name:
                    model_step = int(file_name.split('_')[0])
                    max_model_step = max(model_step, max_model_step)
        if max_model_step > 0:
            model_checkpoint_path = os.path.join(self.config_manager.save_model_dir, str(max_model_step) + '_model', "model.pth")
            self.log_manager.print_info(f"Loading checkpoint from {model_checkpoint_path}")
            self.load_checkpoint(model_checkpoint_path)
            self.local_step = max_model_step
            self.warmup_steps = max_model_step

        if max_model_step == 0:
            self.log_manager.print_info(f"Saving checkpoint_0 to {self.config_manager.save_model_dir}")
            os.makedirs(self.config_manager.save_model_dir, exist_ok=True)
            self.save_checkpoint(self.config_manager.save_model_dir)
        # self.model_manager.send_model(self.config_manager.save_model_dir, self.config_manager.send_model_dir)

    def _do_train(self):
        self.log_manager.print_info('Start training...')
        self.net.train()
        start_time = time.time()
        local_start_time = time.time()
        waste_time = 0
        train_time = 0
        for _ in range(self.warmup_steps, self.args.max_steps):
            th.cuda.synchronize()
            batch_begin = time.time()
            # self.optimizer.zero_grad()
            results = {}
            batch_read_start_time = time.time()
            input_datas = self.dataset.next_batch()
            waste_time += time.time() - batch_read_start_time
            th.cuda.synchronize()
            before_train_start_time = time.time()
            total_loss, info_dict = self.net.step(input_datas)

            results["total_loss"] = total_loss.item()
            results["info_list"] = []

            th.cuda.synchronize()
            train_time += time.time() - before_train_start_time

            batch_duration = time.time() - batch_begin
            self.local_step += 1
            if self.local_step % self.config_manager.save_model_steps != 0:
                self.step_train_times.append(batch_duration)

            if self.is_chief_rank and (self.local_step == 0 or self.local_step % self.config_manager.display_every == 0):
                results['ip'] = self.config_manager.ips[0]
                results['batch_size'] = self.config_manager.batch_size
                results['step'] = self.local_step
                results['gpu_nums'] = self.rank_size
                results['sample_recv_speed'] = 0
                results['sample_consume_speed'] = self.get_sample_consume_speed(self.config_manager.batch_size, self.step_train_times)
                self.log_manager.print_result(results)

                ###############tensorboard
                for key in info_dict:
                    self.tb_writer.add_scalar('train/' + key, info_dict[key], global_step=self.local_step)
                self.tb_writer.add_scalar('train/loss', total_loss, self.local_step)
                self.tb_writer.add_scalar('train/total_average_train_step_per_s', self.local_step / (time.time() - start_time), self.local_step)
                self.tb_writer.add_scalar('config/batch_size', self.args.batch_size, self.local_step)
                self.tb_writer.add_scalar('config/cql_alpha', self.args.cql_alpha, self.local_step)
                self.tb_writer.add_scalar('config/oppo_levels', int(self.args.levels), self.local_step)
                self.tb_writer.add_scalar('config/lstm_time_steps', self.args.lstm_time_steps, self.local_step)
                self.tb_writer.add_scalar('config/lr', self.args.lr, self.local_step)
                self.tb_writer.add_scalar('config/target_update_frequence', self.args.target_update_freq, self.local_step)
                self.tb_writer.add_scalar('config/gamma', self.args.gamma, self.local_step)
                self.tb_writer.add_scalar('config/train_step_per_buffer', self.args.train_step_per_buffer, self.local_step)
                ##########################

            if self.local_step % self.config_manager.save_model_steps == 0 and self.is_chief_rank:
                self.save_checkpoint(self.config_manager.save_model_dir)
            if self.local_step % self.args.target_update_freq == 0 and self.is_chief_rank:
                self.net.update_target_net()
                # self.net.soft_update()

            if self.local_step % self.config_manager.display_every == 0:
                th.cuda.synchronize()
                print(
                    'Run_Prefix: {}, Levels: {}, Training steps: {}, Average training steps per second: {}, Total time: {} hours, Local training time:{} min, ratio:{}, Local waste time:{} min, ratio:{}'.format(
                        self.args.run_prefix,
                        self.args.levels,
                        self.local_step,
                        self.config_manager.display_every / (time.time() - local_start_time),
                        (time.time() - start_time) / 3600,
                        train_time / 60,
                        train_time / (time.time() - local_start_time),
                        waste_time / 60,
                        waste_time / (time.time() - local_start_time),
                    )
                )
                local_start_time = time.time()
                waste_time = 0
                train_time = 0
            if self.local_step > self.args.max_steps:
                break

        images_per_sec = (time.time() - start_time) / (self.args.max_steps - self.warmup_steps) * self.args.batch_size
        self.log_manager.print_info('-' * 64)
        self.log_manager.print_info('total images/sec: %.2f' % images_per_sec)
        self.log_manager.print_info('-' * 64)
        # Save the model checkpoint.
        # if self.is_chief_rank:
        #     self.save_checkpoint(self.config_manager.save_model_dir)

    def run(self):
        self._do_train()

    def save_checkpoint(self, checkpoint_dir: str):
        for file_name in os.listdir(checkpoint_dir):
            if '_model' in file_name:
                model_step = int(file_name.split('_')[0])
                if abs(model_step - self.local_step) > 10000:
                    os.remove(os.path.join(checkpoint_dir, file_name, 'model.pth'))
                    os.rmdir(os.path.join(checkpoint_dir, file_name))
        os.makedirs(os.path.join(checkpoint_dir, str(self.local_step) + "_model"), exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, str(self.local_step) + "_model", "model.pth")
        if not self.is_chief_rank:
            return  # only save checkpoint on master node
        save_dict = self.net.save_dict()
        torch.save(save_dict, checkpoint_file)

    def load_checkpoint(self, checkpoint_file: str):
        if not self.is_chief_rank:
            return  # only load checkpoint on master node and then broadcast
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
        else:
            checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))

        self.net.load_save_dict(checkpoint)

    def get_sample_consume_speed(self, batch_size, step_train_times, scale=1):
        if not step_train_times:
            return ''
        if len(step_train_times) <= 1:
            return step_train_times[0]
        times = np.array(step_train_times[1:])
        speed_mean = scale * batch_size / np.mean(times)
        return speed_mean

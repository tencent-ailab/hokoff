import json
import logging
import logging.handlers
import os
import time

LOG_FN = logging.getLogger('main').debug
from train_eval_framework import *


class LogManagerBase(object):
    def __init__(self, backend='tensorflow'):
        self.backend = backend
        # create logger
        logger = logging.getLogger('main')
        logger.setLevel(logging.DEBUG)

        # set formatter
        formatter = logging.Formatter(fmt='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # set file handler: file rotates with time
        log_dir = './log'
        train_log_path = os.path.join(log_dir, "train.log")
        loss_log_path = os.path.join(log_dir, "loss.txt")

        os.makedirs(log_dir, exist_ok=True)

        rf_handler = logging.handlers.TimedRotatingFileHandler(filename=train_log_path, when='H', interval=12, backupCount=1)
        rf_handler.setLevel(logging.DEBUG)
        rf_handler.setFormatter(formatter)

        # set console handler: display on the console
        console = logging.StreamHandler()
        console.setLevel(logging.ERROR)
        console.setFormatter(formatter)

        logger.addHandler(rf_handler)
        logger.addHandler(console)
        self.loss_writer = open(loss_log_path, 'wt')
        self.total_noise_scale = 0.0

    def print_result(self, results):
        local_step = results['step']
        batch_size = results['batch_size']
        gpu_nums = results['gpu_nums']
        recv_speed = results['sample_recv_speed']
        consume_speed = results['sample_consume_speed']

        loss = results["total_loss"]
        log_str = ""
        log_str += "step: %i" % local_step
        log_str += " images/sec mean = %.1f" % consume_speed
        if recv_speed is not None and recv_speed > 0:
            log_str += " recv_sample/sec = %i" % recv_speed
        log_str += " total_loss: %s" % loss
        LOG_FN(log_str)

        if results['info_list']:
            self._write_loss_log(results)

    def _write_loss_log(self, results):
        local_step = results['step']
        hostname = results['ip']
        timestamp = time.strftime("%m/%d/%Y-%H:%M:%S", time.localtime())
        loss_str = " "
        for info in results['info_list']:
            _info = info
            loss_str += '%s ' % str(_info)
        loss_log = {"role": "learner", "ip_address": str(hostname), "step": str(local_step), "loss": loss_str, "timestamp": timestamp}
        for key, val in sorted(loss_log.items()):
            if hasattr(val, 'dtype'):
                loss_log[key] = float(val)
        self.loss_writer.write(json.dumps(loss_log) + '\n')
        self.loss_writer.flush()

    def print_info(self, info):
        LOG_FN(info)


@singleton
class LogManager(LogManagerBase):
    pass

import logging
import math

from train_eval_framework.log_manager_base import LogManagerBase


class LogManager(LogManagerBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # monitor logger
        # self.monitor_logger = logging.getLogger("monitor")
        # self.monitor_logger.setLevel(logging.INFO)
        # monitor_handler = InfluxdbMonitorHandler("127.0.0.1")
        # monitor_handler.setLevel(logging.INFO)
        # self.monitor_logger.addHandler(monitor_handler)

    def _add_float(self, data, key, val):
        try:
            val = float(val)
            if not math.isnan(val) and not math.isinf(val):
                data[key] = val
        except Exception as e:
            pass

    def upload_monitor_data(self, data: dict):
        print(data)
        # self.monitor_logger.info(data)

    def print_result(self, results):
        super().print_result(results)
        local_step = results["step"]
        batch_size = results["batch_size"]
        gpu_nums = results["gpu_nums"]
        loss = results["total_loss"]

        recv_speed = results["sample_recv_speed"]
        consume_speed = results["sample_consume_speed"]

        monitor_data = {}
        monitor_data["step"] = int(local_step)
        self._add_float(monitor_data, "sample_consumption_rate", consume_speed * 60 * gpu_nums)
        self._add_float(monitor_data, "total_loss", loss)
        for idx, info in enumerate(results["info_list"]):
            if type(info) == list:
                for idx2, _info in enumerate(info):
                    self._add_float(monitor_data, f"loss_{idx}_{idx2}", _info)

            else:
                self._add_float(monitor_data, f"loss_{idx}", info)
        if recv_speed is not None and recv_speed > 0:
            monitor_data["sample_generation_rate"] = float(recv_speed * 60 * gpu_nums)
        self.upload_monitor_data(monitor_data)

import logging
from datetime import datetime
import functools
from functools import partial
import os

LOGGER_NAME = 'TSAnomalyDetectionLogger'


def get_logs_path():
    path = os.getcwd()
    logs_path = path + '\logs'
    return logs_path


def path_exists(path):
    return os.path.isdir(path)


def create_logger():
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y%H%M%S")

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)

    logs_path = get_logs_path()
    if not path_exists(logs_path):
        os.mkdir(logs_path)

    f_handler = logging.FileHandler(logs_path + '\log_{}.log'.format(dt_string))
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)

    logger.addHandler(f_handler)


def get_logger():
    return logging.getLogger(LOGGER_NAME)


class MethodLogger:
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.logger = logging.getLogger(LOGGER_NAME)

    def __call__(self, instance, *args, **kwargs):
        try:
            class_name = instance.__class__.__name__
            self.logger.info("Running function {0} in - {1} with arguments - {2} - {3}"
                             .format(self.func.__name__, class_name, args, kwargs))
            result = self.func(instance, *args, **kwargs)
            self.logger.info(result)
            return result
        except Exception as e:
            self.logger.exception("Exception occurred: {}".format(e))

    def __get__(self, instance, owner):
        return partial(self, instance)

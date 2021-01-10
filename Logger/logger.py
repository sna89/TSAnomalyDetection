import logging
from datetime import datetime
import functools
from functools import partial
import os
from Helpers.file_helper import FileHelper


def get_logger(name):
    existing_loggers = [log_name for log_name in logging.Logger.manager.loggerDict.keys()]
    logger = logging.getLogger(name)

    if name not in existing_loggers:
        add_handlers(logger)
        logger.setLevel(logging.DEBUG)

    return logger


def add_handlers(logger):
    add_file_handler(logger)
    add_stream_handler(logger)


def get_logs_path():
    logs_path = FileHelper.get_logs_path()
    FileHelper.create_directory(logs_path)
    return logs_path


def add_file_handler(logger):
    now = datetime.now()

    dt_string = now.strftime("%d%m%Y%H")
    logs_path = get_logs_path()

    log_file_name = os.path.join(logs_path, 'log_{}.log'.format(dt_string))
    f_handler = logging.FileHandler(log_file_name)
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    f_handler.setLevel(logging.DEBUG)
    logger.addHandler(f_handler)


def add_stream_handler(logger):
    c_handler = logging.StreamHandler()
    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    c_handler.setLevel(logging.DEBUG)
    logger.addHandler(c_handler)


class MethodLogger:
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.logger = logging.getLogger(func.__name__)

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

import logging
from datetime import datetime
import functools
from functools import partial

LOGGER_NAME = 'TSAnomalyDetectionLogger'


def create_logger():
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y%H%M%S")

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)

    f_handler = logging.FileHandler('log_{}.log'.format(dt_string))
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)

    logger.addHandler(f_handler)


class LoggerDecorator:
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.logger = logging.getLogger(LOGGER_NAME)

    def __call__(self, instance, *args, **kwargs):
        try:
            class_name = instance.__class__.__name__
            self.logger.info("Running function {0} in - {1} - {2} - {3}"
                             .format(self.func.__name__, class_name, args, kwargs))
            return self.func(instance, *args, **kwargs)
        except Exception as e:
            self.logger.exception("Exception occurred: {}".format(e))

    def __get__(self, instance, owner):
        return partial(self, instance)

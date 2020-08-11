import os
from os import path

G_COLAB_PATH = '/content/TSAnomalyDetection/'

class FileHelper:
    def __init__(self):
        pass

    @staticmethod
    def get_file_path(params_filename):
        if path.exists(params_filename):
            return params_filename
        else:
            fullpath = G_COLAB_PATH + params_filename
            if path.exists(fullpath):
                return fullpath
            else:
                raise ValueError("Cannot file file: {}".format(params_filename))


    @staticmethod
    def get_logs_path():
        path = os.getcwd()
        logs_path = path + '\logs'
        return logs_path

    @staticmethod
    def path_exists(path):
        return os.path.isdir(path)



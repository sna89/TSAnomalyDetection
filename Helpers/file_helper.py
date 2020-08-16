import os
from pathlib import Path


class FileHelper:
    def __init__(self):
        pass

    @staticmethod
    def get_file_path(filename):
        curr_path = Path(os.getcwd())

        for path in Path(curr_path).rglob('*.*'):
            if path.name == filename:
                return path

        msg = "Can't find filename: {}".format(filename)
        raise ValueError(msg)

    @staticmethod
    def get_logs_path():
        path = os.getcwd()
        logs_path = path + '\logs'
        return logs_path

    @staticmethod
    def path_exists(path):
        return os.path.isdir(path)



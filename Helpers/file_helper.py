import os
from pathlib import Path


class FileHelper:
    def __init__(self):
        pass

    @staticmethod
    def get_file_path(filename, file_regex='*.*', directory=None):
        curr_path = Path(os.getcwd())
        if directory:
            curr_path = os.path.join(curr_path, directory)
        for path in Path(curr_path).rglob(file_regex):
            if path.name == filename:
                return path

    @staticmethod
    def get_logs_path():
        dir_path = os.getcwd()
        return os.path.join(dir_path, 'logs')

    @staticmethod
    def path_exists(path):
        return os.path.isdir(path)

    @staticmethod
    def create_directory(path):
        if not FileHelper.path_exists(path):
            os.mkdir(path)

    @staticmethod
    def delete_file(file_path):
        if os.path.exists(file_path):
            os.remove(file_path)
import os
from pathlib import Path

G_COLAB_PATH = '/content/TSAnomalyDetection/'

class FileHelper:
    def __init__(self):
        pass

    @staticmethod
    def get_file_path(filename):
        curr_path = Path(os.getcwd())
        parent_path = FileHelper.get_parent_directory(curr_path)

        for path in Path(parent_path).rglob('*.*'):
            if path.name == filename:
                return path

        return None

    @staticmethod
    def get_parent_directory(cwd: Path):
        return cwd.parent

    @staticmethod
    def get_logs_path():
        path = os.getcwd()
        logs_path = path + '\logs'
        return logs_path

    @staticmethod
    def path_exists(path):
        return os.path.isdir(path)



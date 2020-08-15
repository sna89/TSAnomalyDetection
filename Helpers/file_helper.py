import os
from pathlib import Path
from Logger.logger import get_logger


class FileHelper:
    logger = get_logger()

    @classmethod
    def get_file_path(cls, filename):
        curr_path = Path(os.getcwd())
        parent_path = FileHelper.get_parent_directory(curr_path)

        for path in Path(parent_path).rglob('*.*'):
            if path.name == filename:
                return path

        msg = "Can't find filename: {}".format(filename)
        cls.logger.error(msg)
        raise ValueError(msg)

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



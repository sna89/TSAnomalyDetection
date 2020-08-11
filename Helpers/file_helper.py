import os


class FileHelper:
    def __init__(self):
        pass

    @staticmethod
    def get_file_path(filename, cwd=None):
        path = None

        if not cwd:
            cwd = os.getcwd()

        for current_filename in os.listdir(cwd):
            if current_filename == filename:
                path = cwd + '\\' + filename

            if os.path.isdir(current_filename):
                wd = current_filename
                sub_wd = cwd + '\\' + wd
                path = FileHelper.get_file_path(filename, sub_wd)

            if path:
                return path

        return path

    @staticmethod
    def get_logs_path():
        path = os.getcwd()
        logs_path = path + '\logs'
        return logs_path

    @staticmethod
    def path_exists(path):
        return os.path.isdir(path)



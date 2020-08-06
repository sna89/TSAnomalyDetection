import yaml


class ParamsHelper:
    def __init__(self, filename):
        with open(filename) as file:
            self.params_dict = yaml.load(file, Loader=yaml.FullLoader)

    def get_params(self, param_name):
        try:
            return self.params_dict[param_name]
        except Exception as e:
            raise ValueError("cannot find parameter {}".format(e))

    def get_experiment_hyperparams(self):
        return self.get_params('experiment_hyperparameters')

    def get_model_hyperparams(self):
        return self.get_params('model_hyperparameters')

    def get_metadata(self):
        return self.get_params('metadata')

    def get_detector_name(self):
        return self.get_params('detector_name')

    def get_test(self):
        return self.get_params('test')

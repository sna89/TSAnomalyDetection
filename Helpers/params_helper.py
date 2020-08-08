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
        detector = self.get_detector_name()
        model_hyperparameters_dict = self.get_params('model_hyperparameters')
        model_hyperparameters = model_hyperparameters_dict[detector]
        return model_hyperparameters

    def get_metadata(self):
        return self.get_params('metadata')

    def get_detector_name(self):
        return self.get_params('detector_name')

    def get_preprocess_data_params(self):
        return self.get_params('preprocess_data_params')

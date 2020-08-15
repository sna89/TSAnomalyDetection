import unittest
from Helpers.params_helper import ParamsHelper


class TestParams(unittest.TestCase):
    def setUp(self):
        self.params_helper = ParamsHelper('params_test.yml')

    def test_model_hyperparamters(self):
        model_hyperparameters = self.params_helper.get_model_hyperparams()

        key = next(iter(model_hyperparameters))
        assert key == "test_hyperparameter_key", "Can't read model hyperparameter key correctly"

        value = model_hyperparameters[key]
        assert value == "test_hyperparameter_value", "Can't read model hyperparameter value correctly"

    def test_experiment_name(self):
        experiment_name = self.params_helper.create_experiment_name()
        assert 'test_test_hyperparameter_key=test_hyperparameter_value_' in experiment_name, \
            "Error creating experiment name"

if __name__ == '__main__':
    unittest.main()
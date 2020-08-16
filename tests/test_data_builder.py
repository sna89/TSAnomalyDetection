import unittest
from Helpers.file_helper import FileHelper
from Helpers.params_helper import ParamsHelper
from Helpers.params_validator import ParamsValidator
from Builders.data_builder import DataConstructor
import pandas as pd
from pandas.util.testing import assert_frame_equal


class TestDataBuilder(unittest.TestCase):
    def test_schema_type_1(self):
        # Sensor U106748.csv

        params_helper = ParamsHelper('params_test_type_1.yml')
        ParamsValidator(params_helper).validate()

        filename = 'expected_schema_type_1.csv'
        csv_file_path = FileHelper.get_file_path(filename)
        expected_data = pd.read_csv(csv_file_path, index_col="sampletime", parse_dates=True)

        metadata = params_helper.get_metadata()
        preprocess_data_params = params_helper.get_preprocess_data_params()

        data = DataConstructor(metadata, preprocess_data_params).read().build()
        assert_frame_equal(data, expected_data)

    def test_schema_type_2(self):
        # 54863.csv with fill missing time

        params_helper = ParamsHelper('params_test_type_2.yml')
        ParamsValidator(params_helper).validate()

        metadata = params_helper.get_metadata()
        preprocess_data_params = params_helper.get_preprocess_data_params()

        for method in ['bfill', 'ffill', 'interpolate']:
            preprocess_data_params['fill'] = method

            data = DataConstructor(metadata, preprocess_data_params).read().build()

            filename = 'expected_schema_type_2_' + method + '.csv'
            csv_file_path = FileHelper.get_file_path(filename)
            expected_data = pd.read_csv(csv_file_path, index_col=[0], parse_dates=True)
            assert_frame_equal(data, expected_data)

    def test_schema_type_3(self):
        # Sensor U106755.csv with skiprows

        params_helper = ParamsHelper('params_test_type_3.yml')
        ParamsValidator(params_helper).validate()

        filename = 'expected_schema_type_3.csv'
        csv_file_path = FileHelper.get_file_path(filename)
        expected_data = pd.read_csv(csv_file_path, index_col="sampletime", parse_dates=True)

        metadata = params_helper.get_metadata()
        preprocess_data_params = params_helper.get_preprocess_data_params()

        data = DataConstructor(metadata, preprocess_data_params).read().build()
        assert_frame_equal(data, expected_data)


if __name__ == '__main__':
    unittest.main()
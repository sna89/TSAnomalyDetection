import unittest
from Helpers.data_creator import DataCreator


class TestDataCreator(unittest.TestCase):
    def test_1_dataset_10_min_gran(self):
        num_of_series = 1
        holiday = weekend = higher_freq = False
        start_date = '2016-01-01 00:00'
        end_date = '2016-01-02 00:00'
        granularity = '10min'

        data_creator = DataCreator()
        df, anomalies_df = data_creator.create_dataset(start_date,
                                                       end_date,
                                                       granularity,
                                                       higher_freq,
                                                       weekend,
                                                       holiday,
                                                       num_of_series)
        assert df.shape[0] == 144 and df.shape[1] == num_of_series + 1, 'dataset size is not as expected'
        assert anomalies_df.shape[0] == 1, 'number of  anomalies is not as expected'

    def test_1_dataset_1_hr_gran(self):
        num_of_series = 1
        holiday = weekend = higher_freq = False
        start_date = '2016-01-01 00:00'
        end_date = '2016-01-02 00:00'
        granularity = '1H'

        data_creator = DataCreator()
        df, anomalies_df = data_creator.create_dataset(start_date,
                                                       end_date,
                                                       granularity,
                                                       higher_freq,
                                                       weekend,
                                                       holiday,
                                                       num_of_series)
        assert df.shape[0] == 24 and df.shape[1] == num_of_series + 1, 'dataset size is not as expected'
        assert anomalies_df.shape[0] == 1, 'number of  anomalies is not as expected'

    def test_2_dataset_10_min_gran(self):
        num_of_series = 2
        holiday = weekend = higher_freq = False
        start_date = '2016-01-01 00:00'
        end_date = '2016-01-02 00:00'
        granularity = '10min'

        data_creator = DataCreator()
        df, anomalies_df = data_creator.create_dataset(start_date,
                                                       end_date,
                                                       granularity,
                                                       higher_freq,
                                                       weekend,
                                                       holiday,
                                                       num_of_series)
        assert df.shape[0] == 144 and df.shape[1] == num_of_series + 1, 'dataset size is not as expected'
        assert anomalies_df.shape[0] >= 1 and anomalies_df.shape[0] <= num_of_series + 1, \
            'number of anomalies is not as expected'

    def test_10_dataset_10_min_gran(self):
        num_of_series = 10
        holiday = weekend = higher_freq = False
        start_date = '2016-01-01 00:00'
        end_date = '2016-01-02 00:00'
        granularity = '10min'

        data_creator = DataCreator()
        df, anomalies_df = data_creator.create_dataset(start_date,
                                                       end_date,
                                                       granularity,
                                                       higher_freq,
                                                       weekend,
                                                       holiday,
                                                       num_of_series)
        assert df.shape[0] == 144 and df.shape[1] == num_of_series + 1, 'dataset size is not as expected'
        assert anomalies_df.shape[0] >= 1 and anomalies_df.shape[0] <= num_of_series + 1, \
            'number of anomalies is not as expected'



if __name__ == '__main__':
    unittest.main()
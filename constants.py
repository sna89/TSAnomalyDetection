import os


class Paths:
    output_path = os.path.join(os.getcwd(), 'outputs')


class AnomalyDfColumns:
    Index = 'SampleTime'
    Feature = 'Feature'
    IsAnomaly = 'IsAnomaly'
    Actual = 'Actual'
    Distance = 'Distance'
    Threshold = 'Threshold'
    Prediction = 'Prediction'
    LowerBound = 'LowerBound'
    UpperBound = 'UpperBound'


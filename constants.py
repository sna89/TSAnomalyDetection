import os


class Paths:
    output_path = os.path.join(os.getcwd(), 'outputs')


class AnomalyDfColumns:
    Index = 'SampleTime'
    Feature = 'Feature'
    IsAnomaly = 'IsAnomaly'
    Actual = 'Actual'
    Prediction = 'Prediction'
    LowerBound = 'LowerBound'
    UpperBound = 'UpperBound'
    McVar = 'MCVariance'
    InherentNoise = 'InherentNoise'
    Uncertainty = 'Uncertainty'
    Bootstrap = 'Bootstrap'
    PercentileValue = 'PercentileValue'
    Dropout = 'Dropout'
    NumOfSeries = 'NumOfSeries'






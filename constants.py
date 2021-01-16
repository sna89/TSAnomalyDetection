import os
from dataclasses import dataclass


class Paths:
    output_path = os.path.join(os.getcwd(), 'outputs')


@dataclass
class AnomalyDfColumns:
    IsAnomaly = 'IsAnomaly'
    Y = 'y'
    Distance = 'Distance'
    Threshold = 'Threshold'
    McMean = 'Mean'
    LowerBound = 'LowerBound'
    UpperBound = 'UpperBound'


from dataclasses import dataclass


@dataclass
class Period:
    minutes: int
    hours: int
    days: int
    weeks: int


class TenMinConverterConst:
    MINUTES = 10
    HOURS = 6
    DAYS = HOURS * 24
    WEEKS = DAYS * 7


class TimeFreqConverter:
    def __init__(self):
        pass

    @staticmethod
    def convert_to_num_samples(period: Period, freq: str):
        num_samples = 0
        if freq == "10min":
            num_samples += period.minutes // TenMinConverterConst.MINUTES
            num_samples += period.hours * TenMinConverterConst.HOURS
            num_samples += period.days * TenMinConverterConst.DAYS
            num_samples += period.weeks * TenMinConverterConst.WEEKS
        return num_samples

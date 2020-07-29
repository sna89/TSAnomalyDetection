from Helpers.params_helper import ParamsHelper


class ParamsValidator:
    def __init__(self, params_helper: ParamsHelper):
        self.params_helper = params_helper

    def validate(self):
        detector_type = self.params_helper.get_detector_type()
        metadata = self.params_helper.get_metadata()

        if detector_type == 'esd' and len(metadata) > 1:
            raise Exception('esd is uni-variate model. Got multiple time series in metadata')

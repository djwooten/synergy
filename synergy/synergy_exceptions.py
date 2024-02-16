class ModelNotParameterizedError(Exception):
    """
    The model must be parameterized prior to use. This can be done by calling
    fit(), or setParameters().
    """
    def __init__(self, msg='The model must be parameterized prior to use. This can be done by calling fit(), or setParameters().', *args, **kwargs):
        super().__init__(msg, *args, **kwargs)

class FeatureNotImplemented(Warning):
    def __init__(self, msg="This feature is not yet implemented", *args, **kwargs):
        super().__init__(msg, *args, **kwargs)

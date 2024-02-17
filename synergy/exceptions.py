class ModelNotParameterizedError(Exception):
    """Thrown when a model's parameters are expected, but they are not yet set."""


class ModelNotFitToDataError(Exception):
    """Thrown when a model must have been fit to data, but was not yet."""

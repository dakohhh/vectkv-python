class VectKvException(Exception):
    """Base Exception class"""

class PageVectorDimException(VectKvException):
    """Supplied vector dim not the same as page vector dim"""

class VectKvIdAlreadyExistException(VectKvException):
    """Supplied vector id already exists"""
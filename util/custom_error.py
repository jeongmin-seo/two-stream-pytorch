import datetime


class WrongSelectError(Exception):

    def __init__(self, message):
        Exception.__init__(message)
        self.when = datetime.now()
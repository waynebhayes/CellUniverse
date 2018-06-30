# -*- coding: utf-8 -*-

"""
cellannealer.errorinfo
~~~~~~~~~~~~~~~~~~~~~~
"""

from enum import IntEnum

class CellAnnealerError(Exception):
    pass


# exit codes
class ExitCode(IntEnum):
    SUCCESS = 0
    ERROR = 1
    INTERRUPTED = 2
    UNKNOWN = -1

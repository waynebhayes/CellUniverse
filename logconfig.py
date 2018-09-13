# -*- coding: utf-8 -*-

"""
cellannealer.logconfig
~~~~~~~~~~~~~~~~~~~~~~

This module handles logging.
"""

import contextlib
import logging
import logging.config
import logging.handlers
import os
import pathlib


class MaxLevelFilter(logging.Filter):
    """TFilters out logs above a certain level."""

    def __init__(self, level):
        super().__init__()
        self.level = level

    def filter(self, record):
        return record.levelno < self.level


class BetterRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Ensures that the directory where the log file resides exists."""

    def _open(self):

        directory = pathlib.Path(self.baseFilename).parent
        directory.mkdir(parents=True, exist_ok=True)    # pylint: disable=E1101

        return logging.handlers.RotatingFileHandler._open(self)


def _color_wrap(color):
    def wrapped(message):
        return ''.join([color, message, '\033[0m'])
    return wrapped


class ColorizedStreamHandler(logging.StreamHandler):
    """Adds color to the logging output in a terminal."""

    COLORS = [
        # This needs to be in order from highest logging level to lowest.
        # (logging.CRITICAL, _color_wrap('\033[1;31m')),
        # (logging.ERROR, _color_wrap('\033[31m')),
        # (logging.WARNING, _color_wrap('\033[33m')),
    ]

    def __init__(self, stream=None):
        super().__init__(stream)

    def should_color(self):
        """Returns True if we should color; False otherwise."""

        # If the stream is a tty we should color it
        # if hasattr(self.stream, 'isatty') and self.stream.isatty():
        #     return True

        # If we have an ANSI term, we should color it
        # if os.environ.get('TERM') == 'ANSI':
        #     return True

        # If we have xterm, we should color it
        # if os.environ.get('TERM').startswith('xterm'):
        #     return True

        # If anything else we should not color it
        return False

    def format(self, record):
        msg = logging.StreamHandler.format(self, record)

        if self.should_color():
            for level, color in self.COLORS:
                if record.levelno >= level:
                    msg = color(msg)
                    break

        return msg


def configure(log='', verbosity=0):

    if verbosity >= 1:
        level = 'DEBUG'
    elif verbosity == -1:
        level = 'WARNING'
    elif verbosity == -2:
        level = 'ERROR'
    elif verbosity <= -3:
        level = 'CRITICAL'
    else:
        level = 'INFO'

    # The root logger should match the 'console' level *unless* we
    # specified '--log' to send debug logs to a file.
    root_level = level
    if log:
        root_level = 'DEBUG'

    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'filters': {
            'exclude_warnings': {
                '()': MaxLevelFilter,
                'level': logging.WARNING
            },
        },
        'formatters': {
            'normal': {
                'style': '{',
                'format': '{message}'
            },
            'warn': {
                'style': '{',
                'format': '{levelname}: {message}'
            },
            'logged': {
                'style': '{',
                'format': '[{asctime}] {levelname}: {message}'
            },
        },
        'handlers': {
            'console': {
                'level': level,
                'class': 'logconfig.ColorizedStreamHandler',
                'stream': 'ext://sys.stdout',
                'filters': ['exclude_warnings'],
                'formatter': 'normal'
            },
            'console_errors': {
                'level': 'WARNING',
                'class': 'logconfig.ColorizedStreamHandler',
                'stream': 'ext://sys.stderr',
                'formatter': 'warn',
            },
            'user_log': {
                'level': 'DEBUG',
                'class': 'logconfig.BetterRotatingFileHandler',
                'filename': log or '/dev/null',
                'delay': True,
                'formatter': 'logged'
            },
        },
        'root': {
            'level': root_level,
            'handlers': list(filter(None, [
                'console',
                'console_errors',
                'user_log' if log else None,
            ])),
        }
    }

    logging.config.dictConfig(config)
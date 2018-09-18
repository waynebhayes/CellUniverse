"""
Adds some helpful decorators and functions for making multiprocessing easier to
use, read, and debug.
"""

from __future__ import print_function

import functools
import os
import traceback
from multiprocessing.pool import Pool


def doublestar_function(func):
    """
    Decorates a function to accept a dictionary as keyword arguments.
    """
    @functools.wraps(func)
    def func_wrapper(kwargs):
        return func(**kwargs)
    return func_wrapper


class InterruptablePool(Pool):
    """
    A multiprocessing pool that has an keyboard-interruptable map function.
    """

    def map(self, func, iterable, chunksize=None):
        '''
        Equivalent of `map()` builtin.
        '''
        result = self.map_async(func, iterable, chunksize)

        try:
            while not result.ready():
                result.wait(1)
        except KeyboardInterrupt:
            print("Program has been terminated by user.")
            self.terminate()
            self.join()
            print("Exiting.")
            exit(0)

        return result.get()


class HandleExceptions(object):
    """
    A static class which has the ability to wrap a try-except statement around
    a function (using `decorate`).
    """

    print_func = print  # default is `print`

    @staticmethod
    def set_print(print_func):
        """
        Sets the print function to use in the current process.
        """
        HandleExceptions.print_func = staticmethod(print_func)

    @staticmethod
    def decorate(func):
        """
        Wraps a try-except statement around the function which will print a
        traceback and halt the process in the event of an exception.
        """
        @functools.wraps(func)
        def func_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as error:
                HandleExceptions.print_func(
                    "[PID: {pid}] {tb}".format(pid=os.getpid(),
                                               tb=traceback.format_exc()))
                raise error
        return func_wrapper


def kwargs(**kwargs):
    """
    Creates a dictionary from a function call.
    """
    return kwargs

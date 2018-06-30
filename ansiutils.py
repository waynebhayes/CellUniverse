# -*- coding: utf-8 -*-

"""
cellannealer.ansiutils
~~~~~~~~~~~~~~~~~~~~~~
"""

import platform

_system = platform.system()

if _system == 'Windows':

    from ctypes import pointer, windll, WinError
    from ctypes.wintypes import DWORD, HANDLE, LPDWORD

    STD_OUTPUT = -11
    ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004


    def _check_error(value):
        if value == 0:
            raise WinError()
        return value


    GetStdHandle = windll.kernel32.GetStdHandle
    GetStdHandle.argtypes = [DWORD]
    GetStdHandle.restype = _check_error

    GetConsoleMode = windll.kernel32.GetConsoleMode
    GetConsoleMode.argtypes = [HANDLE, LPDWORD]
    GetConsoleMode.restype = _check_error

    SetConsoleMode = windll.kernel32.SetConsoleMode
    SetConsoleMode.argtypes = [HANDLE, DWORD]
    SetConsoleMode.restype = _check_error


    def _enable_virtual_terminal():
        '''Enable virtual terminal processing.'''

        # Get current modes
        output_mode = DWORD(0)
        GetConsoleMode(_output_handle, pointer(output_mode))

        # Enable virtual terminal output
        output_mode = DWORD(output_mode.value|ENABLE_VIRTUAL_TERMINAL_PROCESSING)
        SetConsoleMode(_output_handle, output_mode)


    # Get handle
    _output_handle = GetStdHandle(DWORD(STD_OUTPUT))

    # Enable virtual terminal
    _enable_virtual_terminal()
import sys
import platform
from pathlib import Path

PLATFORM = platform.system()

PARENT_DIR = Path(sys.argv[0]).parent
PROJECT_DIR = Path('..' if PARENT_DIR == Path('.') else PARENT_DIR / '..').resolve()

import ctypes
from ctypes import wintypes

INFINITE = 0xFFFFFFFF
WAIT_IO_COMPLETION = 0x000000C0
WAIT_OBJECT_0 = 0x00000000
WAIT_TIMEOUT = 0x00000102
WAIT_FAILED = 0xFFFFFFFF

kernel32 = ctypes.windll.kernel32

kernel32.CreateEventA.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.BOOL, wintypes.LPCSTR)
kernel32.CreateEventA.restype = wintypes.HANDLE

kernel32.SetEvent.argtypes = (wintypes.HANDLE,)
kernel32.SetEvent.restype = wintypes.BOOL

kernel32.ResetEvent.argtypes = (wintypes.HANDLE,)
kernel32.ResetEvent.restype = wintypes.BOOL

kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
kernel32.WaitForSingleObject.restype = wintypes.DWORD

kernel32.WaitForSingleObjectEx.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.BOOL)
kernel32.WaitForSingleObjectEx.restype = wintypes.DWORD

class Event:

    def __init__(self, name, manual_reset=False, initial_state=False):
        self.event = kernel32.CreateEventA(0, manual_reset, initial_state, ctypes.c_char_p(name.encode('utf-8')))

    def set(self):
        return kernel32.SetEvent(self.event)
    
    def reset(self):
        return kernel32.ResetEvent(self.event)

    def wait(self, alertable=True):
        if alertable:
            while True:
                result = kernel32.WaitForSingleObjectEx(self.event, INFINITE, True)
                if result != WAIT_IO_COMPLETION:
                    return result
        else:
            return kernel32.WaitForSingleObjectEx(self.event, INFINITE, False)
            

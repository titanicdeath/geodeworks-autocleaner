"""Thin ctypes wrappers over user32/kernel32. No hooks, no process access beyond
reading the foreground window's executable name.

The INPUT structures use fixed-width fields so their layout can be unit-tested.
"""
import ctypes
import sys
from ctypes import wintypes

# SendInput mouse flags
MOVE = 0x0001
LEFTDOWN = 0x0002
LEFTUP = 0x0004
RIGHTDOWN = 0x0008
RIGHTUP = 0x0010
MOVE_NOCOALESCE = 0x2000
VIRTUALDESK = 0x4000
ABSOLUTE = 0x8000

VK_ESCAPE = 0x1B
VK_F6 = 0x75
VK_F7 = 0x76
VK_F8 = 0x77

BUTTONS = {'left': (LEFTDOWN, LEFTUP), 'right': (RIGHTDOWN, RIGHTUP)}


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [('dx', ctypes.c_int32), ('dy', ctypes.c_int32),
                ('mouseData', ctypes.c_uint32), ('dwFlags', ctypes.c_uint32),
                ('time', ctypes.c_uint32), ('dwExtraInfo', ctypes.c_size_t)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [('wVk', ctypes.c_uint16), ('wScan', ctypes.c_uint16),
                ('dwFlags', ctypes.c_uint32), ('time', ctypes.c_uint32),
                ('dwExtraInfo', ctypes.c_size_t)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [('mi', MOUSEINPUT), ('ki', KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [('type', ctypes.c_uint32), ('u', _INPUTUNION)]


def expected_input_size():
    return 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28


def normalize_absolute(x, y, bounds):
    """Screen pixel -> 0..65535 virtual-desktop coordinates for ABSOLUTE|VIRTUALDESK."""
    left, top, width, height = bounds
    if width < 2 or height < 2:
        raise ValueError('Invalid virtual desktop size.')
    if not (left <= x < left + width and top <= y < top + height):
        raise ValueError(f'Point ({x},{y}) is outside the virtual desktop {bounds}.')
    return (int(round((x - left) * 65535 / (width - 1))),
            int(round((y - top) * 65535 / (height - 1))))


class Win32:
    """Real Windows backend. Tests substitute a fake with the same methods."""

    def __init__(self):
        if sys.platform != 'win32':
            raise RuntimeError('Live operation requires Windows.')
        self.user = ctypes.WinDLL('user32', use_last_error=True)
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.dpi_mode = self._set_dpi_awareness()
        u = self.user
        u.SendInput.argtypes = [ctypes.c_uint32, ctypes.POINTER(INPUT), ctypes.c_int]
        u.SendInput.restype = ctypes.c_uint32
        u.GetForegroundWindow.restype = wintypes.HWND
        u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        u.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        u.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        u.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        u.GetAsyncKeyState.argtypes = [ctypes.c_int]
        u.GetAsyncKeyState.restype = ctypes.c_short
        u.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
        u.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
        u.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        u.IsWindow.argtypes = [wintypes.HWND]
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
        if ctypes.sizeof(INPUT) != expected_input_size():
            raise RuntimeError('Unexpected INPUT structure size.')

    def _set_dpi_awareness(self):
        try:
            if self.user.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
                return 'per-monitor-v2'
        except (AttributeError, OSError):
            pass
        try:
            self.user.SetProcessDPIAware()
            return 'system-aware'
        except (AttributeError, OSError):
            return 'unknown'

    # --- input -------------------------------------------------------------
    def send_mouse(self, flags, dx=0, dy=0):
        ev = INPUT()
        ev.type = 0
        ev.u.mi = MOUSEINPUT(int(dx), int(dy), 0, int(flags), 0, 0)
        sent = self.user.SendInput(1, ctypes.byref(ev), ctypes.sizeof(INPUT))
        if sent != 1:
            raise RuntimeError(f'SendInput rejected the event (error {ctypes.get_last_error()}). '
                               'Is Roblox running as administrator?')

    def set_cursor(self, x, y):
        if not self.user.SetCursorPos(int(x), int(y)):
            raise RuntimeError('SetCursorPos failed.')

    def get_cursor(self):
        p = wintypes.POINT()
        if not self.user.GetCursorPos(ctypes.byref(p)):
            raise RuntimeError('GetCursorPos failed.')
        return (p.x, p.y)

    def virtual_bounds(self):
        return tuple(self.user.GetSystemMetrics(i) for i in (76, 77, 78, 79))

    def key_down(self, vk):
        return bool(self.user.GetAsyncKeyState(vk) & 0x8000)

    # --- windows -----------------------------------------------------------
    def foreground(self):
        return self.user.GetForegroundWindow()

    def window_title(self, hwnd):
        buf = ctypes.create_unicode_buffer(512)
        self.user.GetWindowTextW(hwnd, buf, len(buf))
        return buf.value

    def window_exe(self, hwnd):
        pid = wintypes.DWORD()
        self.user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        handle = self.kernel.OpenProcess(0x1000, False, pid.value)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return ''
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(len(buf))
            if self.kernel.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                return buf.value.rsplit('\\', 1)[-1]
            return ''
        finally:
            self.kernel.CloseHandle(handle)

    def window_rect(self, hwnd):
        r = wintypes.RECT()
        self.user.GetWindowRect(hwnd, ctypes.byref(r))
        return (r.left, r.top, r.right, r.bottom)

    def client_rect(self, hwnd):
        r = wintypes.RECT()
        self.user.GetClientRect(hwnd, ctypes.byref(r))
        p = wintypes.POINT(0, 0)
        self.user.ClientToScreen(hwnd, ctypes.byref(p))
        return (p.x, p.y, p.x + r.right, p.y + r.bottom)

    def environment(self, hwnd=None):
        """Read-only facts that affect input/capture. Nothing here is changed."""
        info = dict(dpi_awareness=self.dpi_mode, virtual_bounds=self.virtual_bounds(),
                    swap_buttons=bool(self.user.GetSystemMetrics(23)),
                    input_struct_size=ctypes.sizeof(INPUT))
        speed = ctypes.c_int()
        if self.user.SystemParametersInfoW(0x0070, 0, ctypes.byref(speed), 0):  # SPI_GETMOUSESPEED
            info['mouse_speed_1_20'] = speed.value
        accel = (ctypes.c_int * 3)()
        if self.user.SystemParametersInfoW(0x0003, 0, accel, 0):  # SPI_GETMOUSE
            info['enhance_pointer_precision'] = bool(accel[2])
        try:
            info['system_dpi'] = self.user.GetDpiForSystem()
        except (AttributeError, OSError):
            pass
        if hwnd:
            info.update(window_title=self.window_title(hwnd), window_exe=self.window_exe(hwnd),
                        window_rect=self.window_rect(hwnd), client_rect=self.client_rect(hwnd))
            try:
                info['window_dpi'] = self.user.GetDpiForWindow(hwnd)
            except (AttributeError, OSError):
                pass
        return info

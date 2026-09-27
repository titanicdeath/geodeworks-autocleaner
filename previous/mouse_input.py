"""Ordinary Windows SendInput mouse movement. No drivers or process access."""
import ctypes


class MouseInput(ctypes.Structure):
    # Windows LONG/DWORD are 32-bit even in 64-bit processes. Fixed widths let
    # offline ABI tests check this correctly on Linux as well as Windows.
    _fields_ = [('dx', ctypes.c_int32), ('dy', ctypes.c_int32),
                ('mouseData', ctypes.c_uint32), ('dwFlags', ctypes.c_uint32),
                ('time', ctypes.c_uint32), ('dwExtraInfo', ctypes.c_size_t)]


class InputUnion(ctypes.Union):
    # MOUSEINPUT is the largest member of the Win32 INPUT union.
    _fields_ = [('mi', MouseInput)]


class Input(ctypes.Structure):
    _fields_ = [('type', ctypes.c_uint32), ('data', InputUnion)]


def normalized_position(x, y, bounds):
    left, top, width, height = bounds
    if width < 2 or height < 2:
        raise ValueError('Invalid virtual display dimensions.')
    if not (left <= x < left+width and top <= y < top+height):
        raise ValueError('Requested position is outside the virtual display.')
    return (int(round((x-left)*65535/(width-1))),
            int(round((y-top)*65535/(height-1))))


class MouseMovement:
    backend = 'SendInput absolute movement across virtual desktop'

    def __init__(self, user):
        self.user = user
        self.user.SendInput.argtypes = [ctypes.c_uint32, ctypes.POINTER(Input), ctypes.c_int]
        self.user.SendInput.restype = ctypes.c_uint32
        self.bounds = tuple(user.GetSystemMetrics(i) for i in (76, 77, 78, 79))
        expected = 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28
        if ctypes.sizeof(Input) != expected:
            raise RuntimeError('Unexpected Windows INPUT structure size.')

    def move(self, x, y):
        dx, dy = normalized_position(x, y, self.bounds)
        event = Input()
        event.type = 0  # INPUT_MOUSE
        event.data.mi = MouseInput(dx, dy, 0, 0x0001 | 0x8000 | 0x4000, 0, 0)
        sent = self.user.SendInput(1, ctypes.byref(event), ctypes.sizeof(Input))
        if sent != 1:
            error = getattr(ctypes, 'get_last_error', lambda: 0)()
            raise RuntimeError(f'Windows did not accept the mouse movement event (error {error}).')

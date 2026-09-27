"""Safety: kill switch, focus binding, bounded runs, guaranteed button release.

Every synthetic input goes through Guard.send(). A watchdog thread polls the kill
keys and the foreground window every ~10 ms; when it trips, it releases any held
buttons and all further non-release input is refused.
"""
import threading
import time

from . import winapi as w

ROBLOX_EXES = {'robloxplayerbeta.exe', 'windows10universal.exe', 'robloxplayer.exe'}


class Stop(Exception):
    pass


def is_roblox(api, hwnd):
    if not hwnd:
        return False
    if api.window_title(hwnd).strip() != 'Roblox':
        return False
    exe = api.window_exe(hwnd).lower()
    # An unreadable exe name (access denied) is accepted on the title alone.
    return exe == '' or exe in ROBLOX_EXES or 'roblox' in exe


class Guard:
    def __init__(self, api, log=print, max_seconds=300, max_clicks=50, dry_run=False,
                 corner_stop=True):
        self.api = api
        self.log = log
        self.max_seconds = max_seconds
        self.max_clicks = max_clicks
        self.dry_run = dry_run
        self.corner_stop = corner_stop
        self.hwnd = None
        self.rect = None
        self.deadline = None
        self.reason = None
        self.clicks = 0
        self.held = set()
        self.sent = []          # (time, flags, dx, dy) for diagnostics
        self._lock = threading.RLock()
        self._thread = None
        self._running = False

    # --- binding -----------------------------------------------------------
    def wait_for_roblox(self, timeout=90):
        self.log('Switch to the Roblox window now (click its title bar or Alt-Tab). '
                 'Waiting for Roblox to become the active window...')
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self._kill_pressed():
                raise Stop('Stopped with Esc/F8 before start.')
            hwnd = self.api.foreground()
            if is_roblox(self.api, hwnd):
                self.hwnd = hwnd
                self.rect = self.api.window_rect(hwnd)
                self.deadline = time.monotonic() + self.max_seconds
                self.log(f'Bound to Roblox window {self.rect}. Hold F8 (or Esc) at any time to stop.')
                time.sleep(0.4)  # let the focus click (if any) finish
                return hwnd
            time.sleep(0.05)
        raise Stop('Roblox never became the active window.')

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def close(self):
        self._running = False
        self.release_all()
        if self._thread:
            self._thread.join(timeout=1)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # --- checks ------------------------------------------------------------
    def _kill_pressed(self):
        return self.api.key_down(w.VK_F8) or self.api.key_down(w.VK_ESCAPE)

    def _problem(self):
        if self._kill_pressed():
            return 'Stopped with F8/Esc.'
        if self.deadline is not None and time.monotonic() > self.deadline:
            return f'Time limit ({self.max_seconds:.0f}s) reached.'
        if self.hwnd is not None:
            if self.api.foreground() != self.hwnd:
                return 'Roblox is no longer the active window.'
            if self.api.window_rect(self.hwnd) != self.rect:
                return 'Roblox window moved or resized.'
        if self.corner_stop:
            x, y = self.api.get_cursor()
            if x <= 1 and y <= 1:
                return 'Cursor pushed into the top-left screen corner.'
        return None

    def trip(self, reason):
        with self._lock:
            if self.reason is None:
                self.reason = reason
                self.log(f'STOP: {reason}')
            self.release_all()

    def _watch(self):
        while self._running:
            try:
                problem = self._problem()
            except Exception as exc:  # never let the watchdog die silently
                problem = f'Watchdog error: {exc}'
            if problem:
                self.trip(problem)
            time.sleep(0.01)

    def check(self):
        if self.reason is None:
            problem = self._problem()
            if problem:
                self.trip(problem)
        if self.reason is not None:
            raise Stop(self.reason)

    def sleep(self, seconds):
        end = time.monotonic() + seconds
        while True:
            self.check()
            left = end - time.monotonic()
            if left <= 0:
                return
            time.sleep(min(0.005, left))

    # --- input gate --------------------------------------------------------
    def send(self, flags, dx=0, dy=0):
        releasing = flags & (w.LEFTUP | w.RIGHTUP)
        with self._lock:
            if not releasing:
                self.check()
            if flags & w.LEFTDOWN:
                if self.clicks >= self.max_clicks:
                    raise Stop(f'Click budget ({self.max_clicks}) used up.')
                self.clicks += 1
            self.sent.append((time.perf_counter(), flags, dx, dy))
            if not self.dry_run:
                self.api.send_mouse(flags, dx, dy)
            for name, (down, up) in w.BUTTONS.items():
                if flags & down:
                    self.held.add(name)
                if flags & up:
                    self.held.discard(name)

    def set_cursor(self, x, y):
        with self._lock:
            self.check()
            self.sent.append((time.perf_counter(), 'SetCursorPos', x, y))
            if not self.dry_run:
                self.api.set_cursor(x, y)

    def release_all(self):
        with self._lock:
            for name in list(self.held):
                try:
                    if not self.dry_run:
                        self.api.send_mouse(w.BUTTONS[name][1])
                finally:
                    self.held.discard(name)

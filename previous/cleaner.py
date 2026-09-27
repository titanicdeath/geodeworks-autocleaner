"""Geodeworks screen-driven Windows prototype. Run without arguments for menu."""
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import sys
import time
import zipfile

import cv2
import numpy as np
from vision import observe, annotated, candidate_points, default_config
from mouse_input import MouseMovement

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / 'calibration.json'


class StopRun(Exception):
    pass


class Desktop:
    def __init__(self):
        if sys.platform != 'win32':
            raise StopRun('Live operation requires Windows. Offline analysis works elsewhere.')
        self.user = ctypes.WinDLL('user32', use_last_error=True)
        try:
            self.user.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except (AttributeError, OSError):
            self.user.SetProcessDPIAware()
        self.user.GetForegroundWindow.restype = wintypes.HWND
        self.user.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        self.user.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self.user.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self.user.GetAsyncKeyState.restype = ctypes.c_short
        self.user.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
        self.mouse = MouseMovement(self.user)
        import mss
        factory = getattr(mss, 'MSS', None) or mss.mss
        self.capture = factory()
        self.monitor = self.capture.monitors[1]
        self.hwnd = None
        self.window_rect = None
        self.deadline = float('inf')
        self.clicks = 0
        self.drags = 0
        self.held = 0

    def rect(self, hwnd):
        r = wintypes.RECT()
        if not self.user.GetWindowRect(hwnd, ctypes.byref(r)):
            raise StopRun('Cannot read game window bounds.')
        return [r.left, r.top, r.right, r.bottom]

    def bind(self):
        self.hwnd = self.user.GetForegroundWindow()
        text = ctypes.create_unicode_buffer(512)
        self.user.GetWindowTextW(self.hwnd, text, len(text))
        if text.value.strip().lower() != 'roblox':
            raise StopRun('Foreground window must have the exact title Roblox.')
        self.window_rect = self.rect(self.hwnd)

    def guard(self):
        if self.user.GetAsyncKeyState(0x1B) & 0x8000 or self.user.GetAsyncKeyState(0x77) & 0x8000:
            raise StopRun('Stopped with Esc or F8.')
        if time.monotonic() >= self.deadline:
            raise StopRun('Time limit reached.')
        if self.hwnd and (self.user.GetForegroundWindow() != self.hwnd or
                          self.rect(self.hwnd) != self.window_rect):
            raise StopRun('Roblox lost focus, moved, or resized.')
        p = wintypes.POINT()
        if self.user.GetCursorPos(ctypes.byref(p)):
            mx, my = self.monitor['left'], self.monitor['top']
            if p.x <= mx+2 and p.y <= my+2:
                raise StopRun('Stopped at upper-left screen corner.')

    def wait(self, seconds):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            self.guard()
            time.sleep(.015)

    def frame(self):
        self.guard()
        return np.array(self.capture.grab(self.monitor))[:, :, :3].copy()

    def move(self, point):
        self.guard()
        x, y = np.rint(point).astype(int)
        if not (3 < x < self.monitor['width']-3 and 3 < y < self.monitor['height']-3):
            raise StopRun('Requested mouse point is outside the calibrated display.')
        # Send simulated movement events through the ordinary Windows input queue.
        # A matching GetCursorPos alone does not prove the game updated its aim.
        origin = np.array(self.cursor(), dtype=float)
        target = np.array([x,y], dtype=float)
        for step in range(1, 7):
            self.guard()
            px, py = np.rint(origin+(target-origin)*step/6).astype(int)
            self.mouse.move(int(px+self.monitor['left']), int(py+self.monitor['top']))
            self.wait(.01)

    def cursor(self):
        p = wintypes.POINT()
        if not self.user.GetCursorPos(ctypes.byref(p)):
            raise StopRun('Windows could not read the cursor position.')
        return [p.x-self.monitor['left'], p.y-self.monitor['top']]

    def release(self):
        # Always release buttons, even after focus loss or a stop signal.
        if self.held:
            self.user.mouse_event(self.held, 0, 0, 0, 0)
            self.held = 0

    def click(self, point):
        self.last_click = dict(cursor_before=self.cursor(), requested_point=list(map(int,point)),
                               movement_backend=MouseMovement.backend)
        self.move(point)
        # Give the game time to update its hover/raycast after cursor movement.
        self.wait(.20)
        actual = self.cursor()
        self.last_click['cursor_after_movement'] = actual
        if np.linalg.norm(np.array(actual)-point) > 3:
            raise StopRun(f'Cursor did not reach the requested point: requested={point}, actual={actual}.')
        try:
            self.guard()
            self.held = 0x0004
            self.user.mouse_event(0x0002, 0, 0, 0, 0)
            self.wait(.10)
        finally:
            self.release()
        self.clicks += 1
        return actual

    def drag(self, delta, cfg):
        if self.drags >= 60:
            raise StopRun('Drag limit reached.')
        delta = np.array(delta, dtype=float)
        maximum = min(160, cfg['scope'][2]*.30)
        delta *= min(1., maximum/max(1., np.linalg.norm(delta)))
        origin = np.array(cfg['scope'][:2], dtype=float)
        self.move(origin)
        try:
            self.guard()
            self.held = 0x0010
            self.user.mouse_event(0x0008, 0, 0, 0, 0)
            for step in range(1, 16):
                self.move(origin+delta*step/15)
                self.wait(.018)
        finally:
            self.release()
        self.drags += 1
        self.wait(cfg['settle'])


def countdown(desktop):
    print('Switch to the Roblox microscope on your PRIMARY monitor.', flush=True)
    for seconds in range(5, 0, -1):
        print(f'  Starting in {seconds}...', flush=True)
        desktop.wait(1)
    desktop.bind()


def reveal_window(title):
    """Keep our own image window visible above Roblox; do not send game input."""
    cv2.waitKey(1)  # Pump the window creation messages first.
    try:
        cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        print(f'If the image window is hidden, use Alt-Tab to select {title!r}.', flush=True)
    if sys.platform == 'win32':
        user = ctypes.WinDLL('user32', use_last_error=True)
        user.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
        user.FindWindowW.restype = wintypes.HWND
        user.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        user.SetForegroundWindow.argtypes = [wintypes.HWND]
        hwnd = user.FindWindowW(None, title)
        if hwnd:
            user.ShowWindow(hwnd, 9)  # Restore this program's window only.
            user.SetForegroundWindow(hwnd)  # Windows may deny focus stealing.
    print(f'Opened window: {title}. Use Alt-Tab if it is not visible.', flush=True)


def show(image, title):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, min(1280, image.shape[1]),
                     int(image.shape[0]*min(1280/image.shape[1], 1)))
    cv2.imshow(title, image)
    reveal_window(title)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def calibrate(desktop):
    countdown(desktop)
    print('Capturing the microscope for calibration...', flush=True)
    frame = desktop.frame()
    print('Capture complete. Opening a separate calibration window...', flush=True)
    prompts = ['Microscope CENTER', 'Microscope RIGHT rim', 'Minimap CENTER',
               'Minimap RIGHT rim', 'Survey button CENTER', 'Bright button CENTER',
               'Polarized button CENTER']
    points = []
    title = 'Calibration - click the frozen image, not the game'
    # Resize the image itself, so click coordinates have an explicit scale.
    scale = min(1., 1280/frame.shape[1], 760/frame.shape[0])
    display = cv2.resize(frame, None, fx=scale, fy=scale)
    def clicked(event, x, y, flags, data):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(prompts):
            points.append([x/scale, y/scale])
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(title, 40, 40)
    cv2.setMouseCallback(title, clicked)
    revealed = False
    try:
        while True:
            out = display.copy()
            for p in points:
                cv2.drawMarker(out, tuple(np.rint(np.array(p)*scale).astype(int)),
                               (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
            text = ('Click: '+prompts[len(points)]) if len(points) < len(prompts) else 'ENTER to save'
            cv2.rectangle(out, (0, 0), (out.shape[1], 62), (20, 20, 20), -1)
            cv2.putText(out, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
            cv2.putText(out, 'Right-click: undo | Esc: cancel', (12, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, .50, (200, 200, 200), 1)
            cv2.imshow(title, out)
            if not revealed:
                reveal_window(title)
                revealed = True
            key = cv2.waitKey(25) & 255
            if key == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                raise StopRun('Calibration cancelled.')
            if key == 13 and len(points) == len(prompts):
                break
    finally:
        cv2.destroyAllWindows()
    scope_r = float(np.linalg.norm(np.array(points[1])-points[0]))
    map_r = float(np.linalg.norm(np.array(points[3])-points[2]))
    if not (150 < scope_r < frame.shape[0]*.60 and 20 < map_r < 150):
        raise StopRun('Circle sizes are implausible; calibrate again.')
    cfg = dict(size=[frame.shape[1], frame.shape[0]], scope=points[0]+[scope_r],
               minimap=points[2]+[map_r], modes=points[4:], contrast=20.0,
               min_area=55, settle=.4, window_rect=desktop.window_rect,
               monitor_origin=[desktop.monitor['left'], desktop.monitor['top']])
    CONFIG.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    print('Calibration saved. Next run Preview and inspect the red crosses.')


def load_config(desktop):
    if not CONFIG.exists():
        raise StopRun('Run calibration first.')
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    countdown(desktop)
    if cfg['size'] != [desktop.monitor['width'], desktop.monitor['height']] or \
       cfg['window_rect'] != desktop.window_rect or \
       cfg['monitor_origin'] != [desktop.monitor['left'], desktop.monitor['top']]:
        raise StopRun('Display/window geometry changed. Calibrate again.')
    return cfg


def navigation_matrix(desktop, cfg):
    """Measure map response to two orthogonal drags; reject boundary clamping."""
    columns = []
    step = min(100., cfg['scope'][2]*.25)
    for delta in ([step, 0.], [0., step]):
        before = observe(desktop.frame(), cfg).viewport
        if before is None:
            raise StopRun('Cannot track the minimap viewing ring.')
        desktop.drag(delta, cfg)
        after = observe(desktop.frame(), cfg).viewport
        if after is None:
            raise StopRun('Viewing ring lost during pan calibration.')
        movement = np.array(after[:2])-before[:2]
        desktop.drag(-np.array(delta), cfg)
        restored = observe(desktop.frame(), cfg).viewport
        if restored is None or np.linalg.norm(np.array(restored[:2])-before[:2]) > 2.0:
            raise StopRun('Pan calibration hit a boundary or did not reverse. Pan nearer the map center and retry.')
        if np.linalg.norm(movement) < 1.5 or abs(after[2]-before[2]) > 1.5:
            raise StopRun('Drag response is too small or viewing scale changed.')
        columns.append(movement/step)
    matrix = np.column_stack(columns)
    if np.linalg.cond(matrix) > 8:
        raise StopRun('Pan calibration is ambiguous; stopping.')
    return matrix


class ClickDiagnostics:
    """Local evidence only; no uploads. One raw before/after pair per attempt."""
    def __init__(self, cfg):
        self.folder = ROOT/'diagnostics'/f'run-{time.time_ns()}'
        self.folder.mkdir(parents=True)
        (self.folder/'calibration.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
        self.count = 0

    def start(self, frame, point, source):
        self.count += 1
        stem = f'click-{self.count:03d}'
        if not cv2.imwrite(str(self.folder/f'{stem}-before.png'), frame):
            raise StopRun('Could not save diagnostic screenshot.')
        data = dict(requested_point=list(map(int, point)), source=source,
                    status='prepared', note='Visual progress is a heuristic, not game confirmation.')
        self.update(stem, data)
        return stem, data

    def update(self, stem, data):
        (self.folder/f'{stem}.json').write_text(json.dumps(data, indent=2), encoding='utf-8')

    def finish(self):
        destination = ROOT/'last-test.zip'
        with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self.folder.iterdir()):
                archive.write(path, path.name)
        print(f'Diagnostics saved: {destination}', flush=True)


def measured_click(desktop, cfg, point, frame, evidence, source='automatic'):
    before = observe(frame, cfg)
    stem, data = evidence.start(frame, point, source)
    try:
        actual = desktop.click(point)
        desktop.wait(max(.75, cfg['settle']))
        after_frame = desktop.frame()
        if not cv2.imwrite(str(evidence.folder/f'{stem}-after.png'), after_frame):
            raise StopRun('Could not save diagnostic screenshot.')
        after = observe(after_frame, cfg)
        x, y = point
        old_area = int(before.mask[max(0,y-12):y+13, max(0,x-12):x+13].sum())
        new_area = int(after.mask[max(0,y-12):y+13, max(0,x-12):x+13].sum())
        progress = len(after.dots) < len(before.dots) or new_area < .55*old_area
        data.update(status='input_sent', actual_cursor=actual, visual_progress=bool(progress),
                    markers_before=len(before.dots), markers_after=len(after.dots),
                    local_foreground_before=old_area, local_foreground_after=new_area)
        if isinstance(getattr(desktop, 'last_click', None), dict):
            data['input_details'] = desktop.last_click
        evidence.update(stem, data)
        print(f'Click {desktop.clicks}: requested={point}; cursor={actual}; visual progress={progress}', flush=True)
        return progress
    except Exception as exc:
        data.update(status='interrupted_or_failed', error=str(exc))
        if isinstance(getattr(desktop, 'last_click', None), dict):
            data['input_details'] = desktop.last_click
        evidence.update(stem, data)
        raise


def input_test(desktop, cfg, evidence):
    """User selects a known target; bypass vision-based target selection."""
    desktop.deadline = time.monotonic()+60
    print('Hover the CENTER of a known cleanable block in this mode, then tap F6. Do not click.', flush=True)
    held = bool(desktop.user.GetAsyncKeyState(0x75) & 0x8000)
    while True:
        desktop.guard()
        pressed = bool(desktop.user.GetAsyncKeyState(0x75) & 0x8000)
        if pressed and not held:
            point = desktop.cursor()
            cx, cy, radius = cfg['scope']
            x, y = point
            if np.hypot(x-cx, y-cy) > radius*.9 or not cy-radius*.77 < y < cy+radius*.8:
                print('Choose a feature farther inside the microscope, away from overlays.', flush=True)
            else:
                measured_click(desktop, cfg, point, desktop.frame(), evidence, source='user_hover_F6')
                print('One-click input test ended. Check whether your chosen object disappeared.', flush=True)
                return
        held = pressed
        desktop.wait(.02)


def run_cleaner(desktop, cfg, args, evidence):
    desktop.deadline = time.monotonic()+args.seconds
    blocked = []
    matrix = None
    stalled = 0
    mode_indices = range(3) if args.cycle else [None]
    for mode in mode_indices:
        if mode is not None:
            if desktop.clicks >= args.clicks:
                raise StopRun('Click limit reached.')
            desktop.click(cfg['modes'][mode])
            desktop.wait(cfg['settle']*2)
        matrix, blocked, stalled = None, [], 0
        for iteration in range(150):
            desktop.guard()
            if desktop.clicks >= args.clicks:
                raise StopRun('Click limit reached.')
            frame = desktop.frame()
            ob = observe(frame, cfg)
            if ob.viewport is None:
                raise StopRun('Viewing ring is missing or ambiguous. No further input.')
            if not ob.dots:
                # Two settled observations are required, still not a purity guarantee.
                desktop.wait(cfg['settle']*2)
                if not observe(desktop.frame(), cfg).dots:
                    print('No map markers detected in this mode.')
                    break
                continue
            proposals = candidate_points(ob, cfg)
            proposals = [p for p in proposals if all(np.linalg.norm(np.array(p)-q) > 18 for q in blocked)]
            if proposals:
                p = proposals[0]
                progress = measured_click(desktop, cfg, p, frame, evidence)
                blocked.append(p)  # Never repeatedly hammer one screen coordinate.
                stalled = 0 if progress else stalled+1
                if stalled >= 3:
                    raise StopRun('Three clicks without clear progress. Inspect Preview before continuing.')
                continue
            if not args.pan:
                print('No untried, map-supported clicks in this view. The click budget is a limit, not a quota.')
                if stalled:
                    print('No clear progress on the last attempt. Use option 5 to test input on a target you choose.')
                else:
                    print('Pan manually and run again.')
                break
            # Do not chase a target already near center if it cannot be clicked.
            view = np.array(ob.viewport[:2])
            distances = [np.linalg.norm(np.array(p)-view) for p in ob.dots]
            destination = np.array(ob.dots[int(np.argmin(distances))])
            if min(distances) < ob.viewport[2]*.25:
                raise StopRun('A map marker is centered but has no reliable click candidate.')
            if matrix is None:
                matrix = navigation_matrix(desktop, cfg)
                # Re-observe after calibration before choosing the next action.
                continue
            delta = np.linalg.solve(matrix, destination-view)*.75
            desktop.drag(delta, cfg)
            new = observe(desktop.frame(), cfg)
            if new.viewport is None:
                raise StopRun('Viewing ring lost after dragging.')
            moved = np.linalg.norm(np.array(new.viewport[:2])-view)
            if moved < .6:
                raise StopRun('Panning is not moving the viewing ring.')
            if np.linalg.norm(destination-np.array(new.viewport[:2])) > min(distances)+1:
                raise StopRun('Pan moved away from the target. Recalibration is needed.')
            blocked = []
        else:
            raise StopRun('Iteration limit reached.')
    print('Pass ended. Check purity and remaining dots; completion is not guaranteed.')


def main():
    print('Geodeworks prototype v0.4 - SendInput movement and click diagnostics', flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', choices=['calibrate', 'preview', 'run', 'analyze', 'input-test'])
    parser.add_argument('--image', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'preview.png')
    parser.add_argument('--pan', action='store_true', help='Enable measured right-drag navigation')
    parser.add_argument('--cycle', action='store_true', help='Visit Survey, Bright, Polarized')
    parser.add_argument('--clicks', type=int, default=25, help='Total input click budget, including mode changes')
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument('--contrast', type=float, help='Override background contrast threshold (default 20)')
    args = parser.parse_args()
    if args.command is None:
        print('1 Calibrate\n2 Preview (no input)\n3 Try up to five clicks in this view\n4 Pan and clean all three modes\n5 Test ONE click on a target you choose (hover + F6)')
        choice = input('Select 1-5: ').strip()
        if choice not in ('1', '2', '3', '4', '5'):
            return
        args.command = {'1':'calibrate', '2':'preview', '3':'run', '4':'run', '5':'input-test'}[choice]
        args.clicks = 5 if choice == '3' else 120
        args.pan = args.cycle = choice == '4'
    if args.clicks < 1 or args.seconds <= 0:
        parser.error('Limits must be positive.')
    if args.command == 'analyze':
        if args.image is None:
            parser.error('analyze requires --image FILE')
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise StopRun('Cannot open image.')
        cfg = default_config(frame)
        if args.contrast is not None:
            cfg['contrast'] = args.contrast
        cv2.imwrite(str(args.output), annotated(frame, observe(frame, cfg), cfg))
        print(f'Saved {args.output}')
        return
    desktop = Desktop()
    evidence = None
    try:
        if args.command == 'calibrate':
            calibrate(desktop)
        else:
            if args.command == 'input-test':
                print('After the countdown: hover a known cleanable block, then tap F6 for ONE click.', flush=True)
            cfg = load_config(desktop)
            if args.contrast is not None:
                cfg['contrast'] = args.contrast
            if args.command == 'preview':
                frame = desktop.frame()
                out = annotated(frame, observe(frame, cfg), cfg)
                cv2.imwrite(str(args.output), out)
                print('Red crosses: proposed clicks. Orange: contrasting shapes. Yellow: map markers.')
                show(out, 'Preview - frozen screenshot - any key closes')
            else:
                evidence = ClickDiagnostics(cfg)
                if args.command == 'input-test':
                    input_test(desktop, cfg, evidence)
                else:
                    run_cleaner(desktop, cfg, args, evidence)
    finally:
        desktop.release()
        desktop.capture.close()
        if evidence is not None:
            evidence.finish()


if __name__ == '__main__':
    try:
        main()
    except (StopRun, KeyboardInterrupt) as exc:
        print(f'Stopped: {exc}')
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)

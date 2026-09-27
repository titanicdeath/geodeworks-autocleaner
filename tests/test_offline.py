"""Offline tests: vision on ./samples, registration, input strategies and safety
against a fake Windows/Roblox. No real input or screen capture."""
import ctypes
import os
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from geoclean import vision as v
from geoclean import winapi as w
from geoclean.inputs import MouseDriver, STRATEGIES, M0_STRATEGIES
from geoclean.safety import Guard, Stop
from geoclean import m0
from geoclean.diag import Run
from tests.fakes import FakeApi, FakeScreen

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'samples' / 'raw'
MINI = ROOT / 'samples' / 'minimap'
F6_TARGET = (977, 581)


def load(name):
    im = cv2.imread(str(RAW / name))
    assert im is not None, name
    return im


def shifted_slide(frame, layout, dx, dy):
    """Move the slide content by (dx, dy) under a fixed microscope mask and UI."""
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    moved = cv2.warpAffine(frame, m, frame.shape[1::-1], borderMode=cv2.BORDER_REFLECT)
    out = frame.copy()
    mask = layout.valid_mask()
    out[mask] = moved[mask]
    return out


class LayoutTests(unittest.TestCase):
    def test_scope_and_minimap_on_all_samples(self):
        files = sorted(RAW.glob('*.png'))
        self.assertGreaterEqual(len(files), 10)
        for f in files:
            im = cv2.imread(str(f))
            L = v.detect_layout(im)
            cx, cy, r = L.scope
            with self.subTest(f.name):
                self.assertTrue(940 < cx < 975 and 515 < cy < 540 and 455 < r < 478, L.scope)
                self.assertIsNotNone(L.minimap)
                self.assertTrue(44 < L.minimap[2] < 56, L.minimap)
                info = v.minimap_features(im, L.minimap)
                self.assertIsNotNone(info.ring, 'view ring not found')
                self.assertTrue(14 < info.ring[2] < 24, info.ring)
                if f.name == 'white-rock-on-faded-pair.png':
                    self.assertEqual(len(info.dots), 1)       # taken with one Survey dot left
                elif f.name == 'green-cluster-missed.png':
                    # Live 2026-09-27 (v0.9): ~10 overlapping dots merged into one blob
                    # and were all thrown away, so Survey looked empty with 10 left.
                    self.assertTrue(8 <= len(info.dots) <= 11, len(info.dots))
                else:
                    self.assertGreater(len(info.dots), 5)

    def test_minimap_crops(self):
        for f in sorted(MINI.glob('*.png')):
            im = cv2.imread(str(f))
            c = v._circle_blob(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), min_r=20)
            with self.subTest(f.name):
                self.assertIsNotNone(c)
                info = v.minimap_features(im, c)
                self.assertIsNotNone(info.ring)
                self.assertGreater(len(info.dots), 10)

    def test_non_microscope_scene_rejected(self):
        rng = np.random.default_rng(0)
        scene = cv2.GaussianBlur(rng.integers(40, 220, (1080, 1920, 3), dtype=np.uint8), (0, 0), 9)
        with self.assertRaises(ValueError):
            v.detect_layout(scene)

    def test_ui_excluded_from_valid_area(self):
        L = v.detect_layout(load('green-f6-before.png'))
        m = L.valid_mask()
        self.assertFalse(m[109, 960])       # purity text
        self.assertFalse(m[962, 780])       # Survey button
        self.assertFalse(m[134, 1855])      # minimap
        self.assertTrue(m[527, 960])        # slide centre
        self.assertTrue(L.is_clickable_area(*F6_TARGET))
        self.assertFalse(L.is_clickable_area(960, 120))


class ClickJudgeTests(unittest.TestCase):
    def setUp(self):
        self.L = v.detect_layout(load('green-f6-before.png'))

    def test_manual_f6_success_is_removed(self):
        r = v.judge_click(load('green-f6-before.png'), load('green-f6-after.png'), F6_TARGET, self.L)
        self.assertTrue(r['removed'], r)
        self.assertGreater(r['hud_changed_px'], 20)
        self.assertLess(r['minimap_dots_after'], r['minimap_dots_before'])

    def test_v03_automatic_clicks_changed_nothing(self):
        for i, p in [(1, (1140, 204)), (2, (936, 228)), (3, (774, 465))]:
            b = load(f'green-v03auto-click-00{i}-before.png')
            a = load(f'green-v03auto-click-00{i}-after.png')
            r = v.judge_click(b, a, p, self.L)
            with self.subTest(i):
                self.assertFalse(r['removed'])
                self.assertTrue(r['local']['cursor_on_object'])   # the aim was right
                self.assertEqual(r['local']['patch_changed_px'], 0)
                self.assertEqual(r['hud_changed_px'], 0)

    def test_live_m0_removals(self):
        # Live M0 run 2026-09-25 (teal mineral). S4 clicked one of two overlapping
        # blocks; only that one vanished, which the whole-component metric missed.
        for name, p in [('s1', (1206, 331)), ('s4', (927, 828))]:
            b, a = load(f'teal-live-{name}-before.png'), load(f'teal-live-{name}-after.png')
            r = v.judge_click(b, a, p, v.detect_layout(b))
            with self.subTest(name):
                self.assertTrue(r['removed'], r)

    def test_view_movement_flagged(self):
        b = load('green-f6-before.png')
        a = shifted_slide(b, self.L, 60, 0)
        r = v.judge_click(b, a, F6_TARGET, self.L)
        self.assertIn('unreliable', r['verdict'])


class RegistrationTests(unittest.TestCase):
    def test_recovers_known_shift_with_sign(self):
        for name in ('green-f6-before.png', 'tan-survey-panned.png', 'tan-survey-polarized-lines.png'):
            f = load(name)
            L = v.detect_layout(f)
            for dx, dy in [(37, -21), (-120, 0), (0, 95), (5, 3)]:
                g = shifted_slide(f, L, dx, dy)
                mx, my, resp = v.phase_shift(f, g, L)
                with self.subTest(name=name, shift=(dx, dy)):
                    self.assertAlmostEqual(mx, dx, delta=1.0)
                    self.assertAlmostEqual(my, dy, delta=1.0)
                    self.assertGreater(resp, 0.05)

    def test_featureless_view_reports_low_confidence(self):
        f = load('green-f6-before.png')
        L = v.detect_layout(f)
        flat = f.copy()
        flat[L.valid_mask()] = (170, 200, 175)
        self.assertLess(v.phase_shift(flat, flat, L)[2], 0.05)


class WinapiTests(unittest.TestCase):
    def test_input_struct_size(self):
        self.assertEqual(ctypes.sizeof(w.INPUT), w.expected_input_size())

    def test_normalize_absolute(self):
        b = (0, 0, 3840, 1080)
        self.assertEqual(w.normalize_absolute(0, 0, b), (0, 0))
        self.assertEqual(w.normalize_absolute(3839, 1079, b), (65535, 65535))
        with self.assertRaises(ValueError):
            w.normalize_absolute(3840, 5, b)
        self.assertEqual(w.normalize_absolute(-1920, 0, (-1920, 0, 3840, 1080)), (0, 0))


def quiet(*a, **k):
    pass


class InputTests(unittest.TestCase):
    def driver(self, api, **kw):
        g = Guard(api, quiet, corner_stop=False, **kw)
        g.hwnd, g.rect = api.HWND, api.rect
        return g, MouseDriver(g, api, quiet)

    def test_rel_closed_loop_exact_under_acceleration(self):
        for gain in (0.6, 1.0, 1.7):
            api = FakeApi(cursor=(300, 700), rel_gain=gain)
            g, d = self.driver(api)
            d.rel_closed_loop((977, 581), step_px=12, step_delay=0)
            with self.subTest(gain=gain):
                self.assertLessEqual(max(abs(api.get_cursor()[0] - 977), abs(api.get_cursor()[1] - 581)), 1)

    def test_every_strategy_arrives_and_wiggle_restores(self):
        for s in M0_STRATEGIES:
            if s.move == 'none':
                continue
            api = FakeApi(cursor=(1400, 300))
            g, d = self.driver(api)
            tr = d.move_to((977, 581), s)
            with self.subTest(s.name):
                self.assertEqual(api.get_cursor(), (977, 581))
                self.assertEqual(tr['final_error_px'], 0.0)
                rel = [e for e in api.events if e[0] != 'SetCursorPos' and not e[0] & w.ABSOLUTE]
                self.assertEqual(bool(rel), s.move == 'rel' or s.wiggle)

    def test_v04_replica_uses_only_absolute_moves(self):
        api = FakeApi(cursor=(1400, 300))
        g, d = self.driver(api)
        d.move_to((977, 581), STRATEGIES['S1-abs-path'])
        self.assertEqual(len(api.events), 6)
        self.assertTrue(all(e[0] & w.ABSOLUTE and not e[0] & w.MOVE_NOCOALESCE for e in api.events))


class SafetyTests(unittest.TestCase):
    def guard(self, api, **kw):
        g = Guard(api, quiet, corner_stop=False, **kw)
        g.hwnd, g.rect = api.HWND, api.rect
        return g

    def test_kill_key_blocks_input_and_releases(self):
        api = FakeApi()
        g = self.guard(api)
        g.send(w.RIGHTDOWN)
        api.keys.add(w.VK_F8)
        with self.assertRaises(Stop):
            g.send(w.MOVE, 1, 0)
        self.assertEqual(api.events[-1][0], w.RIGHTUP)
        self.assertFalse(g.held)

    def test_watchdog_releases_on_focus_loss(self):
        api = FakeApi()
        g = self.guard(api)
        g.start()
        try:
            g.send(w.LEFTDOWN)
            api.fg = 999
            time.sleep(0.1)
            self.assertIn('no longer the active window', g.reason)
            self.assertEqual(api.events[-1][0], w.LEFTUP)
            with self.assertRaises(Stop):
                g.send(w.LEFTDOWN)
        finally:
            g.close()

    def test_click_releases_when_stopped_mid_hold(self):
        api = FakeApi()
        g = self.guard(api)
        d = MouseDriver(g, api, quiet)
        g.deadline = time.monotonic() + 0.02
        with self.assertRaises(Stop):
            d.click('left', hold=0.2)
        self.assertEqual(api.events[-1][0], w.LEFTUP)

    def test_click_budget(self):
        api = FakeApi()
        g = self.guard(api, max_clicks=1)
        d = MouseDriver(g, api, quiet)
        d.click('left', 0)
        with self.assertRaises(Stop):
            d.click('left', 0)

    def test_dry_run_sends_nothing(self):
        api = FakeApi(cursor=(100, 100))
        g = self.guard(api, dry_run=True)
        d = MouseDriver(g, api, quiet)
        d.move_to((977, 581), STRATEGIES['S4-rel'])
        d.click('left', 0)
        self.assertEqual(api.events, [])

    def test_only_roblox_is_accepted(self):
        api = FakeApi()
        api.fg = 999
        g = Guard(api, quiet)
        with self.assertRaises(Stop):
            g.wait_for_roblox(timeout=0.1)


class ClickTestFlow(unittest.TestCase):
    """End-to-end M0 click trials against a fake Roblox that only tracks the
    pointer from relative/physical moves (the leading hypothesis)."""

    def run_trials(self, names, roblox_sees):
        before, after = load('green-f6-before.png'), load('green-f6-after.png')
        layout = v.detect_layout(before)
        api = FakeApi(cursor=F6_TARGET, roblox_sees=roblox_sees)
        screen = FakeScreen(before, after)

        def pressed(pos, button):
            if button == 'left' and np.hypot(*(pos - F6_TARGET)) < 20:
                screen.removed = True
        api.on_press = pressed
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        run = Run('test', root=tmp.name)
        run.log = lambda *a, **k: None
        g = Guard(api, quiet, corner_stop=False, max_clicks=10)
        g.hwnd, g.rect = api.HWND, api.rect
        g.start()
        d = MouseDriver(g, api, quiet)
        results = []
        try:
            for i, n in enumerate(names, 1):
                s = STRATEGIES[n]
                screen.removed = False
                api.physical_move(*F6_TARGET)
                away = None if s.move == 'none' else (lambda: api.physical_move(1250, 700))
                api.press_key(w.VK_F6, then=away)
                results.append(m0.click_trial(i, len(names), s, api, screen, g, d, layout, run))
        finally:
            g.close()
            run.close()
        return {r['strategy']: r for r in results}, Path(run.folder)

    def test_relative_only_roblox(self):
        res, folder = self.run_trials(['S0-control', 'S1-abs-path', 'S4-rel', 'S2-setcursor-wiggle'],
                                      roblox_sees=('rel', 'physical'))
        self.assertTrue(res['S0-control']['removed'])
        self.assertFalse(res['S1-abs-path']['removed'])      # reproduces the live v0.4 failure
        self.assertTrue(res['S4-rel']['removed'])
        self.assertTrue(res['S2-setcursor-wiggle']['removed'])
        self.assertTrue((folder / 'trial-03-S4-rel' / 'zoom.png').exists())
        data = json.loads((folder / 'trial-02-S1-abs-path' / 'trial.json').read_text())
        self.assertEqual(data['move_trace']['final_error_px'], 0.0)
        self.assertIn('Removed by: S0-control, S4-rel, S2-setcursor-wiggle',
                      m0.click_summary(list(res.values())))

    def test_skip_with_f7(self):
        api = FakeApi(cursor=F6_TARGET)
        before = load('green-f6-before.png')
        layout = v.detect_layout(before)
        g = Guard(api, quiet, corner_stop=False)
        g.hwnd, g.rect = api.HWND, api.rect
        api.press_key(w.VK_F7)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        run = Run('test', root=tmp.name)
        run.log = lambda *a, **k: None
        r = m0.click_trial(1, 1, STRATEGIES['S4-rel'], api, FakeScreen(before, before), g,
                           MouseDriver(g, api, quiet), layout, run)
        run.close()
        self.assertEqual(r['verdict'], 'skipped')


class DragScreen:
    """Fake microscope: while right is held, the slide follows Roblox's pointer
    offset from where the button went down, times `gain` (displacement model)."""

    def __init__(self, api, frame, layout, gain):
        self.api, self.frame, self.layout, self.gain = api, frame, layout, gain
        self.base = np.zeros(2)
        self.anchor = None
        self.origin = (0, 0)
        api.on_press = self.press
        api.on_release = self.release

    def press(self, pos, button):
        if button == 'right':
            self.anchor = pos

    def release(self, pos, button):
        if button == 'right' and self.anchor is not None:
            self.base += (pos - self.anchor) * self.gain
            self.anchor = None

    def grab(self):
        off = self.base.copy()
        if self.anchor is not None:
            off += (self.api.roblox_pos - self.anchor) * self.gain
        return shifted_slide(self.frame, self.layout, off[0], off[1])

    def to_screen(self, p):
        return (int(round(p[0])), int(round(p[1])))

    def to_image(self, p):
        return p

    def info(self):
        return {}


class DragFlowTests(unittest.TestCase):
    def run_drag(self, method, roblox_sees, gain=0.8):
        f0 = load('green-f6-before.png')
        L = v.detect_layout(f0)
        api = FakeApi(cursor=(1300, 300), roblox_sees=roblox_sees)
        screen = DragScreen(api, f0, L, gain)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        run = Run('test', root=tmp.name)
        run.log = lambda *a, **k: None
        g = Guard(api, quiet, corner_stop=False, max_clicks=0)
        g.hwnd, g.rect = api.HWND, api.rect
        try:
            a = m0.drag_run(method, api, screen, g, MouseDriver(g, api, quiet), L, run, amp=100)
        finally:
            g.close()
            run.close()
        self.assertFalse(g.held)
        self.assertEqual(api.events[-1][0] & w.RIGHTUP, w.RIGHTUP)
        return a

    def test_rel_drag_measures_gain(self):
        a = self.run_drag('rel', ('rel', 'physical'))
        self.assertTrue(a['view_moved'])
        self.assertAlmostEqual(float(np.median(a['gains'])), 0.8, delta=0.05)
        self.assertIn('DISPLACEMENT-like', a['text'])

    def test_abs_drag_ignored_by_relative_only_roblox(self):
        a = self.run_drag('abs', ('rel', 'physical'))
        self.assertTrue(a['cursor_followed'])
        self.assertFalse(a['view_moved'])


class LiveDragSampleTests(unittest.TestCase):
    """Frames from the live M0 drag run (2026-09-25, relative moves)."""

    def test_live_rel_drag(self):
        folder = ROOT / 'samples' / 'drag'
        env = json.loads((folder / 'env.json').read_text(encoding='utf-8'))['layout']
        L = v.Layout(tuple(env['scope']), tuple(env['minimap']), tuple(env['size']))
        records = json.loads((folder / 'live-rel' / 'records.json').read_text(encoding='utf-8'))
        for r in records:
            r['frame'] = 'live-rel/' + r['frame'].split('/', 1)[1]
        a = m0.analyze_drag(folder, 'rel', records, L)
        rows = {r['label']: r for r in a['rows']}
        self.assertTrue(a['view_moved'] and a['cursor_followed'])
        self.assertIn('DISPLACEMENT-like', a['text'])
        # The ring moves only 2-5 px in this run, so ~0.3 px of centre noise makes the
        # scale loose here; the M1 run (30 px of ring travel) pins it at ~26.
        self.assertTrue(22 < a['ring_scale'] < 31, a['ring_scale'])
        self.assertTrue(0.8 < float(np.median(a['gains'])) < 0.95)
        # Every stop moves along the commanded axis only, in the commanded direction.
        for r in a['rows']:
            ox, oy = r['offset_cmd']
            if (ox, oy) != (0, 0):
                along = r['shift'][0] if ox else r['shift'][1]
                across = r['shift'][1] if ox else r['shift'][0]
                self.assertGreater(along * (ox or oy), 0, r)
                self.assertLess(abs(across), 3, r)
        self.assertAlmostEqual(rows['at+120+0-0.75s']['shift'][0], 101, delta=3)


class DragAnalysisTests(unittest.TestCase):
    def test_displacement_gain_and_return(self):
        f0 = load('green-f6-before.png')
        L = v.detect_layout(f0)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        folder = Path(tmp.name)
        (folder / 'drag-rel').mkdir()
        records = []
        gain = 1.0
        for i, (label, off) in enumerate([('start', (0, 0)), ('at+60+0-0.25s', (60, 0)),
                                           ('at+60+0-0.75s', (60, 0)), ('at+0+0-0.25s', (0, 0)),
                                           ('at+0+0-0.75s', (0, 0)), ('at+0+60-0.25s', (0, 60)),
                                           ('at+0+60-0.75s', (0, 60))]):
            name = f'drag-rel/{i:02d}-{label}.jpg'
            cv2.imwrite(str(folder / name), shifted_slide(f0, L, off[0] * gain, off[1] * gain),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            records.append(dict(label=label, offset_cmd=list(off), cursor=list(off), frame=name))
        a = m0.analyze_drag(folder, 'rel', records, L)
        self.assertTrue(a['view_moved'])
        self.assertAlmostEqual(float(np.median(a['gains'])), gain, delta=0.05)
        self.assertLess(max(a['drift']), 2)
        self.assertIn('DISPLACEMENT-like', a['text'])


if __name__ == '__main__':
    unittest.main()


class RegistrationLargeShiftTests(unittest.TestCase):
    def test_large_synthetic_shifts_with_rough_prior(self):
        from geoclean import register as R
        for name in ('green-f6-before.png', 'tan-survey-panned.png', 'teal-live-s4-before.png'):
            f = load(name)
            L = v.detect_layout(f)
            for dx, dy in [(300, 0), (-250, 180), (0, -330)]:
                g = shifted_slide(f, L, dx, dy)
                mx, my, score = R.register(f, g, L, prior=(dx + 17, dy - 12))
                with self.subTest(name=name, shift=(dx, dy)):
                    self.assertAlmostEqual(mx, dx, delta=0.6)
                    self.assertAlmostEqual(my, dy, delta=0.6)
                    self.assertGreater(score, 0.9)

    def test_live_pairs_with_minimap_prior(self):
        # Live M0 frames. Truth from the stop-to-stop chain (M0 analysis).
        from geoclean import register as R
        folder = ROOT / 'samples' / 'drag'
        env = json.loads((folder / 'env.json').read_text(encoding='utf-8'))['layout']
        L = v.Layout(tuple(env['scope']), tuple(env['minimap']), tuple(env['size']))
        recs = json.loads((folder / 'live-rel' / 'records.json').read_text(encoding='utf-8'))
        fr = [cv2.imread(str(folder / 'live-rel' / r['frame'].split('/', 1)[1])) for r in recs]
        rings = [R.ring_center(f, L) for f in fr]
        for i, j, truth in [(0, 5, (100.9, -1.0)), (0, 11, (0, 108.4)), (5, 17, (-211.5, 0)),
                            (0, 23, (0.4, -116.7))]:
            s, score, src = R.measure_shift(fr[i], fr[j], L, rings[i], rings[j], 26.0)
            with self.subTest(pair=(i, j)):
                self.assertEqual(src, 'image')
                self.assertLess(np.hypot(*(s - np.array(truth))), 3.0, (s, truth))

    def test_featureless_falls_back_to_minimap(self):
        from geoclean import register as R
        f = load('green-f6-before.png')
        L = v.detect_layout(f)
        flat = f.copy()
        flat[L.valid_mask()] = (170, 200, 175)
        s, score, src = R.measure_shift(flat, flat, L, np.array([1850., 130.]), np.array([1846., 131.]), 26.0)
        self.assertEqual(src, 'minimap')
        self.assertTrue(np.allclose(s, [104, -26]))


class LivePanSampleTests(unittest.TestCase):
    """Frames from the first live M1 run (2026-09-25), incl. slide corners."""

    def setUp(self):
        folder = ROOT / 'samples' / 'pan'
        env = json.loads((folder / 'env.json').read_text(encoding='utf-8'))['layout']
        self.L = v.Layout(tuple(env['scope']), tuple(env['minimap']), tuple(env['size']))
        self.f = {n: cv2.imread(str(folder / f'{n}.jpg')) for n in
                  ('031', '032', '036', '037', '038', '044', '047', '048')}

    def test_ring_found_at_corners(self):
        # The ring runs off the round minimap at the corners; it used to be missed.
        for n, f in self.f.items():
            with self.subTest(n):
                ring = v.minimap_features(f, self.L.minimap).ring
                self.assertIsNotNone(ring)
                self.assertAlmostEqual(ring[2], v.RING_R_FRAC * self.L.minimap[2], delta=0.2)
        r36 = v.minimap_features(self.f['036'], self.L.minimap).ring
        r47 = v.minimap_features(self.f['047'], self.L.minimap).ring
        self.assertGreater(r36[0] - self.L.minimap[0], 25)     # bottom-right corner
        self.assertGreater(r36[1] - self.L.minimap[1], 25)
        self.assertLess(r47[0] - self.L.minimap[0], -25)       # bottom-left corner

    def test_big_live_shift_with_ring_prior(self):
        from geoclean import register as R
        a, b = self.f['031'], self.f['032']
        ra, rb = R.ring_center(a, self.L), R.ring_center(b, self.L)
        s, score, src = R.measure_shift(a, b, self.L, ra, rb, 26.0)
        self.assertEqual(src, 'image')
        self.assertLess(np.hypot(*(s - np.array([418.9, 3.2]))), 3.0)

    def test_featureless_corner_is_not_read_as_no_motion(self):
        from geoclean import register as R
        a, b = self.f['036'], self.f['037']
        s, score, src = R.measure_shift(a, b, self.L, R.ring_center(a, self.L), R.ring_center(b, self.L),
                                        26.0, predicted=(-240, -240))
        self.assertNotEqual(src, 'none')


class LiveTrackingReplayTests(unittest.TestCase):
    """Replays the recorded drags of the two good live M1 runs (2026-09-25 23:03 and
    23:05). Truth = image-measured view displacement between the first and last
    visit to the centre (closure check): (-4.65, 27.32) and (-9.97, 41.47)."""

    RUNS = [('drags-live5.jsonl', (-4.65, 27.32)), ('drags-live6.jsonl', (-9.97, 41.47))]
    MINIMAP = np.array([1855.5, 134.5])     # layout minimap centre in those runs

    def replay(self, recs, scale, blend):
        V = Vc = None
        first_edge = next(i for i, r in enumerate(recs) if r['tag'].startswith('edge'))
        for i, r in enumerate(recs):
            Vr = (np.array(r['ring']) - self.MINIMAP) * scale if r.get('ring') else None
            if V is None:
                V = Vr
            if i == first_edge:
                Vc = V.copy()
            if r['source'] == 'minimap' and Vr is not None:
                V = Vr
                continue
            V = V - np.array(r['shift'])
            if Vr is not None and r['source'] == 'image':
                V = V + blend * (Vr - V)
                if np.linalg.norm(V - Vr) > 60:
                    V = Vr
        return V - Vc

    def test_per_axis_scale_and_bounded_drift(self):
        from geoclean.m1 import fit_ring_scale
        for name, truth in self.RUNS:
            recs = [json.loads(l) for l in (ROOT / 'samples' / 'pan' / name).read_text().splitlines()]
            k, n, rms = fit_ring_scale(recs)
            with self.subTest(name):
                self.assertAlmostEqual(k[0], 26.2, delta=0.2)          # x
                self.assertAlmostEqual(k[1], 25.1, delta=0.35)         # y (foreshortened)
                est = self.replay(recs, np.array(k), blend=0.25)
                self.assertLess(np.hypot(*(est - truth)), 10.0, (est, truth))


class PanModelTests(unittest.TestCase):
    def test_fit_gain_recovers_matrix(self):
        from geoclean.pan import fit_gain
        G = np.array([[0.87, 0.02], [-0.01, 0.9]])
        C = np.array([[60, 0], [-60, 0], [0, 60], [0, -60], [200, 0], [0, -200], [150, 90]], float)
        Gf, rms, rel = fit_gain(C, C @ G.T)
        self.assertTrue(np.allclose(Gf, G, atol=1e-6))
        self.assertLess(rms, 1e-6)

    def test_save_load_roundtrip(self):
        from geoclean.pan import PanModel, save_model, load_model
        m = PanModel(G=[[0.9, 0], [0, 0.85]], ring_scale=25.5, style='jump')
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'cal.json'
            save_model(path, m, extra=dict(extent_min=500))
            m2, data = load_model(path)
        self.assertEqual(m2, m)
        self.assertEqual(data['extent_min'], 500)

    def test_drag_endpoints_stay_in_safe_box(self):
        from geoclean.pan import Panner, PanModel
        L = v.detect_layout(load('green-f6-before.png'))
        p = Panner(None, None, None, None, L, PanModel())
        x0, y0, x1, y1 = p.box
        for d in [(420, 0), (-420, 300), (0, -500), (300, 300)]:
            a, b = p._endpoints(d)
            for q in (a, b):
                self.assertTrue(x0 - 1e-6 <= q[0] <= x1 + 1e-6 and y0 - 1e-6 <= q[1] <= y1 + 1e-6, (d, q))
            self.assertTrue(np.allclose(b - a, d))


class PanCalibrateFlow(unittest.TestCase):
    """Full M1 flow against a fake slide (sparse, circular edge, minimap ring)."""

    FAST = dict(anchor=0.002, press=0.002, settle=0.005, release=0.002, step_delay=0.0, event=0.001)

    def run_flow(self, quantum=0, styles=('jump', 'step10'), fixed_timing=FAST, **fake):
        from geoclean import m1
        from tests.fakes import FakeSlide
        base = load('green-f6-before.png')
        L = v.detect_layout(base)
        api = FakeApi(cursor=(1300, 400), roblox_sees=('abs', 'rel', 'physical'))
        opts = dict(gain=0.87, radius=600, scale=26.0, quantum=quantum, px_per_block=80000)
        opts.update(fake)
        slide = FakeSlide(api, base, L, **opts)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        run = Run('test', root=tmp.name)
        run.log = lambda *a, **k: None
        api.press_key(w.VK_F6)
        old = m1.CALIBRATION
        m1.CALIBRATION = Path(tmp.name) / 'pan_calibration.json'
        try:
            args = SimpleNamespace(seconds=600, dry_run=False)
            out = m1.pan_calibrate(api, slide, run, args, styles=styles,
                                   timing=fixed_timing, fixed_timing=fixed_timing)
            saved = m1.CALIBRATION.exists()
        finally:
            m1.CALIBRATION = old
            run.close()
        return out, slide, saved

    def test_calibrates_gain_scale_edges_and_accuracy(self):
        out, slide, saved = self.run_flow()
        self.assertNotIn('stopped', out)
        self.assertTrue(saved)
        G = np.array(out['model']['G'])
        self.assertTrue(np.allclose(G, 0.87 * np.eye(2), atol=0.02), G)
        for s in out['model']['ring_scale']:
            self.assertAlmostEqual(s, 26.0, delta=1.0)
        extents = [e['extent'] for e in out['edges']]
        self.assertEqual(len(extents), 8)
        for e in extents:
            self.assertAlmostEqual(e, 600, delta=30)
        self.assertLessEqual(max(a['error'] for a in out['accuracy']), 5.0)
        # Ground truth: the tracker's final position vs the fake's real view position.
        self.assertLess(np.hypot(*(np.array(out['closure']['pan_to']['V']) - slide.V)), 8.0)
        self.assertLess(out['closure']['true_residual_px'], 6.0)

    # Live M1 runs 5-6: minimap scale x 26.2 / y ~25.0, ring drawn on a 0.5 px grid,
    # view clamped at exactly 30 minimap px (x 786, y 750), no overshoot at the edge.
    LIVE_LIKE = dict(gain=1.0, bounds=('square', (786, 750)), overshoot=0, grain=0, content_radius=850,
                     fps=60, scale=(26.2, 25.0), ring_grid=0.5)

    @unittest.skipUnless(os.environ.get('GEOCLEAN_SLOW'), 'slow (~90 s): set GEOCLEAN_SLOW=1')
    def test_live_like_square_slide(self):
        # Square bounds, nothing to register against near the edges, per-axis minimap
        # scale with a coarse ring, and a game that samples the cursor once per frame
        # (a press sharing a frame with the preceding move corrupts the drag).
        # Real frame-gap timing is used, so this is the slow test.
        out, slide, saved = self.run_flow(styles=('v1600', 'v800'), fixed_timing=None, **self.LIVE_LIKE)
        self.assertNotIn('stopped', out)
        self.assertTrue(saved, out.get('quality_problems'))
        self.assertEqual(out['styles']['chosen'], 'v1600')
        sx, sy = out['model']['ring_scale']
        self.assertAlmostEqual(sx, 26.2, delta=0.4)
        self.assertAlmostEqual(sy, 25.0, delta=0.4)
        b = out['bounds']
        self.assertEqual(b['shape'], 'square', b)
        self.assertAlmostEqual(b['ring_limit_px'], 30.0, delta=0.6)
        self.assertEqual(sum(1 for e in out['edges'] if e['extent']), 8)
        self.assertLessEqual(max(a['error'] for a in out['accuracy']), 6.0)
        self.assertLess(out['closure']['true_residual_px'], 12.0)
        self.assertLess(np.hypot(*(np.array(out['closure']['pan_to']['V']) - slide.V)), 15.0)

    def test_unregistered_drag_start_is_caught_not_saved(self):
        # Reproduces the live re-run failure: the game hadn't registered the move to
        # the drag start when the button went down, so it counted the jump from the
        # previous drag's end as part of the drag. Modelled deterministically by not
        # moving to the drag start at all. Must stop, not save a bad calibration.
        from geoclean import m1, pan
        from geoclean.inputs import Strategy
        bare = Strategy('anchor', 'none', wiggle=False, dwell=0.0)
        old = (pan.ANCHOR_MOVE, m1.ANCHOR_MOVE)
        pan.ANCHOR_MOVE = m1.ANCHOR_MOVE = bare
        try:
            out, slide, saved = self.run_flow(
                styles=('v800',), fixed_timing=dict(anchor=0.001, press=0.001, release=0.001, settle=0.05),
                **self.LIVE_LIKE)
        finally:
            pan.ANCHOR_MOVE, m1.ANCHOR_MOVE = old
        self.assertIn('inconsistent', out.get('stopped', ''))
        self.assertFalse(saved)

    def test_lossy_drags_prefer_single_jump(self):
        out, slide, saved = self.run_flow(quantum=10, styles=('jump', 'step3'))
        self.assertEqual(out['styles']['chosen'], 'jump')
        self.assertLessEqual(max(a['error'] for a in out['accuracy']), 15.0)   # up to 10 px lost per drag
        self.assertTrue(all(a['drags'] <= 6 for a in out['accuracy']))
        self.assertFalse(out['styles']['fits']['step3']['usable'])


class M2PieceTests(unittest.TestCase):
    def test_grid_covers_the_slide(self):
        from geoclean.mapping import plan_grid
        lim = (782.2, 765.0)
        pts = np.array(plan_grid(lim))
        self.assertEqual(len(pts), 16)
        # Every point the view centre can reach must be seen well inside some view:
        # within 0.94 r of its centre and below the HUD band (cy - 0.80 r).
        r = 466.2
        for x in np.linspace(-lim[0], lim[0], 25):
            for y in np.linspace(-lim[1], lim[1], 25):
                d = np.array([x, y]) - pts
                ok = (np.hypot(*d.T) < 0.94 * r - 40) & (d[:, 1] > -0.80 * r + 40) & (d[:, 1] < 0.84 * r - 40)
                self.assertTrue(ok.any(), (x, y))

    def test_alpha_model_picks_the_opaque_mode(self):
        from geoclean.detect import alpha_mode
        C = np.array([120.0, 150.0, 110.0])
        bg = {'survey': np.array([160.0, 128, 140]), 'bright': np.array([200.0, 126, 132]),
              'polarized': np.array([90.0, 140, 110])}
        for own in bg:
            obs = {m: ((C if m == own else bg[m] + 0.35 * (C - bg[m])), bg[m]) for m in bg}
            with self.subTest(own):
                self.assertEqual(alpha_mode(obs)[0], own)

    def test_mode_buttons_on_samples(self):
        from geoclean import modes as MO
        for f in sorted(RAW.glob('*.png')):
            im = cv2.imread(str(f))
            st = MO.read_buttons(im, v.detect_layout(im))
            with self.subTest(f.name):
                self.assertEqual(st['active'], 'survey')
                self.assertTrue(all(st['present']))

    def test_detection_counts_on_samples(self):
        # Hand-checked: green = 7 blocks + faded block + half-visible block at the
        # bottom (the stacked pair counts once); tan = 4 rock blocks, 1 faded brown
        # block, 1 cream speck (+ 1 under the HUD); teal = busy frame, ~20-24.
        from geoclean import detect as D
        for name, lo, hi in [('green-f6-before.png', 7, 9), ('tan-survey-panned.png', 5, 7),
                             ('teal-live-s4-before.png', 18, 25)]:
            im = load(name)
            objs = D.detect_objects({'survey': D.contrast_map(im, v.detect_layout(im))})
            with self.subTest(name):
                self.assertTrue(lo <= len(objs) <= hi, len(objs))
                for o in objs:     # click point is on the object itself
                    x, y, w, h = o.bbox
                    self.assertTrue(o.mask[int(o.click[1]) - y, int(o.click[0]) - x])

    def test_f6_removal_seen_by_detection(self):
        from geoclean import detect as D
        b, a = load('green-f6-before.png'), load('green-f6-after.png')
        L = v.detect_layout(b)
        nb = D.detect_objects({'survey': D.contrast_map(b, L)})
        na = D.detect_objects({'survey': D.contrast_map(a, L)})
        self.assertEqual(len(nb) - len(na), 1)
        self.assertFalse(any(np.hypot(o.centroid[0] - 977, o.centroid[1] - 584) < 20 for o in na))


class M2FlowTests(unittest.TestCase):
    """Full map-slide sweep + analysis against a fake microscope with modes."""

    def run_sweep(self, bounds=(500, 480), n_objects=15):
        from geoclean import m2
        from geoclean.pan import PanModel, save_model
        from tests.fakes import FakeScope
        base = load('green-f6-before.png')
        L = v.detect_layout(base)
        api = FakeApi(cursor=(1300, 400), roblox_sees=('abs', 'rel', 'physical'))
        scope = FakeScope(api, base, L, gain=1.0, bounds=('square', bounds), scale=(26.2, 25.0), ring_grid=0.5,
                          start=(40, -30), n_objects=n_objects)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        model = PanModel(G=[[1.0, 0], [0, 1.0]], ring_scale=[26.2, 25.0], style='v1600',
                         timing=dict(anchor=0.002, press=0.002, settle=0.005, release=0.002, step_delay=0.0,
                                     event=0.001))
        cal = Path(tmp.name) / 'cal.json'
        save_model(cal, model, extra=dict(bounds=dict(V_limit_from_ring=list(bounds))))
        old = m2.CALIBRATION
        m2.CALIBRATION = cal
        run = Run('map', root=tmp.name)
        run.log = lambda *a, **k: None
        api.press_key(w.VK_F6)
        try:
            meta = m2.sweep(api, scope, run, SimpleNamespace(seconds=600, dry_run=False,
                                                             modes=['survey', 'bright', 'polarized'], spacing=540.0))
        finally:
            m2.CALIBRATION = old
            run.close()
        res = json.loads((Path(run.folder) / 'targets.json').read_text())
        return meta, res, scope, Path(run.folder)

    @staticmethod
    def dist(t, o):
        p = np.array(t['S'])
        if o['kind'] == 'block':
            return float(np.linalg.norm(p - o['S']))
        d = np.array([np.cos(np.radians(o['angle'])), np.sin(np.radians(o['angle']))])
        s = np.clip(np.dot(p - o['S'], d), -o['length'] / 2, o['length'] / 2)
        return float(np.linalg.norm(p - (o['S'] + s * d)))

    def test_map_finds_and_classifies(self):
        # The fake zooms Bright 1.40x and Polarized 2.08x like the game; mapping must
        # only pan and capture in Survey, and get modes from the minimap dots.
        meta, res, scope, folder = self.run_sweep()
        self.assertNotIn('stopped', meta)
        self.assertEqual(scope.mode, 'survey')                 # left back in Survey
        self.assertEqual(scope.zoomed_drags, 0)                # never panned while zoomed
        self.assertLessEqual(len(scope.clicks), 3)             # S->B->P->S once
        self.assertLess(meta['minimap_info']['view_after_return']['shift_px'], 2.0)
        T = res['targets']
        found = wrong = 0
        for o in scope.objects:
            d = [self.dist(t, o) for t in T]
            k = int(np.argmin(d))
            if d[k] < 12:
                found += 1
                wrong += T[k]['mode'] != o['mode']
        extra = sum(min(self.dist(t, o) for o in scope.objects) >= 12 for t in T)
        self.assertEqual(found, len(scope.objects))
        self.assertLessEqual(wrong, 1)
        self.assertLessEqual(extra, 1)
        s = res['summary']
        self.assertLess(s['pair_residual_px']['max'], 1.0)
        for m in ('survey', 'bright', 'polarized'):
            self.assertLessEqual(s['minimap'][m]['dots_without_target'], 1, m)
        for f in ('map.png', 'mosaic-survey.jpg', 'map-report.txt', 'sweep.json', 'minimap-bright.png'):
            self.assertTrue((folder / f).exists(), f)


class LiveMapSampleTests(unittest.TestCase):
    """The live map-slide run of 2026-09-26 01:06. The user cleaned the slide
    afterwards and counted 37 imperfections; by eye the Survey mosaic shows 22
    rock blocks, 5 faded blocks, ~8 line segments, plus 4 floating specks."""

    def test_live_map(self):
        import shutil
        from geoclean import mapping as MP
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        folder = Path(tmp.name) / 'run'
        shutil.copytree(ROOT / 'samples' / 'map-live1', folder)
        s, targets = MP.analyze(folder, log=lambda *a, **k: None)
        self.assertTrue(30 <= s['targets'] <= 40, s['targets'])
        mm = s['minimap']
        self.assertGreaterEqual(mm['survey']['matched'], 14)
        self.assertGreaterEqual(mm['bright']['matched'], 4)
        self.assertGreaterEqual(mm['polarized']['matched'], 6)
        self.assertGreaterEqual(s['specks'], 3)
        lines = [t for t in targets if t['kind'] == 'line']
        # ~8 real segments; currently 13 because a few short duplicate/fragment
        # pieces survive (known over-count, marked 'faded?' on the map).
        self.assertTrue(6 <= len(lines) <= 14, len(lines))
        self.assertTrue(all(t['mode'] in ('polarized', 'faded?') for t in lines))
        blocks = [t for t in targets if t['kind'] == 'block']
        self.assertTrue(22 <= len(blocks) <= 28, len(blocks))


class M3FlowTests(unittest.TestCase):
    """map + clean against the fake microscope: zooming modes, click-to-remove
    only for the current mode's objects, minimap dots per mode."""

    def run_clean(self, modes, n_objects=15, hide=0):
        from geoclean import m2, m3
        from geoclean.pan import PanModel, save_model
        from tests.fakes import FakeScope
        base = load('green-f6-before.png')
        L = v.detect_layout(base)
        api = FakeApi(cursor=(1300, 400), roblox_sees=('abs', 'rel', 'physical'))
        bounds = (500, 480)
        scope = FakeScope(api, base, L, gain=1.0, bounds=('square', bounds), scale=(26.2, 25.0), ring_grid=0.5,
                          start=(40, -30), n_objects=n_objects)
        for o in [o for o in scope.objects if o['mode'] == 'survey'][:hide]:
            scope.hidden_dots.add(id(o))
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        model = PanModel(G=[[1.0, 0], [0, 1.0]], ring_scale=[26.2, 25.0], style='v1600',
                         timing=dict(anchor=0.002, press=0.002, settle=0.005, release=0.002, step_delay=0.0,
                                     event=0.001))
        cal = Path(tmp.name) / 'cal.json'
        save_model(cal, model, extra=dict(bounds=dict(V_limit_from_ring=list(bounds))))
        old = m2.CALIBRATION
        m2.CALIBRATION = cal
        run = Run('clean', root=tmp.name)
        run.log = lambda *a, **k: None
        api.press_key(w.VK_F6)
        try:
            out = m3.clean(api, scope, run, SimpleNamespace(seconds=900, dry_run=False, modes=modes, max_clicks=200))
        finally:
            m2.CALIBRATION = old
            run.close()
        return out, scope

    def check(self, out, scope, modes):
        self.assertNotIn('stopped', out)
        self.assertEqual(scope.missed_clicks, [])            # every click removed something
        for o in scope.objects:
            self.assertEqual(bool(o.get('removed')), o['mode'] in modes, (o['mode'], o['kind'], o['S']))
        # Each cleaned mode ends with its button greyed out (live game behaviour),
        # and the cleaner recognised that.
        self.assertEqual(scope.done, set(modes))
        for m in modes:
            self.assertTrue(out['results'][m]['finished'], m)

    def test_clean_survey(self):
        out, scope = self.run_clean(['survey'])
        self.check(out, scope, ['survey'])
        self.assertEqual(scope.zoomed_drags, 0)               # Survey is never zoomed

    @unittest.skipUnless(os.environ.get('GEOCLEAN_SLOW'), 'slow (~70 s): set GEOCLEAN_SLOW=1')
    def test_clean_all_modes(self):
        modes = ['survey', 'bright', 'polarized']
        out, scope = self.run_clean(modes)
        self.check(out, scope, modes)
        # Survey is greyed out after its turn, so Bright/Polarized must pan zoomed;
        # the fake moves 1.15/zoom slide px per cursor px there, which must be learned.
        self.assertGreater(scope.zoomed_drags, 0)
        adj = [p['gain_adj'] for p in out['pans'] if p['mode'] != 'survey']
        self.assertAlmostEqual(adj[-1], 1.15, delta=0.08)

    @unittest.skipUnless(os.environ.get('GEOCLEAN_SLOW'), 'slow: set GEOCLEAN_SLOW=1')
    def test_unfinished_mode_is_revisited(self):
        # User 2026-09-27: a mode left with imperfections must be recognised as not
        # finished (its button is not greyed out) and revisited. One Survey dot is
        # hidden on the first pass, so Survey ends pass 1 unfinished.
        modes = ['survey', 'bright', 'polarized']
        out, scope = self.run_clean(modes, n_objects=7, hide=1)
        self.check(out, scope, modes)
        self.assertGreaterEqual(out['results']['survey']['rounds'], 2)


class M3PieceTests(unittest.TestCase):
    def test_greyed_button_means_mode_done(self):
        # Live 2026-09-27: once Survey was fully cleaned its button turned grey
        # (BGR ~64,56,52), no mode was highlighted, and it could not be selected.
        from geoclean import modes as MO
        im = load('white-rock-on-faded-pair.png').copy()
        L = v.detect_layout(im)
        cx, cy, r = L.scope
        for k, (x, y) in enumerate(MO.button_centers(L)[:1]):
            cv2.rectangle(im, (int(x - 0.12 * r), int(y - 0.043 * r)), (int(x + 0.12 * r), int(y + 0.043 * r)),
                          (64, 56, 52), -1)
        st = MO.read_buttons(im, L)
        self.assertEqual(st['states'][:3], ['disabled', 'enabled', 'enabled'])
        self.assertIsNone(st['active'])
        self.assertEqual(st['done'], ['survey'])
        self.assertTrue(all(st['present']))

    def cleaner(self, im):
        from geoclean import m3
        L = v.detect_layout(im)
        c = m3.Cleaner.__new__(m3.Cleaner)
        c.layout, c.mode = L, 'survey'
        c.zoom = {'survey': dict(zoom=1.0, centre=list(L.scope[:2]))}
        c.p = SimpleNamespace(V=np.zeros(2))
        return c, L

    def test_rock_on_faded_pair_clicks_the_rock(self):
        # Live 2026-09-26 02:55: a full-contrast rock block lying on two faded Bright
        # blocks merged into one blob; the blob's deepest point was on a faded block
        # and 4 clicks did nothing. The click must land on the rock (~(940, 515)).
        im = load('white-rock-on-faded-pair.png')
        c, L = self.cleaner(im)
        objs = c.candidates(im)
        blob = max(objs, key=lambda o: o['obj'].area)
        S = np.array([940 - L.scope[0], 515 - L.scope[1]])
        pt, kind = c.click_point(blob, c.pred(S))
        self.assertEqual(kind, 'block')
        self.assertLess(np.hypot(pt[0] - 940, pt[1] - 515), 20, pt)
        # Uniform single blocks keep their own deepest point.
        for o in objs:
            if o is not blob and o['obj'].area > 3000:
                p2, _ = c.click_point(o, np.array(o['obj'].click))
                self.assertLess(np.hypot(p2[0] - o['obj'].click[0], p2[1] - o['obj'].click[1]), 1.0)

    def test_quick_judge_on_live_clicks(self):
        # The patch-only check that replaced the full-frame judge (speed) must
        # agree with it on the live before/after pairs.
        from geoclean import detect as D
        cases = [('green-f6', F6_TARGET, 'removed'), ('teal-live-s1', (1206, 331), 'removed'),
                 ('teal-live-s4', (927, 828), 'removed')]
        cases += [(f'green-v03auto-click-00{i}', p, 'no_change')
                  for i, p in [(1, (1140, 204)), (2, (936, 228)), (3, (774, 465))]]
        for name, p, want in cases:
            b, a = load(f'{name}-before.png'), load(f'{name}-after.png')
            c, L = self.cleaner(b)
            r = c.quick_judge(D.contrast_map(b, L), a, p)
            with self.subTest(name):
                self.assertEqual(r['outcome'], want, r)

    def test_survey_never_picks_smooth_objects(self):
        # Live 2026-09-27 (v0.9): all 5 Survey clicks without effect were on cyan
        # Polarized lines lying near a Survey dot.
        for f in ('tan-survey-polarized-lines.png', 'teal-live-s1-before.png'):
            im = load(f)
            c, L = self.cleaner(im)
            objs = c.candidates(im)
            c.dead = {}
            for o in objs:
                x, y = o['obj'].click
                S = np.array([x - L.scope[0], y - L.scope[1]], float)
                hit = c.pick(objs, S, 30, set())
                with self.subTest(f, at=(x, y)):
                    if hit is not None:
                        self.assertGreaterEqual(objs[hit[0]]['itex'], 1.5)
                    if o['itex'] >= 2.0:
                        self.assertIsNotNone(hit)             # rock still gets clicked

    def test_specks_are_never_candidates(self):
        # The white-mineral screenshot has a speck at ~(901, 898).
        im = load('white-rock-on-faded-pair.png')
        c, _ = self.cleaner(im)
        for o in c.candidates(im):
            self.assertGreaterEqual(o['obj'].area, 600)

    def test_view_plan_covers_every_dot_with_few_views(self):
        from geoclean import m3
        c = m3.Cleaner.__new__(m3.Cleaner)
        c.p = SimpleNamespace(V=np.zeros(2))
        rng = np.random.default_rng(2)
        dots = [rng.uniform(-700, 700, 2) for _ in range(15)]
        R = 317.0
        views = c.plan_views(dots, R)
        covered = [g for _, grp in views for g in grp]
        self.assertEqual(len(covered), len(dots))
        for centre, grp in views:
            for g in grp:
                self.assertLessEqual(np.linalg.norm(g - centre), R + 1e-6)
        self.assertLess(len(views), len(dots) * 0.7)

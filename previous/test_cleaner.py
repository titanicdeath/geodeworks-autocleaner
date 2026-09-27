"""Offline checks. No desktop capture or real mouse input is used."""
import time
import json
import tempfile
import ctypes
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import cv2
import numpy as np
from vision import observe, candidate_points
from cleaner import Desktop, StopRun, reveal_window, input_test, measured_click, ClickDiagnostics
from mouse_input import MouseMovement, Input, normalized_position


def scene(bg=(210, 215, 220), foreground=(160, 30, 40)):
    im = np.zeros((600, 900, 3), dtype='uint8')
    cv2.circle(im, (400, 300), 240, bg, -1)
    cv2.rectangle(im, (383, 283), (417, 317), foreground, -1)
    cv2.circle(im, (830, 80), 45, (110, 20, 65), -1)
    cv2.circle(im, (830, 80), 16, (240, 225, 240), 2)
    cv2.circle(im, (830, 80), 3, (160, 220, 130), -1)
    cfg = dict(scope=[400, 300, 240], minimap=[830, 80, 45], contrast=20, min_area=40)
    return im, cfg


class VisionTests(unittest.TestCase):
    def test_changing_colors(self):
        for bg, fg in [((230,230,230),(200,30,30)), ((100,120,140),(220,240,180)),
                       ((220,215,210),(80,90,110))]:
            im, cfg = scene(bg, fg)
            ob = observe(im, cfg)
            points = candidate_points(ob, cfg)
            self.assertTrue(points)
            for x, y in points:
                self.assertTrue(383 <= x <= 417 and 283 <= y <= 317)

    def test_blue_minimap_background(self):
        im, cfg = scene()
        region = np.all(im == (110,20,65), axis=2)
        im[region] = (245,195,115)
        ob = observe(im, cfg)
        self.assertEqual(len(ob.dots), 1)
        self.assertIsNotNone(ob.viewport)

    def test_no_marker_no_click(self):
        im, cfg = scene()
        cv2.circle(im, (830, 80), 5, (110,20,65), -1)
        self.assertEqual(candidate_points(observe(im, cfg), cfg), [])

    def test_edge_projection_moves_to_shape_interior(self):
        im, cfg = scene()
        cv2.circle(im, (830,80), 5, (110,20,65), -1)
        # This marker projects just outside the right edge of the square.
        cv2.circle(im, (832,80), 3, (160,220,130), -1)
        ob = observe(im,cfg)
        points = candidate_points(ob,cfg)
        self.assertTrue(points)
        depth = cv2.distanceTransform(ob.mask, cv2.DIST_L2, 5)
        for x,y in points:
            self.assertGreater(depth[y,x], 10)

    def test_no_ring_no_click(self):
        im, cfg = scene()
        cv2.circle(im, (830, 80), 16, (110,20,65), 4)
        self.assertEqual(candidate_points(observe(im, cfg), cfg), [])

    def test_bent_shape_click_is_on_shape(self):
        im, cfg = scene()
        cv2.circle(im, (400,300), 240, (210,215,220), -1)
        cv2.polylines(im, [np.array([[340,340],[400,280],[470,340]])], False,
                      (160,30,40), 14)
        ob = observe(im, cfg)
        points = candidate_points(ob, cfg)
        self.assertTrue(points)
        for x, y in points:
            self.assertEqual(ob.mask[y,x], 1)


class ControlTests(unittest.TestCase):
    def test_sendinput_normalization_handles_negative_monitor_origins(self):
        self.assertEqual(normalized_position(-1920, 0, (-1920,0,3840,1080)), (0,0))
        self.assertEqual(normalized_position(1919,1079,(-1920,0,3840,1080)), (65535,65535))
        with self.assertRaises(ValueError):
            normalized_position(1920,1079,(-1920,0,3840,1080))

    def test_sendinput_uses_windows_abi_and_movement_flags(self):
        user = Mock()
        user.GetSystemMetrics.side_effect = [0,0,1920,1080]
        packets = []
        def send(count, pointer, size):
            event = pointer._obj
            packets.append((count, size, event.type, event.data.mi.dwFlags,
                            event.data.mi.dx, event.data.mi.dy))
            return 1
        user.SendInput.side_effect = send
        movement = MouseMovement(user)
        movement.move(1919,1079)
        expected_size = 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28
        self.assertEqual(packets, [(1,expected_size,0,0xC001,65535,65535)])

    def test_rejected_movement_reports_failure(self):
        user = Mock()
        user.GetSystemMetrics.side_effect = [0,0,1920,1080]
        user.SendInput.return_value = 0
        with self.assertRaisesRegex(RuntimeError, 'did not accept'):
            MouseMovement(user).move(100,100)

    def test_controller_moves_with_events_instead_of_cursor_teleport(self):
        d = self.desktop()
        d.cursor = Mock(return_value=[400,300])
        d.wait = Mock()
        d.mouse = Mock()
        d.move([600,400])
        self.assertEqual(d.mouse.move.call_count,6)
        self.assertEqual(d.mouse.move.call_args.args,(600,400))
        d.user.SetCursorPos.assert_not_called()

    def test_image_window_requests_topmost(self):
        with patch('cleaner.sys.platform', 'linux'), \
             patch('cleaner.cv2.waitKey') as pump, \
             patch('cleaner.cv2.setWindowProperty') as prop:
            reveal_window('Calibration')
        pump.assert_called_once_with(1)
        prop.assert_called_once_with('Calibration', cv2.WND_PROP_TOPMOST, 1)

    def desktop(self):
        d = Desktop.__new__(Desktop)
        d.user = Mock()
        d.user.GetAsyncKeyState.return_value = 0
        d.user.GetForegroundWindow.return_value = 42
        d.user.GetCursorPos.return_value = 0
        d.hwnd = 42
        d.window_rect = [0,0,900,600]
        d.rect = Mock(return_value=d.window_rect)
        d.deadline = float('inf')
        d.monitor = dict(left=0,top=0,width=900,height=600)
        d.held = 0
        return d

    def test_focus_loss_stops(self):
        d = self.desktop()
        d.user.GetForegroundWindow.return_value = 99
        with self.assertRaises(StopRun):
            d.guard()

    def test_window_move_stops(self):
        d = self.desktop()
        d.rect.return_value = [1,0,901,600]
        with self.assertRaises(StopRun):
            d.guard()

    def test_time_and_key_stops(self):
        d = self.desktop()
        d.deadline = time.monotonic()-1
        with self.assertRaises(StopRun):
            d.guard()
        d.deadline = float('inf')
        d.user.GetAsyncKeyState.return_value = 0x8000
        with self.assertRaises(StopRun):
            d.guard()

    def test_preview_cleanup_sends_no_input(self):
        d = self.desktop()
        d.release()
        d.user.mouse_event.assert_not_called()

    def test_click_interruption_releases_button(self):
        d = self.desktop()
        d.move = Mock()
        d.cursor = Mock(return_value=[400,300])
        d.wait = Mock(side_effect=[None, StopRun('interrupted')])
        with self.assertRaises(StopRun):
            d.click([400,300])
        flags = [call.args[0] for call in d.user.mouse_event.call_args_list]
        self.assertEqual(flags, [2,4])
        self.assertEqual(d.held, 0)

    def test_click_waits_for_cursor_before_press(self):
        d = self.desktop()
        d.move = Mock()
        d.cursor = Mock(return_value=[400,300])
        d.clicks = 0
        def waited(seconds):
            if seconds == .20:
                d.user.mouse_event.assert_not_called()
            elif seconds == .10:
                self.assertEqual(d.held, 4)
        d.wait = Mock(side_effect=waited)
        self.assertEqual(d.click([400,300]), [400,300])
        self.assertEqual(d.clicks, 1)

    def test_cursor_mismatch_never_presses(self):
        d = self.desktop()
        d.move = Mock()
        d.wait = Mock()
        d.cursor = Mock(return_value=[800,500])
        with self.assertRaises(StopRun):
            d.click([400,300])
        d.user.mouse_event.assert_not_called()

    def test_f6_test_uses_user_target_once(self):
        im, cfg = scene()
        d = Mock()
        d.user.GetAsyncKeyState.side_effect = [0, 0x8000]
        d.cursor.return_value = [400,300]
        d.frame.return_value = im
        evidence = Mock()
        with patch('cleaner.measured_click', return_value=False) as click:
            input_test(d,cfg,evidence)
        click.assert_called_once_with(d,cfg,[400,300],im,evidence,source='user_hover_F6')

    def test_diagnostics_save_raw_pair_and_cursor(self):
        im,cfg = scene()
        cfg['settle'] = .4
        after = im.copy()
        cv2.rectangle(after,(383,283),(417,317),(210,215,220),-1)
        d = Mock()
        d.click.return_value = [400,300]
        d.frame.return_value = after
        d.clicks = 1
        with tempfile.TemporaryDirectory() as folder, patch('cleaner.ROOT',Path(folder)):
            evidence = ClickDiagnostics(cfg)
            self.assertTrue(measured_click(d,cfg,[400,300],im,evidence))
            metadata = json.loads((evidence.folder/'click-001.json').read_text())
            self.assertEqual(metadata['actual_cursor'], [400,300])
            self.assertEqual(metadata['status'], 'input_sent')
            self.assertTrue((evidence.folder/'click-001-before.png').exists())
            self.assertTrue((evidence.folder/'click-001-after.png').exists())
            evidence.finish()
            self.assertTrue((Path(folder)/'last-test.zip').exists())


if __name__ == '__main__':
    unittest.main()

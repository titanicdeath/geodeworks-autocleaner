"""Primary-monitor capture with mss.MSS. Image pixel (x, y) == screen (x+left, y+top)."""
import numpy as np


class Screen:
    def __init__(self):
        import mss
        self.sct = mss.MSS()
        self.mon = dict(self.sct.monitors[1])
        self.origin = (self.mon['left'], self.mon['top'])

    def grab(self):
        return np.array(self.sct.grab(self.mon))[:, :, :3].copy()

    def to_screen(self, p):
        return (int(round(p[0])) + self.origin[0], int(round(p[1])) + self.origin[1])

    def to_image(self, p):
        return (p[0] - self.origin[0], p[1] - self.origin[1])

    def info(self):
        return dict(monitor=self.mon, all_monitors=[dict(m) for m in self.sct.monitors])

    def close(self):
        self.sct.close()

"""Mode zoom: Bright and Polarized magnify the view about (roughly) the
microscope centre. Measured offline from the 2026-09-26 frames: Bright 1.40x,
Polarized ~2.08x, centre within a few px of the microscope centre.

A screen point in the zoomed view relates to the slide as:
    screen = C + z * (S - V)        (C = zoom centre on screen, V = view centre)
"""
import cv2
import numpy as np

DEFAULT_ZOOM = {'survey': 1.0, 'bright': 1.40, 'polarized': 2.08}


def measure_zoom(survey_frame, zoomed_frame, layout, expected, min_inliers=10):
    """Similarity transform zoomed -> survey via SIFT + RANSAC.

    Returns dict(zoom, centre=(x, y) screen, inliers) or None when there are too
    few features (sparse slides) or the result is implausible."""
    cx, cy, r = layout.scope
    mask = np.zeros(survey_frame.shape[:2], np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(0.85 * r), 255, -1)
    mask[:int(cy - 0.75 * r)] = 0
    mask[int(cy + 0.80 * r):] = 0
    sift = cv2.SIFT_create(4000)
    ka, da = sift.detectAndCompute(cv2.cvtColor(survey_frame, cv2.COLOR_BGR2GRAY), mask)
    kb, db = sift.detectAndCompute(cv2.cvtColor(zoomed_frame, cv2.COLOR_BGR2GRAY), mask)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return None
    pairs = cv2.BFMatcher().knnMatch(db, da, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < min_inliers:
        return None
    pb = np.float32([kb[g.queryIdx].pt for g in good])
    pa = np.float32([ka[g.trainIdx].pt for g in good])
    M, inl = cv2.estimateAffinePartial2D(pb, pa, method=cv2.RANSAC, ransacReprojThreshold=3)
    if M is None or int(inl.sum()) < min_inliers:
        return None
    s = float(np.hypot(M[0, 0], M[1, 0]))          # survey = s * zoomed + t  ->  z = 1/s
    z = 1.0 / s
    if not (0.85 * expected < z < 1.15 * expected):
        return None
    A, t = M[:, :2], M[:, 2]
    centre = np.linalg.solve(np.eye(2) - A, t)       # fixed point of the similarity
    if np.hypot(centre[0] - cx, centre[1] - cy) > 0.2 * r:
        return None
    return dict(zoom=round(z, 4), centre=[round(float(centre[0]), 1), round(float(centre[1]), 1)],
                inliers=int(inl.sum()))

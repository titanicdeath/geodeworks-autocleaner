"""Screen-only candidate detection. All points are image pixels, BGR input."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Observation:
    mask: np.ndarray
    targets: list
    dots: list
    viewport: object
    background: list


def disk(shape, circle, fraction=1.0):
    x, y, r = circle
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (xx-x)**2 + (yy-y)**2 < (r*fraction)**2


def default_config(image):
    """Approximate geometry for screenshot diagnostics ONLY; live use calibrates."""
    h, w = image.shape[:2]
    # Recover the large illuminated disk, excluding the black surroundings.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = (gray > 35).astype('uint8')
    binary[:, :int(w*.20)] = 0
    binary[:, int(w*.80):] = 0
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large = max(contours, key=cv2.contourArea)
    arc = large[:, 0, :].astype(float)
    arc = arc[(arc[:, 1] > h*.22) & (arc[:, 1] < h*.80)]
    ax, ay = arc.T
    cx, cy, constant = np.linalg.lstsq(np.column_stack([2*ax, 2*ay, np.ones(len(ax))]),
                                      ax*ax+ay*ay, rcond=None)[0]
    r = float(np.sqrt(constant+cx*cx+cy*cy))
    # Minimap is isolated in upper right. Fit its largest nonblack component.
    x0, y0 = int(w*.93), int(h*.065)
    crop = gray[y0:int(h*.21), x0:]
    contours, _ = cv2.findContours((crop > 40).astype('uint8'), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    (mx, my), mr = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
    return dict(size=[w, h], scope=[cx, cy, r], minimap=[mx+x0, my+y0, mr],
                contrast=20.0, min_area=55, modes=[], settle=.4)


def map_features(image, circle):
    """Adaptive map-background subtraction; independently fit the pale view ring."""
    mx, my, mr = circle
    x0, y0 = max(0, int(mx-mr-2)), max(0, int(my-mr-2))
    crop = image[y0:int(my+mr+3), x0:int(mx+mr+3)]
    local = [mx-x0, my-y0, mr]
    inside = disk(crop.shape, local, .90)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # View ring is pale, unlike both purple and blue map backgrounds supplied.
    white = (hsv[:, :, 1] < 75) & (hsv[:, :, 2] > 180) & inside
    yy, xx = np.where(white)
    viewport = None
    if len(xx) >= 18:
        # Least-squares circle fit: x^2+y^2 = 2*cx*x+2*cy*y+c.
        a = np.column_stack([2*xx, 2*yy, np.ones(len(xx))])
        solution = np.linalg.lstsq(a, xx*xx+yy*yy, rcond=None)[0]
        vx, vy, c = solution
        vr = np.sqrt(max(0, c+vx*vx+vy*vy))
        residual = np.abs(np.hypot(xx-vx, yy-vy)-vr)
        # Reject missing, occluded, or ambiguous rings instead of guessing.
        if .10*mr < vr < .48*mr and np.median(residual) < 1.4:
            angles = np.arctan2(yy-vy, xx-vx)
            coverage = len(np.unique(np.floor((angles+np.pi)/(2*np.pi)*16)))
            if coverage >= 10:
                viewport = [float(vx+x0), float(vy+y0), float(vr)]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype('float32')
    bg = np.median(lab[inside], axis=0)
    contrast = np.linalg.norm(lab-bg, axis=2)
    # Remove ring before finding small dots. Dot outlines are retained.
    foreground = ((contrast > 20) & inside & ~white &
                  (hsv[:, :, 1] > 65) & (hsv[:, :, 2] > 120)).astype('uint8')
    if viewport is not None:
        gy, gx = np.ogrid[:crop.shape[0], :crop.shape[1]]
        ring_edge = np.abs(np.hypot(gx-vx, gy-vy)-vr) < 2.2
        foreground[ring_edge] = 0
    n, labels, stats, centers = cv2.connectedComponentsWithStats(foreground)
    dots = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if 3 <= area <= 120*(mr/50)**2 and max(w, h) < .30*mr:
            # White ring fragments are not targets; dots have chromatic pixels.
            pixels = hsv[labels == i]
            if np.median(pixels[:, 1]) < 65:
                continue
            dx, dy = centers[i]
            dots.append([float(dx+x0), float(dy+y0)])
    return dots, viewport


def observe(image, config):
    cx, cy, r = config['scope']
    valid = disk(image.shape, config['scope'], .94)
    yy = np.arange(image.shape[0])[:, None]
    # Exclude upper purity/event overlay and lower mode buttons.
    valid &= (yy > cy-.77*r) & (yy < cy+.80*r)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype('float32')
    background = np.median(lab[valid], axis=0)
    difference = np.linalg.norm(lab-background, axis=2)
    mask = ((difference > config.get('contrast', 20)) & valid).astype('uint8')
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), 'uint8'))
    n, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    cleaned = np.zeros_like(mask)
    targets = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < config.get('min_area', 55) or area > valid.sum()*.30:
            continue
        component = (labels[y:y+h, x:x+w] == i).astype('uint8')
        # Padding ensures distance-to-edge is well-defined for cropped shapes.
        distance = cv2.distanceTransform(np.pad(component, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        iy, ix = np.unravel_index(np.argmax(distance), distance.shape)
        if distance[iy, ix] < 2:
            continue
        cleaned[y:y+h, x:x+w][component > 0] = 1
        targets.append(dict(point=[int(x+ix), int(y+iy)], area=int(area),
                            box=[int(x), int(y), int(w), int(h)]))
    dots, viewport = map_features(image, config['minimap'])
    return Observation(cleaned, targets, dots, viewport, background.tolist())


def candidate_points(observation, config):
    """Project nearby map dots and snap to a confidently detected shape interior.

    Ring-to-scope projection is a hypothesis, to validate in the live preview.
    Never fall back to clicking uncorroborated foreground shapes.
    """
    if observation.viewport is None:
        return []
    vx, vy, vr = observation.viewport
    cx, cy, r = config['scope']
    interior = cv2.erode(observation.mask, np.ones((5, 5), 'uint8'))
    _, labels = cv2.connectedComponents(observation.mask)
    clearance = cv2.distanceTransform(observation.mask, cv2.DIST_L2, 5)
    ys, xs = np.where(interior)
    if not len(xs):
        return []
    points = []
    for dx, dy in observation.dots:
        if np.hypot(dx-vx, dy-vy) > vr*.78:
            continue
        px, py = cx+(dx-vx)*r/vr, cy+(dy-vy)*r/vr
        distances = (xs-px)**2 + (ys-py)**2
        index = int(np.argmin(distances))
        if distances[index] > (r*.16)**2:
            continue
        # The nearest pixel can hug an object's edge. After matching a shape,
        # choose a deeper point in that same component, near the original match.
        near_x, near_y = xs[index], ys[index]
        same_shape = labels[ys, xs] == labels[near_y, near_x]
        nearby = (xs-near_x)**2 + (ys-near_y)**2 < (r*.16)**2
        choices = np.flatnonzero(same_shape & nearby)
        scores = clearance[ys[choices], xs[choices]] - np.sqrt(distances[choices])/(r*10)
        best = choices[int(np.argmax(scores))]
        p = [int(xs[best]), int(ys[best])]
        if all(np.linalg.norm(np.array(p)-q) > 12 for q in points):
            points.append(p)
    return points


def annotated(image, observation, config):
    out = image.copy()
    tint = out.copy()
    tint[observation.mask > 0] = (30, 160, 255)
    out = cv2.addWeighted(out, .72, tint, .28, 0)
    for t in observation.targets:
        x, y, w, h = t['box']
        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 180, 255), 1)
    for p in observation.dots:
        cv2.circle(out, tuple(np.rint(p).astype(int)), 5, (0, 255, 255), 1)
    if observation.viewport:
        x, y, r = np.rint(observation.viewport).astype(int)
        cv2.circle(out, (x, y), r, (255, 255, 255), 1)
    points = candidate_points(observation, config)
    for p in points:
        cv2.drawMarker(out, tuple(p), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    text = f'{len(observation.targets)} shapes | {len(observation.dots)} map blobs | {len(points)} proposed clicks'
    cv2.rectangle(out, (10, image.shape[0]-40), (1050, image.shape[0]-4), (20, 20, 20), -1)
    cv2.putText(out, text, (20, image.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
    return out

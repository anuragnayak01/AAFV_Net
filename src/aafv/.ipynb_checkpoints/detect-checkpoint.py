"""E1 decision (2026-11-04): detector chain. Every image records which stage found the face."""

def _area(r):
    return (r.right() - r.left()) * (r.bottom() - r.top())

def detect_chain(rgb, dlib_det, scrfd=None):
    """dlib_det(rgb, upsample) -> list of rects with left/top/right/bottom.
    scrfd(rgb) -> list of (x1, y1, x2, y2, score) or None.
    Returns (box (l,t,r,b) or None, method, n_candidates). Largest face wins; n_candidates > 1 is flagged by the caller."""
    for up, tag in ((1, "dlib_up1"), (2, "dlib_up2")):
        d = list(dlib_det(rgb, up))
        if d:
            r = max(d, key=_area)
            return (r.left(), r.top(), r.right(), r.bottom()), tag, len(d)
    if scrfd is not None:
        f = list(scrfd(rgb))
        if f:
            b = max(f, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
            return tuple(int(v) for v in b[:4]), "scrfd_box", len(f)
    return None, "none", 0

"""FFHQ-style face alignment from 68 landmarks. A port of the alignment function in the official SAM repo
(scripts/align_all_parallel.py, commit c1895aef275e), changed only to take landmarks and a PIL image instead of a file path,
and to use current Pillow constant names. LICENSE FLAG (07 R-25): the SAM file says it is borrowed from NVlabs/ffhq-dataset,
whose code license is, as far as I know, non-commercial; re-implement before any sale."""
import numpy as np
import PIL.Image
import scipy.ndimage

_LANCZOS = getattr(PIL.Image, "Resampling", PIL.Image).LANCZOS
_BILINEAR = getattr(PIL.Image, "Resampling", PIL.Image).BILINEAR
_QUAD = getattr(PIL.Image, "Transform", PIL.Image).QUAD


def align_face(img, lm, output_size=256):
    """img: PIL RGB image; lm: (68, 2) array of landmark x, y. Returns an aligned PIL image of output_size."""
    lm = np.asarray(lm, dtype=np.float64)
    lm_eye_left, lm_eye_right, lm_mouth_outer = lm[36:42], lm[42:48], lm[48:60]
    eye_left, eye_right = np.mean(lm_eye_left, axis=0), np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_avg = (lm_mouth_outer[0] + lm_mouth_outer[6]) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    img = img.convert("RGB")
    transform_size = output_size
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, _LANCZOS)
        quad /= shrink
        qsize /= shrink

    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "reflect")
        h, w, _ = img.shape
        yy, xx, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(xx) / pad[0], np.float32(w - 1 - xx) / pad[2]),
                          1.0 - np.minimum(np.float32(yy) / pad[1], np.float32(h - 1 - yy) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), "RGB")
        quad += pad[:2]

    return img.transform((transform_size, transform_size), _QUAD, (quad + 0.5).flatten(), _BILINEAR)

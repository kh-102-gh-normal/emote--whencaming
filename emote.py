#!/usr/bin/env python3
import argparse
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

POSES = ["time_out", "heart", "cover_nose", "crashing_out", "dance", "nose_closed", "flirty", "hand_up",
         "tongue_out", "open_mouth", "disgusted", "arms_up", "suspicious", "spin", "hold_mic", "ban_tim"]
TEST_KEYS = "1234567890-=[];,"

FACE_SCALE = 2.0
HOLD_FRAMES = 10
ARM = {
    "spin": 15, "suspicious": 8, "arms_up": 5, "dance": 6, "crashing_out": 4,
    "open_mouth": 4, "tongue_out": 5, "disgusted": 5, "hold_mic": 5, "ban_tim": 4,
}

Z = dict(
    jaw_open=6.0,
    scream_jaw=3.5,
    tongue_jaw=3.5,
    sneer=4.5,
    disgust=14.0,
    squint=4.0,
)
Z_CAP = 8.0
FLOOR = dict(
    jaw_open=0.30,
    scream_jaw=0.18,
    tongue_jaw=0.18,
    sneer=0.06,
    squint=0.18,
)
T = dict(
    tongue=0.5,
    head_turn=0.15,
    fh_touch=0.30,
    fh_reach=1.05,
    fh_extended=1.50,
    idx_split=1.15,
    idx_hyst=0.08,
    mic_curl=1.35,
)
FH_MAX_OTHERS_OUT = 1
FACE_MEMORY_S = 1.5
HAND_SMOOTH = 0.55
NEUTRAL_M = {k: 0.0 for k in ("jaw", "z_jaw", "sneer", "z_sneer", "z_brow", "z_frown", "z_lip",
                              "squint", "z_squint", "turn", "z_disgust")}
LENIENT = {"ban_tim", "hold_mic", "flirty", "nose_closed", "cover_nose", "heart", "time_out", "hand_up", "arms_up"}

CALIB_FILE = "calibration.json"
CALIB_SECONDS = 7.0
CALIB_WARMUP = 1.5
CALIB_MIN_SAMPLES = 30
SIGMA_FLOOR = 0.015
SIGMA_CEIL = 0.080
CALIB_VERSION = 1

GENERIC_SIGMA = 0.035
GENERIC_MEAN = {
    "jawOpen": 0.08, "eyeSquintLeft": 0.10, "eyeSquintRight": 0.10,
    "eyeBlinkLeft": 0.10, "eyeBlinkRight": 0.10, "noseSneerLeft": 0.03, "noseSneerRight": 0.03,
    "browDownLeft": 0.06, "browDownRight": 0.06, "mouthFrownLeft": 0.05, "mouthFrownRight": 0.05,
    "mouthUpperUpLeft": 0.05, "mouthUpperUpRight": 0.05,
}

INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

MODELS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}
HERE = os.path.dirname(os.path.abspath(__file__))


def ensure_models():
    mdir = os.path.join(HERE, "models")
    os.makedirs(mdir, exist_ok=True)
    paths = {}
    for name, url in MODELS.items():
        path = os.path.join(mdir, name)
        if not os.path.exists(path):
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, path)
        paths[name] = path
    return paths


def preflight(model_path):
    code = (
        "import sys\n"
        "from mediapipe.tasks import python as t\n"
        "from mediapipe.tasks.python import vision\n"
        "vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(\n"
        "    base_options=t.BaseOptions(model_asset_path=sys.argv[1]),\n"
        "    running_mode=vision.RunningMode.VIDEO, num_faces=1,\n"
        "    output_face_blendshapes=True)).close()\n"
    )
    proc = subprocess.run([sys.executable, "-c", code, model_path], capture_output=True, text=True)
    if proc.returncode == 0:
        return
    err = (proc.stderr or "") + (proc.stdout or "")
    print(f"\nMediaPipe cannot start a detector here (python {platform.python_version()}, "
          f"mediapipe {getattr(mp, '__version__', '?')}, exit {proc.returncode}).\n")
    if "Service is unavailable" in err or "MetalHelper" in err or proc.returncode == -6:
        print("Cause: mediapipe 0.10.30+ ships macOS wheels that abort on startup.\n"
              "Fix (Python 3.11 or 3.12) — install the pinned set:\n"
              "  pip install -r requirements.txt\n"
              "If you already installed something newer by hand, force it back:\n"
              '  pip install "mediapipe==0.10.21" "numpy<2" "opencv-python<5" "opencv-contrib-python<5"\n')
    else:
        print(err[-1500:])
    sys.exit(1)


def build_detectors(model_paths):
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_paths["face_landmarker.task"]),
        running_mode=vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True,
        min_face_detection_confidence=0.4, min_face_presence_confidence=0.4, min_tracking_confidence=0.4))
    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_paths["hand_landmarker.task"]),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4, min_tracking_confidence=0.4))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_paths["pose_landmarker_lite.task"]),
        running_mode=vision.RunningMode.VIDEO, num_poses=1))
    return face, hand, pose


def vcam_help():
    system = platform.system()
    if system == "Darwin":
        return ("macOS uses the OBS virtual camera. One-time setup:\n"
                "  1. Install OBS Studio (https://obsproject.com)\n"
                "  2. Open OBS, click 'Start Virtual Camera', then 'Stop Virtual Camera'\n"
                "  3. Quit OBS completely, then run this script again\n"
                "  If macOS asks you to allow a system / camera extension, approve it in\n"
                "  System Settings > Privacy & Security.")
    if system == "Windows":
        return ("Windows needs a virtual camera. Two options:\n"
                "  A) OBS Studio (https://obsproject.com): install it, and close OBS or stop its\n"
                "     'Virtual Camera' while this script runs (only one program can feed it).\n"
                "  B) Unity Capture, no OBS needed (https://github.com/schellingb/UnityCapture):\n"
                "     download the ZIP, unzip it somewhere permanent, run Install\\Install.bat\n"
                "     (right-click > Run as administrator if it fails), then run this script with\n"
                "       --vcam-backend unitycapture\n"
                "     The camera shows up in Zoom / Meet as 'Unity Video Capture'.")
    return ("Linux needs a v4l2loopback device:\n"
            "  sudo apt install v4l2loopback-dkms\n"
            "  sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=\"its giving\" exclusive_caps=1\n"
            "  (exclusive_caps=1 is what makes Chrome / Meet list the device)")


def open_vcam(W, H, fps, backend, device):
    try:
        import pyvirtualcam
    except ImportError:
        sys.exit("pyvirtualcam is not installed.\n  pip install pyvirtualcam\n"
                 "(or run with --no-vcam for a preview only)")
    kwargs = dict(width=W, height=H, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR)
    if backend:
        kwargs["backend"] = backend
    if device:
        kwargs["device"] = device
    try:
        cam = pyvirtualcam.Camera(**kwargs)
    except Exception as e:
        sys.exit(f"\nCould not start the virtual camera: {e}\n\n{vcam_help()}\n\n"
                 "Run with --no-vcam to test the preview without a virtual camera.")
    print(f"Virtual camera: '{cam.device}' ({getattr(cam, 'backend', '?')}) {W}x{H} @ {fps}fps\n"
          "  -> Zoom: Settings > Video > Camera      Meet: Settings > Video > Camera\n"
          "     pick this camera (it shows black until the first frame arrives)")
    return cam


def list_cameras(limit=5):
    print("Cameras OpenCV can open (use the number with --camera):")
    for i in range(limit):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, f = cap.read()
            print(f"  {i}: {f.shape[1]}x{f.shape[0]}" if ok else f"  {i}: opens, but gives no frames")
        cap.release()
    print("Pick your real webcam. Never pick the virtual camera as the input.")


def to_mp_image(frame, det_width):
    if 0 < det_width < frame.shape[1]:
        h = max(1, int(round(frame.shape[0] * det_width / frame.shape[1])))
        frame = cv2.resize(frame, (det_width, h), interpolation=cv2.INTER_AREA)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def update_arm(arm, raw_pose):
    for p in POSES:
        if raw_pose == p:
            arm[p] = min(arm[p] + 1, ARM.get(p, 3) + 3)
        else:
            arm[p] = max(arm[p] - 1, 0) if p in LENIENT else 0


class HandSmoother:
    def __init__(self, alpha=HAND_SMOOTH, keep=4):
        self.alpha, self.keep, self.prev = alpha, keep, []

    def update(self, hands):
        out, kept, used = [], [], set()
        for p in hands:
            p = np.asarray(p, np.float32)
            best, best_d = None, None
            for i, (q, _) in enumerate(self.prev):
                if i in used:
                    continue
                dd = dist(p[0], q[0])
                if best_d is None or dd < best_d:
                    best, best_d = i, dd
            if best is not None and best_d < 1.5 * max(dist(p[0], p[9]), 1.0):
                used.add(best)
                p = self.alpha * p + (1 - self.alpha) * self.prev[best][0]
            out.append(p)
            kept.append((p, 0))
        for i, (q, age) in enumerate(self.prev):
            if i not in used and age + 1 <= self.keep:
                kept.append((q, age + 1))
        self.prev = kept
        return out


class FaceMemory:
    def __init__(self, keep_s):
        self.keep_s, self.face, self.t = keep_s, None, -1e9

    def update(self, face, has_hands, now):
        if face is not None:
            self.face, self.t = face, now
            return face, "live"
        if has_hands and self.face is not None and now - self.t < self.keep_s:
            return self.face, "remembered"
        return None, "none"


class Clock:
    def __init__(self):
        self.t0, self.last = time.monotonic(), -1

    def next(self):
        self.last = max(int((time.monotonic() - self.t0) * 1000), self.last + 1)
        return self.last


class Baseline:
    def __init__(self, mean=None, sigma=None, samples=0, made=None):
        self.mean = mean or {}
        self.sigma = sigma or {}
        self.samples = samples
        self.made = made
        self.generic = not self.mean

    def z(self, name, value):
        if self.generic:
            return (value - GENERIC_MEAN.get(name, 0.02)) / GENERIC_SIGMA
        m = self.mean.get(name)
        if m is None:
            return (value - GENERIC_MEAN.get(name, 0.02)) / GENERIC_SIGMA
        return (value - m) / self.sigma.get(name, SIGMA_CEIL)

    @property
    def neutral_turn(self):
        return self.mean.get("turn_signed", 0.0) if not self.generic else 0.0

    def save(self, path):
        with open(path, "w") as fh:
            json.dump({"version": CALIB_VERSION, "made": self.made, "samples": self.samples,
                       "mean": self.mean, "sigma": self.sigma}, fh, indent=1, sort_keys=True)

    @staticmethod
    def load(path):
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return Baseline()
        if data.get("version") != CALIB_VERSION or not data.get("mean"):
            return Baseline()
        return Baseline(data["mean"], data.get("sigma", {}), data.get("samples", 0), data.get("made"))


class Collector:
    def __init__(self):
        self.n, self.s, self.ss = 0, {}, {}

    def add(self, face):
        self.n += 1
        for name, v in list(face.bs.items()) + [("turn_signed", face.turn_signed)]:
            self.s[name] = self.s.get(name, 0.0) + v
            self.ss[name] = self.ss.get(name, 0.0) + v * v

    def finish(self):
        mean, sigma = {}, {}
        for name, total in self.s.items():
            m = total / self.n
            var = max(self.ss[name] / self.n - m * m, 0.0)
            mean[name] = round(m, 5)
            sigma[name] = round(min(max(var ** 0.5, SIGMA_FLOOR), SIGMA_CEIL), 5)
        sigma["turn_signed"] = min(max(sigma.get("turn_signed", 0.02), 0.01), 0.10)
        return Baseline(mean, sigma, self.n, time.strftime("%Y-%m-%d %H:%M"))


def calibration_warnings(base):
    out = []
    if base.mean.get("jawOpen", 0) > 0.30:
        out.append("your mouth looks like it was open — don't talk during calibration")
    if max(base.mean.get("noseSneerLeft", 0), base.mean.get("noseSneerRight", 0)) > 0.15:
        out.append("your nose was scrunched — hold a bored face, not a reaction")
    if max(base.mean.get("browInnerUp", 0), base.mean.get("browOuterUpLeft", 0)) > 0.35:
        out.append("your eyebrows were up — relax them")
    pinned = sum(1 for k, v in base.sigma.items() if v >= SIGMA_CEIL)
    if pinned > 12:
        out.append("you moved a lot — sit still and try again for a tighter baseline")
    return out


def run_calibration(cap, face_det, clock, args, W, H, window, vcam=None):
    print(f"\nCalibrating for {CALIB_SECONDS:.0f}s. Sit how you normally sit, look at the "
          "camera, hold a bored face.\nBlinking is fine. Don't talk, smile or raise your eyebrows.")
    col, start, seen_face = Collector(), time.monotonic(), 0
    while True:
        elapsed = time.monotonic() - start
        if elapsed > CALIB_SECONDS:
            break
        ok, raw = cap.read()
        if not ok:
            break
        if raw.shape[0] != H or raw.shape[1] != W:
            raw = cv2.resize(raw, (W, H))
        if vcam:
            vcam.send(raw)
        frame = raw if args.no_flip else cv2.flip(raw, 1)
        fr = face_det.detect_for_video(to_mp_image(frame, args.det_width), clock.next())
        face = Face(fr.face_landmarks[0], fr.face_blendshapes[0] if fr.face_blendshapes else None, W, H) \
            if fr.face_landmarks else None
        if face is not None:
            seen_face += 1
            if elapsed > CALIB_WARMUP and face.bs:
                col.add(face)
        draw_calibration(frame, elapsed, col.n, face)
        cv2.imshow(window, frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            print("Calibration cancelled.")
            return None
        if vcam:
            vcam.sleep_until_next_frame()

    if col.n < CALIB_MIN_SAMPLES:
        print(f"Calibration failed: only {col.n} usable frames"
              f"{' — your face was never detected' if not seen_face else ''}.\n"
              "  - light your face from the front, sit head-and-shoulders in frame, and try again")
        return None
    base = col.finish()
    print(f"Calibrated on {base.samples} frames. Your neutral face:")
    for name in ("jawOpen", "noseSneerLeft", "browDownLeft", "mouthFrownLeft", "eyeSquintLeft"):
        if name in base.mean:
            print(f"  {name:16s} {base.mean[name]:.3f} ± {base.sigma[name]:.3f}")
    for w in calibration_warnings(base):
        print(f"  ! {w}")
    return base


def draw_calibration(img, elapsed, samples, face):
    H, W = img.shape[:2]
    left = max(0.0, CALIB_SECONDS - elapsed)
    cv2.rectangle(img, (0, 0), (W, 96), (0, 0, 0), -1)
    cv2.putText(img, "CALIBRATING - hold a bored face", (16, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    cv2.putText(img, f"{left:0.1f}s   {samples} frames" + ("" if face is not None else "   NO FACE"),
                (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255) if face is not None else (0, 140, 255), 2)
    done = int(W * min(elapsed / CALIB_SECONDS, 1.0))
    cv2.rectangle(img, (0, 84), (done, 96), (0, 220, 0), -1)
    if face is not None:
        x0, y0, x1, y1 = face.box
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 220, 0), 1)


class Asset:
    def __init__(self, frames, durations):
        self.frames = frames
        self.durations = durations
        self.cum = np.cumsum(durations)
        self.total = int(self.cum[-1])
        h, w = frames[0].shape[:2]
        self.aspect = w / h
        self._cache = {}

    def frame_at(self, ms):
        if len(self.frames) == 1:
            return 0
        return int(np.searchsorted(self.cum, ms % self.total, side="right"))

    def scaled(self, idx, height):
        key = (idx, height)
        if key not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            w = max(1, int(round(height * self.aspect)))
            self._cache[key] = cv2.resize(self.frames[idx], (w, height), interpolation=cv2.INTER_AREA)
        return self._cache[key]


def to_bgra(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def placeholder(label):
    img = np.zeros((300, 300, 4), np.uint8)
    cv2.circle(img, (150, 150), 140, (0, 0, 255, 220), -1)
    cv2.putText(img, label, (12, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255, 255), 2)
    return Asset([img], [100])


def find_asset_file(pose):
    adir = os.path.join(HERE, "assets")
    if not os.path.isdir(adir):
        return None
    exts = (".gif", ".png", ".jpg", ".jpeg")
    for fn in sorted(os.listdir(adir)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in exts and (stem == pose or stem.endswith("_" + pose)):
            return os.path.join(adir, fn)
    return None


def load_asset(pose):
    path = find_asset_file(pose)
    if path is None:
        print(f"  {pose:16s} missing -> placeholder")
        return placeholder(pose)
    frames, durations = [], []
    if path.lower().endswith(".gif"):
        from PIL import Image, ImageSequence
        with Image.open(path) as im:
            for f in ImageSequence.Iterator(im):
                frames.append(cv2.cvtColor(np.array(f.convert("RGBA")), cv2.COLOR_RGBA2BGRA))
                durations.append(max(20, int(f.info.get("duration", 100))))
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            frames, durations = [to_bgra(img)], [100]
    if not frames:
        print(f"  {pose:16s} could not read {os.path.basename(path)} -> placeholder")
        return placeholder(pose)
    print(f"  {pose:16s} {os.path.basename(path)}  ({len(frames)} frame{'s' if len(frames) > 1 else ''})")
    return Asset(frames, durations)


def overlay(frame, sprite, x, y):
    H, W = frame.shape[:2]
    h, w = sprite.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return frame
    s = sprite[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[:, :, 3:4].astype(np.float32) / 255.0
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (a * s[:, :, :3] + (1 - a) * roi).astype(np.uint8)
    return frame


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class Face:
    def __init__(self, lms, blendshapes, W, H):
        p = np.array([[l.x * W, l.y * H] for l in lms], np.float32)
        self.pts = p
        x0, y0 = p.min(0)
        x1, y1 = p.max(0)
        self.box = (int(x0), int(y0), int(x1), int(y1))
        self.w, self.h = float(x1 - x0), float(y1 - y0)
        self.center = ((x0 + x1) / 2, (y0 + y1) / 2)
        self.nose, self.chin, self.top = p[1], p[152], p[10]
        self.mouth = (p[13] + p[14]) / 2
        self.eye_y = float((p[33][1] + p[263][1]) / 2)
        cl, cr = p[234], p[454]
        self.turn_signed = float((self.nose[0] - cl[0]) / max(cr[0] - cl[0], 1e-3) - 0.5)
        self.bs = {c.category_name: c.score for c in (blendshapes or [])}

    def b(self, name):
        return self.bs.get(name, 0.0)


class Hand:
    def __init__(self, p):
        p = np.asarray(p, np.float32)
        self.palm = p[[0, 5, 9, 13, 17]].mean(0)
        self.knuckles = p[[5, 9, 13, 17]].mean(0)
        self.thumb, self.index, self.middle = p[4], p[8], p[12]
        d = p[9] - p[0]
        self.horizontal = abs(d[0]) > 1.5 * abs(d[1])
        self.vertical = abs(d[1]) > 1.5 * abs(d[0])
        ext = [dist(p[0], p[t]) > 1.2 * dist(p[0], p[t - 2]) for t in (8, 12, 16, 20)]
        self.open = sum(ext) >= 3
        L = max(dist(p[0], p[9]), 1e-3)
        reach = lambda tip, knuckle: dist(p[0], p[tip]) / max(dist(p[0], p[knuckle]), 1e-3)
        reaches = [reach(t, k) for t, k in ((8, 5), (12, 9), (16, 13), (20, 17))]
        self.index_reach = float(reaches[0])
        self.curl = float(np.mean(reaches))
        self.others_out = sum(r > T["fh_extended"] for r in reaches[1:])
        self.touch = min(dist(p[a], p[b]) for a in (3, 4) for b in (7, 8)) / L
        self.tip_reach = dist(p[0], (p[4] + p[8]) / 2) / L


class Body:
    def __init__(self, lms, W, H):
        p = np.array([[l.x * W, l.y * H] for l in lms], np.float32)
        self.shoulders, self.elbows, self.wrists = p[[11, 12]], p[[13, 14]], p[[15, 16]]
        vis = [getattr(lms[i], "visibility", 1.0) for i in (11, 12, 13, 14)]
        self.seen = min(vis) > 0.5
        shoulder_y = float(self.shoulders[:, 1].mean())
        self.elbows_up = self.seen and bool((self.elbows[:, 1] < shoulder_y).all())


def index_split(sticky):
    split = T["idx_split"]
    if sticky == "ban_tim":
        return split - T["idx_hyst"]
    if sticky == "hold_mic":
        return split + T["idx_hyst"]
    return split


def fh_report(h, face, sticky=None):
    split = index_split(sticky)
    r = dict(touch=float(h.touch), others_out=int(h.others_out), tip_reach=float(h.tip_reach),
             index_reach=float(h.index_reach), split=split)
    r["touch_ok"] = r["touch"] < T["fh_touch"]
    r["others_ok"] = r["others_out"] <= FH_MAX_OTHERS_OUT
    r["reach_ok"] = r["tip_reach"] >= T["fh_reach"]
    r["index_ok"] = r["index_reach"] >= split
    r["place_ok"] = bool(abs(h.palm[0] - face.nose[0]) < 3.0 * face.w
                         and face.top[1] - 0.8 * face.h < h.palm[1] < face.chin[1] + 2.0 * face.h)
    r["ok"] = all(r[k] for k in ("touch_ok", "others_ok", "reach_ok", "index_ok", "place_ok"))
    return r


def mic_report(h, face, sticky=None):
    split = index_split(sticky)
    k = h.knuckles
    d = dist(k, face.mouth)
    dx = abs(k[0] - face.mouth[0])
    dy = k[1] - face.mouth[1]
    r = dict(curl=float(h.curl), index_reach=float(h.index_reach), split=split, near=float(d / face.w))
    r["curl_ok"] = r["curl"] <= T["mic_curl"]
    r["index_ok"] = r["index_reach"] < split
    r["upright_ok"] = not h.horizontal
    r["place_ok"] = bool(0.1 * face.w < d < 1.0 * face.w and dx < 0.8 * face.w
                         and -0.3 * face.w < dy < 1.1 * face.w)
    r["ok"] = all(r[key] for key in ("curl_ok", "index_ok", "upright_ok", "place_ok"))
    return r


def arms_up(hands, face):
    if len(hands) < 2:
        return False
    a, b = hands[0], hands[1]
    ceiling = face.top[1] - 0.15 * face.h
    return bool(a.palm[1] < ceiling and b.palm[1] < ceiling
                and abs(a.palm[0] - b.palm[0]) > 1.2 * face.w)


def tongue_score(frame, face, hands, jaw_ready):
    if not jaw_ready:
        return 0.0
    if any(dist(h.palm, face.mouth) < 0.7 * face.w for h in hands):
        return 0.0
    poly = face.pts[INNER_LIPS].astype(np.int32)
    x0, y0 = poly.min(0)
    x1, y1 = poly.max(0)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0
    x0, y0 = max(x0, 0), max(y0, 0)
    roi = frame[y0:y1 + 1, x0:x1 + 1]
    if roi.size == 0:
        return 0.0
    mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly - [x0, y0]], 255)
    k = max(3, int(0.15 * (y1 - y0)))
    mask = cv2.erode(mask, np.ones((k, k), np.uint8))
    n = int(np.count_nonzero(mask))
    if n < 40:
        return 0.0
    h, s, v = cv2.split(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))
    pink = ((h < 12) | (h > 160)) & (s > 70) & (v > 110)
    return float(np.count_nonzero(pink & (mask > 0)) / n)


def measure(face, base):
    zpair = lambda n: (base.z(n + "Left", face.b(n + "Left")) + base.z(n + "Right", face.b(n + "Right"))) / 2
    pair = lambda n: (face.b(n + "Left") + face.b(n + "Right")) / 2
    m = {
        "jaw": face.b("jawOpen"), "z_jaw": base.z("jawOpen", face.b("jawOpen")),
        "sneer": pair("noseSneer"), "z_sneer": zpair("noseSneer"),
        "z_brow": zpair("browDown"), "z_frown": zpair("mouthFrown"), "z_lip": zpair("mouthUpperUp"),
        "squint": max(pair("eyeSquint"), pair("eyeBlink")),
        "z_squint": max(zpair("eyeSquint"), zpair("eyeBlink")),
        "turn": abs(face.turn_signed - base.neutral_turn),
    }
    cap = lambda v: min(v, Z_CAP)
    m["z_disgust"] = 2 * cap(m["z_sneer"]) + cap(m["z_brow"]) + cap(m["z_frown"]) + cap(m["z_lip"])
    return m


def over(key, m, zkey, rawkey):
    return m[zkey] >= Z[key] and m[rawkey] >= FLOOR[key]


def decide(face, hands, body, tongue, m, sticky=None):
    d = {"hands": len(hands)}
    if face is None:
        gone = not hands and (body is None or not body.seen)
        return ("spin" if gone else None), d

    fw = face.w
    near = lambda a, b, k: dist(a, b) < k * fw
    elbows_up = bool(body and body.elbows_up)
    d.update(m, tongue=tongue, elbows_up=elbows_up)
    if hands:
        d["fh"] = fh_report(min(hands, key=lambda h: h.touch), face, sticky)
        d["mic"] = mic_report(min(hands, key=lambda h: h.curl), face, sticky)
    screaming = over("scream_jaw", m, "z_jaw", "jaw")

    if len(hands) >= 2:
        a, b = hands[0], hands[1]
        for top, under in ((a, b), (b, a)):
            if top.horizontal and under.vertical and top.palm[1] < under.palm[1] \
                    and near(under.middle, top.palm, 0.6):
                return "time_out", d
        if near(a.index, b.index, 0.3) and near(a.thumb, b.thumb, 0.3) \
                and (a.index[1] + b.index[1]) < (a.thumb[1] + b.thumb[1]):
            return "heart", d
        if near(a.palm, face.mouth, 0.6) and near(b.palm, face.mouth, 0.6):
            return "cover_nose", d
        on_head = lambda h: (h.palm[1] < face.eye_y and abs(h.palm[0] - face.nose[0]) < 1.1 * fw
                             and h.palm[1] > face.top[1] - 0.8 * face.h)
        if on_head(a) and on_head(b) and screaming:
            return "crashing_out", d

    if arms_up(hands, face):
        return "arms_up", d

    near_head = lambda h: abs(h.palm[0] - face.nose[0]) < 1.3 * fw and h.palm[1] < face.eye_y + 0.3 * face.h
    if elbows_up and all(near_head(h) for h in hands):
        return ("crashing_out" if screaming else "dance"), d

    for h in hands:
        if near(h.thumb, face.nose, 0.35) and near(h.index, face.nose, 0.35) and near(h.thumb, h.index, 0.3):
            return "nose_closed", d
        if fh_report(h, face, sticky)["ok"]:
            return "ban_tim", d
        if mic_report(h, face, sticky)["ok"]:
            return "hold_mic", d
        if near(h.index, face.mouth, 0.22) and not near(h.palm, face.mouth, 0.3):
            return "flirty", d
        if h.open and h.palm[1] < face.nose[1] and abs(h.palm[0] - face.nose[0]) > 0.8 * fw:
            return "hand_up", d

    if tongue > T["tongue"]:
        return "tongue_out", d
    if over("jaw_open", m, "z_jaw", "jaw"):
        return "open_mouth", d
    if over("sneer", m, "z_sneer", "sneer") or m["z_disgust"] >= Z["disgust"]:
        return "disgusted", d
    if m["turn"] > T["head_turn"] and over("squint", m, "z_squint", "squint"):
        return "suspicious", d
    return None, d


def draw_hud(img, shown, raw, d, face, hands, body, base):
    if face:
        x0, y0, x1, y1 = face.box
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 1)
    for h in hands:
        cv2.circle(img, (int(h.palm[0]), int(h.palm[1])), 6, (0, 200, 255), -1)
    if body and body.seen:
        for pt in np.vstack([body.shoulders, body.elbows]):
            cv2.circle(img, (int(pt[0]), int(pt[1])), 6, (255, 120, 0), -1)
    g = d.get
    lines = [
        (f"showing: {shown or '-'}   raw: {raw or '-'}   hands: {g('hands', 0)}"
         f"   elbows up: {'Y' if g('elbows_up') else 'n'}   face: {g('face_state', '-')}", (0, 255, 0)),
        (f"jaw {g('jaw', 0):.2f} = {g('z_jaw', 0):+.1f}s/{Z['jaw_open']:.0f}   "
         f"squint {g('squint', 0):.2f} = {g('z_squint', 0):+.1f}s/{Z['squint']:.0f}   "
         f"tongue {g('tongue', 0):.2f}   turn {g('turn', 0):.2f}", (0, 255, 0)),
        (f"disgust {g('z_disgust', 0):+.1f}s/{Z['disgust']:.0f} = 2x sneer {g('z_sneer', 0):+.1f} "
         f"+ brow {g('z_brow', 0):+.1f} + frown {g('z_frown', 0):+.1f} + lip {g('z_lip', 0):+.1f}", (0, 255, 0)),
        (("NOT CALIBRATED - generic baseline, everything is harder to trigger. press 'c'"
          if base.generic else
          f"calibrated {base.made} on {base.samples} frames   (s = sigma above your neutral)"),
         (0, 140, 255) if base.generic else (200, 200, 200)),
        ("keys: q quit  d hud  c recalibrate  1-9 0 - = [ ] ; , test poses", (0, 255, 0)),
    ]
    fh, mic = g("fh"), g("mic")
    extra = []
    if fh:
        extra.append((f"ban_tim: touch {fh['touch']:.2f}/<{T['fh_touch']:.2f}  index {fh['index_reach']:.2f}/>={fh['split']:.2f}  "
                      f"others {fh['others_out']}/{FH_MAX_OTHERS_OUT}  tips {fh['tip_reach']:.2f}/>={T['fh_reach']:.2f}  "
                      f"place {'ok' if fh['place_ok'] else 'NO'}  ->  {'YES' if fh['ok'] else 'no'}",
                      (0, 255, 0) if fh["ok"] else (200, 200, 200)))
    if mic:
        extra.append((f"hold_mic: curl {mic['curl']:.2f}/<={T['mic_curl']:.2f}  index {mic['index_reach']:.2f}/<{mic['split']:.2f}  "
                      f"mouth {mic['near']:.2f}fw  upright {'ok' if mic['upright_ok'] else 'NO'}  "
                      f"place {'ok' if mic['place_ok'] else 'NO'}  ->  {'YES' if mic['ok'] else 'no'}",
                      (0, 255, 0) if mic["ok"] else (200, 200, 200)))
    lines[3:3] = extra
    for i, (t, colour) in enumerate(lines):
        y = 24 + 22 * i
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1)


def main():
    ap = argparse.ArgumentParser(description="Meme reactions on your face, live in Zoom / Meet / Teams.")
    ap.add_argument("--camera", type=int, default=0, help="webcam index (try 1 if 0 is your iPhone)")
    ap.add_argument("--list-cameras", action="store_true", help="show which camera numbers work, then exit")
    ap.add_argument("--calibrate", action="store_true", help="learn your neutral face, save it, and exit")
    ap.add_argument("--no-calibration", action="store_true", help="ignore calibration.json; use the generic baseline")
    ap.add_argument("--no-vcam", action="store_true", help="preview only; don't start the virtual camera")
    ap.add_argument("--vcam-backend", choices=["obs", "unitycapture", "v4l2loopback"],
                    help="force a virtual camera driver (default: first one found)")
    ap.add_argument("--vcam-device", help="virtual camera device, e.g. /dev/video10 on Linux")
    ap.add_argument("--fps", type=int, default=30, help="virtual camera frame rate (default 30)")
    ap.add_argument("--det-width", type=int, default=640,
                    help="face and body detection run on an image this wide, 0 = full size (default 640); "
                         "hands always use the full image, small hands need the detail")
    ap.add_argument("--skip-check", action="store_true", help="skip the MediaPipe startup check")
    ap.add_argument("--size", default="1280x720", help="capture size, e.g. 1280x720 or 640x480 (lower = faster)")
    ap.add_argument("--no-hud", action="store_true",
                    help="start with the debug overlay hidden (clean preview window to share or capture)")
    ap.add_argument("--no-flip", action="store_true",
                    help="don't mirror YOUR preview (the call always gets the true, unmirrored image)")
    args = ap.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    calib_path = os.path.join(HERE, CALIB_FILE)
    base = Baseline() if args.no_calibration else Baseline.load(calib_path)
    if base.generic and not args.calibrate and not args.no_calibration:
        print(f"No usable {CALIB_FILE}. Running on the generic baseline — everything is harder to\n"
              f"trigger than it should be. Run:  python {os.path.basename(__file__)} --calibrate")

    model_paths = ensure_models()
    if not args.skip_check:
        preflight(model_paths["face_landmarker.task"])

    cap = cv2.VideoCapture(args.camera)
    if cap.isOpened() and "x" in args.size:
        w, h = args.size.lower().split("x")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ok, frame = False, None
    if cap.isOpened():
        for _ in range(5):
            ok, frame = cap.read()
            if not ok:
                break
    if not ok:
        cap.release()
        sys.exit(f"Could not read from camera {args.camera}.\n"
                 "  - run with --list-cameras to see which numbers work, then use --camera N\n"
                 "  - never choose the virtual camera as the input\n"
                 "  - macOS: System Settings > Privacy & Security > Camera: allow your terminal app, then re-run\n"
                 "  - close Zoom / Meet / OBS if they are holding the camera")
    H, W = frame.shape[:2]
    W, H = W - W % 4, H - H % 2
    print(f"Camera {args.camera}: {W}x{H}")

    clock = Clock()
    window = "it's giving v3  (q quit, d HUD, c recalibrate, 1-9 0 - = [ ] ; , test)"

    if args.calibrate:
        face_det = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_paths["face_landmarker.task"]),
            running_mode=vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True))
        try:
            new = run_calibration(cap, face_det, clock, args, W, H, window)
        finally:
            face_det.close()
            cap.release()
            cv2.destroyAllWindows()
        if new is None:
            sys.exit(1)
        new.save(calib_path)
        print(f"Saved {CALIB_FILE}. Now run:  python {os.path.basename(__file__)}")
        return

    print("Assets:")
    assets = {pose: load_asset(pose) for pose in POSES}

    mirror = not args.no_flip
    vcam = face_det = hand_det = pose_det = None
    try:
        if not args.no_vcam:
            vcam = open_vcam(W, H, args.fps, args.vcam_backend, args.vcam_device)
        face_det, hand_det, pose_det = build_detectors(model_paths)
        shown, hold, show_hud = None, 0, not args.no_hud
        arm = {p: 0 for p in POSES}
        hand_smoother, face_memory = HandSmoother(), FaceMemory(FACE_MEMORY_S)
        shown_since = 0.0
        forced, forced_until = None, 0.0
        sm_center, sm_h = np.array([W / 2, H / 2], np.float32), H * 0.45
        print("Running. Focus the preview window: q quit, d HUD, c recalibrate, 1-9 0 - = [ ] ; , test a pose")

        while True:
            ok, raw = cap.read()
            if not ok:
                print("Camera stopped returning frames.")
                break
            if raw.shape[0] != H or raw.shape[1] != W:
                raw = cv2.resize(raw, (W, H))
            det = cv2.flip(raw, 1) if mirror else raw

            now = time.monotonic()
            ts = clock.next()
            img_full = to_mp_image(det, 0)
            img_small = to_mp_image(det, args.det_width) if 0 < args.det_width < W else img_full
            fr = face_det.detect_for_video(img_small, ts)
            hr = hand_det.detect_for_video(img_full, ts)
            pr = pose_det.detect_for_video(img_small, ts)
            face = Face(fr.face_landmarks[0], fr.face_blendshapes[0] if fr.face_blendshapes else None, W, H) \
                if fr.face_landmarks else None
            points = [np.array([[l.x * W, l.y * H] for l in h], np.float32) for h in hr.hand_landmarks]
            hands = [Hand(p) for p in hand_smoother.update(points)]
            body = Body(pr.pose_landmarks[0], W, H) if pr.pose_landmarks else None

            face_for_decide, face_state = face_memory.update(face, bool(hands), now)
            m = measure(face, base) if face is not None else dict(NEUTRAL_M)
            tongue = tongue_score(det, face, hands,
                                  over("tongue_jaw", m, "z_jaw", "jaw")) if face is not None else 0.0
            raw_pose, dbg = decide(face_for_decide, hands, body, tongue, m, shown)
            dbg["face_state"] = face_state

            update_arm(arm, raw_pose)
            fired = None
            for p in POSES:
                if raw_pose == p and arm[p] >= ARM.get(p, 3):
                    fired = p
            if forced and now < forced_until:
                fired = forced
            if fired:
                if fired != shown:
                    shown_since = now
                shown, hold = fired, HOLD_FRAMES
            elif hold > 0:
                hold -= 1
            else:
                shown = None

            if face is not None:
                sm_center = 0.7 * sm_center + 0.3 * np.array(face.center, np.float32)
                sm_h = 0.7 * sm_h + 0.3 * face.h * FACE_SCALE

            if shown:
                asset = assets[shown]
                idx = asset.frame_at(int((now - shown_since) * 1000))
                h = int(min(sm_h, H * 0.98, (W * 0.98) / asset.aspect)) // 8 * 8
                sprite = asset.scaled(idx, max(h, 8))
                sh, sw = sprite.shape[:2]
                y = int(sm_center[1] - sh / 2 - 0.05 * sh)
                if mirror:
                    overlay(det, sprite, int(sm_center[0] - sw / 2), y)
                    overlay(raw, sprite, int(W - sm_center[0] - sw / 2), y)
                else:
                    overlay(raw, sprite, int(sm_center[0] - sw / 2), y)

            if vcam:
                vcam.send(raw)
                vcam.sleep_until_next_frame()

            if show_hud:
                draw_hud(det, shown, raw_pose, dbg, face_for_decide, hands, body, base)
            cv2.imshow(window, det)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                show_hud = not show_hud
            elif key == ord("c"):
                new = run_calibration(cap, face_det, clock, args, W, H, window, vcam)
                if new is not None:
                    base = new
                    base.save(calib_path)
                    print(f"Saved {CALIB_FILE}.")
                shown, hold = None, 0
                arm = {p: 0 for p in POSES}
            elif 0 < key < 256 and chr(key) in TEST_KEYS:
                forced, forced_until = POSES[TEST_KEYS.index(chr(key))], now + 2.0
    finally:
        cap.release()
        for det_ in (face_det, hand_det, pose_det):
            if det_ is not None:
                det_.close()
        if vcam:
            vcam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
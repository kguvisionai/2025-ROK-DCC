#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiMic Visualizer (GUI) — macOS/Windows
- Up to 4 input devices (USB/earphone mic/built-in) mixed
- Prefers 48 kHz, falls back to 44.1 kHz automatically
- Device check screen: Left canvas (drag&drop mic placement), Right device table + VU + controls
- Visualize screen: Left runtime canvas (mics + DOA/position), Right live waveforms
- Start Sound: normal visualization + DOA/position overlay (always)
- Start Drone: local model-based drone-only detection (score >= 0.90) with gating;
               DOA/position overlay only when drone detected
- Clap sync (GCC-PHAT)
"""

import os, sys, time, math, threading
from collections import deque
from typing import Dict, Tuple, List

import numpy as np
import sounddevice as sd
from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

# (Optional) Only needed for "Start Drone"
from transformers import pipeline
from scipy.signal import resample_poly


# ---------------- audio helpers ----------------
class Ring:
    """Simple block ring: append blocks, get last N samples."""
    def __init__(self, maxlen_samples: int):
        self.maxlen = int(maxlen_samples)
        self.buf = deque(maxlen=256)       # deque of np.ndarray blocks
        self.total = 0
        self.lock = threading.Lock()

    def push(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        with self.lock:
            self.buf.append(x.copy())
            self.total += len(x)
            # prune if too long
            while self.total > self.maxlen and self.buf:
                head = self.buf[0]
                if self.total - len(head) >= self.maxlen:
                    self.buf.popleft()
                    self.total -= len(head)
                else:
                    need_trim = self.total - self.maxlen
                    self.buf[0] = head[need_trim:]
                    self.total -= need_trim
                    break

    def get(self, n: int) -> np.ndarray:
        n = int(n)
        with self.lock:
            if self.total == 0 or n <= 0:
                return np.zeros(0, dtype=np.float32)
            blocks = list(self.buf)
        if not blocks:
            return np.zeros(0, dtype=np.float32)
        data = np.concatenate(blocks)[-n:]
        if len(data) < n:
            data = np.pad(data, (n - len(data), 0)).astype(np.float32)
        return data.astype(np.float32)

    def clear(self):
        with self.lock:
            self.buf.clear()
            self.total = 0


def gcc_phat(sig: np.ndarray, ref: np.ndarray, fs: int, max_tau=None):
    """GCC-PHAT delay estimate (sec)."""
    n = 1 << ((len(sig) + len(ref) - 1).bit_length())
    SIG = np.fft.rfft(sig, n)
    REF = np.fft.rfft(ref, n)
    R = SIG * np.conj(REF)
    d = np.abs(R); d[d == 0] = 1e-12
    R /= d
    cc = np.fft.irfft(R, n)
    max_shift = n // 2
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    if max_tau is None:
        shift = np.argmax(np.abs(cc)) - max_shift
    else:
        lim = int(fs * max_tau)
        center = max_shift
        window = cc[center - lim : center + lim + 1]
        shift = np.argmax(np.abs(window)) - lim
    tau = shift / float(fs)
    return tau, cc


# ---------------- placement canvas (device check) ----------------
class MicDot(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, dev_id: int, label: str, r=14, color="#3aa3ff"):
        super().__init__(-r, -r, 2*r, 2*r)
        self.setBrush(QtGui.QBrush(QtGui.QColor(color)))
        self.setPen(QtGui.QPen(QtGui.QColor("#0b6bd6"), 2))
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)
        self.dev_id = dev_id
        self.text = QtWidgets.QGraphicsSimpleTextItem(label, self)
        self.text.setBrush(QtGui.QBrush(QtGui.QColor("#102a43")))
        self.text.setPos(-r, -r-18)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionChange and self.scene():
            rect = self.scene().sceneRect()
            r = self.rect().width() * 0.5
            pos = value
            x = max(rect.left()+r, min(rect.right()-r, pos.x()))
            y = max(rect.top()+r,  min(rect.bottom()-r, pos.y()))
            return QtCore.QPointF(x, y)
        return super().itemChange(change, value)


class MicCanvas(QtWidgets.QGraphicsView):
    """
    Drag&drop mic placement. get_layout() returns dev_id -> (x_norm, y_norm).
    """
    def __init__(self, parent=None, title_text=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.border = self.scene().addRect(0, 0, 640, 480,
                                           QtGui.QPen(QtGui.QColor("#9aa5b1"), 2))
        self._dots: Dict[int, MicDot] = {}
        txt = title_text or "Drag the mics to place them.\n좌표는 나중에 DOA/거리 추정에 사용됩니다."
        self.hint = self.scene().addText(txt)
        self.hint.setDefaultTextColor(QtGui.QColor("#6b7c93"))
        self.hint.setPos(10, 10)
        self.setMinimumSize(380, 300)
        self.setBackgroundBrush(QtGui.QColor("#f7f7fa"))
        self._sync_scene_rect()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._sync_scene_rect()

    def _sync_scene_rect(self):
        w = max(360, self.viewport().width() - 20)
        h = max(260, self.viewport().height() - 20)
        self.scene().setSceneRect(0, 0, w, h)
        self.border.setRect(0, 0, w, h)

    def set_mics(self, dev_items: List[tuple[int, str]]):
        # clear
        for d in self._dots.values():
            self.scene().removeItem(d)
        self._dots.clear()
        rect = self.scene().sceneRect()
        cx, cy = rect.width()/2, rect.height()/2
        offsets = [(-80,-60), (80,-60), (-80,60), (80,60)]
        for i, (dev_id, label) in enumerate(dev_items):
            dot = MicDot(dev_id, label)
            dx, dy = offsets[i % len(offsets)]
            dot.setPos(cx+dx, cy+dy)
            self.scene().addItem(dot)
            self._dots[dev_id] = dot

    def set_mics_fixed(self, layout: Dict[int, Tuple[float, float]]):
        """Place fixed (non-movable) mic dots from normalized layout."""
        # clear
        for d in self._dots.values():
            self.scene().removeItem(d)
        self._dots.clear()
        rect = self.scene().sceneRect()
        for i, (dev_id, (xn, yn)) in enumerate(layout.items()):
            x = xn * rect.width()
            y = yn * rect.height()
            dot = MicDot(dev_id, f"#{dev_id}", r=12, color="#6ee7b7")  # green-ish for fixed
            dot.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)
            dot.setPos(x, y)
            self.scene().addItem(dot)
            self._dots[dev_id] = dot

    def get_layout(self) -> Dict[int, Tuple[float, float]]:
        rect = self.scene().sceneRect()
        out: Dict[int, Tuple[float,float]] = {}
        for dev_id, dot in self._dots.items():
            p = dot.pos()
            x = float(p.x() / max(1.0, rect.width()))
            y = float(p.y() / max(1.0, rect.height()))
            out[dev_id] = (min(max(x,0.0),1.0), min(max(y,0.0),1.0))
        return out

    # ----- runtime overlay helpers -----
    def clear_overlay(self):
        # remove all items except border, hint, and mic dots
        keep = set(self._dots.values()) | {self.border, self.hint}
        for item in list(self.scene().items()):
            if item not in keep:
                # remove arrows/markers/texts but keep core
                if item not in self._dots.values() and item not in {self.border, self.hint}:
                    self.scene().removeItem(item)

    def draw_estimate(self, x_norm: float, y_norm: float, text: str = ""):
        """Draw estimated source point (red) + arrow from center to it."""
        rect = self.scene().sceneRect()
        x = float(np.clip(x_norm, 0, 1)) * rect.width()
        y = float(np.clip(y_norm, 0, 1)) * rect.height()

        # center (mic centroid)
        if self._dots:
            pts = np.array([[d.pos().x(), d.pos().y()] for d in self._dots.values()])
            cx, cy = float(pts[:,0].mean()), float(pts[:,1].mean())
        else:
            cx, cy = rect.width()/2, rect.height()/2

        # arrow line
        pen = QtGui.QPen(QtGui.QColor("#ef4444"), 3)
        self.scene().addLine(cx, cy, x, y, pen)

        # head
        head = QtWidgets.QGraphicsEllipseItem(-6, -6, 12, 12)
        head.setBrush(QtGui.QBrush(QtGui.QColor("#ef4444")))
        head.setPen(QtGui.QPen(QtGui.QColor("#991b1b"), 1))
        head.setPos(x, y)
        self.scene().addItem(head)

        if text:
            t = self.scene().addText(text)
            t.setDefaultTextColor(QtGui.QColor("#991b1b"))
            t.setPos(x+8, y+8)


# ---------------- audio engine ----------------
class AudioEngine(QtCore.QObject):
    status = QtCore.Signal(str)
    vu = QtCore.Signal(list)

    def __init__(self, max_devices=4, fs_pref=48000, fs_fallback=44100, blocksize=512, live_seconds=8):
        super().__init__()
        self.max_devices = max_devices
        self.fs_pref = fs_pref
        self.fs_fallback = fs_fallback
        self.blocksize = blocksize
        self.live_seconds = live_seconds

        self.fs = fs_pref
        self.device_ids: List[int] = []
        self.streams: List[sd.InputStream] = []
        self.rings: List[Ring] = []
        self.qerrs: List[list] = []
        self.sync_delays_samples: List[int] | None = None
        self.resample_ratios: List[float] | None = None

        # ← 저장되는 마이크 배치 (정규화 좌표)
        self.mic_positions: Dict[int, Tuple[float, float]] = {}

        self._vu_timer = QtCore.QTimer()
        self._vu_timer.timeout.connect(self._emit_vu)

    def list_input_devices(self):
        devs = sd.query_devices()
        items = []
        for i, d in enumerate(devs):
            if d["max_input_channels"] >= 1:
                items.append((i, d["name"], int(d.get("default_samplerate") or 0)))
        return items

    def _callback_factory(self, idx):
        def cb(indata, frames, tinfo, status):
            if status:
                self.qerrs[idx].append((time.time(), str(status)))
            self.rings[idx].push(indata[:, 0])
        return cb

    def start_streams(self, device_ids: List[int]):
        self.stop_streams()
        self.device_ids = device_ids[: self.max_devices]
        if not self.device_ids:
            self.status.emit("No input devices selected.")
            return False

        ok = False
        for attempt_fs in [self.fs_pref, self.fs_fallback]:
            try:
                self.fs = attempt_fs
                self.streams, self.rings, self.qerrs = [], [], []
                needed_blocks = int(self.live_seconds * self.fs / self.blocksize) + 10
                for i, dev in enumerate(self.device_ids):
                    info = sd.query_devices(dev)
                    if info["max_input_channels"] < 1:
                        raise RuntimeError(f"Device {dev} has no input")
                    self.rings.append(Ring(maxlen_samples=needed_blocks * self.blocksize))
                    self.qerrs.append([])
                    st = sd.InputStream(
                        device=dev,
                        channels=1,
                        samplerate=self.fs,
                        blocksize=self.blocksize,
                        dtype="float32",
                        callback=self._callback_factory(i),
                    )
                    self.streams.append(st)
                for st in self.streams:
                    st.start()
                ok = True
                break
            except Exception as e:
                self.status.emit(f"Start failed @ {attempt_fs} Hz: {e}")
                for st in self.streams:
                    try:
                        st.stop(); st.close()
                    except Exception:
                        pass
                self.streams = []
                continue

        if not ok:
            self.status.emit("Failed to start streams at both 48k and 44.1k.")
            return False

        self.sync_delays_samples = [0 for _ in self.device_ids]
        self.resample_ratios = [1.0 for _ in self.device_ids]
        self._vu_timer.start(200)  # 5 Hz
        self.status.emit(f"Started {len(self.streams)} streams @ {self.fs} Hz (block {self.blocksize})")
        return True

    def stop_streams(self):
        self._vu_timer.stop()
        for st in self.streams:
            try:
                st.stop(); st.close()
            except Exception:
                pass
        self.streams, self.rings = [], []
        self.status.emit("Streams stopped.")

    def _emit_vu(self):
        if not self.rings:
            return
        vals = []
        n = int(0.25 * self.fs)
        for r in self.rings:
            x = r.get(n)
            if x.size == 0:
                vals.append(0.0); continue
            vals.append(float(np.sqrt(np.mean(x**2))))
        self.vu.emit(vals)

    def calibrate_clap(self, window_sec=1.0):
        if not self.rings:
            self.status.emit("Streams not running.")
            return
        n = int(window_sec * self.fs)
        ref = self.rings[0].get(n)
        if ref.size < n // 3:
            self.status.emit("Not enough data. Make a loud clap and retry.")
            return
        delays = [0]
        for k in range(1, len(self.rings)):
            x = self.rings[k].get(n)
            m = min(len(ref), len(x))
            if m < n // 3:
                delays.append(0); continue
            tau, _ = gcc_phat(x[-m:], ref[-m:], self.fs, max_tau=0.02)
            delays.append(int(round(tau * self.fs)))
        self.sync_delays_samples = delays
        self.status.emit(f"Sync offsets (samples): {delays}")

    def get_aligned_window(self, display_sec=0.5):
        if not self.rings:
            return []
        nwin = int(display_sec * self.fs)
        out = []
        for i, r in enumerate(self.rings):
            x = r.get(nwin)
            if self.sync_delays_samples is not None and i < len(self.sync_delays_samples):
                off = self.sync_delays_samples[i]
                if off > 0:  x = np.pad(x, (off, 0))[:len(x)]
                elif off < 0: x = np.pad(x, (0, -off))[-len(x):]
            out.append(x[-nwin:] if len(x) >= nwin else np.pad(x, (nwin - len(x), 0)))
        return out


# ---------------- drone detector ----------------
class DroneDetector(QtCore.QObject):
    """
    Local-only drone detection with gating.
    Emits detection(is_drone, score, label) where is_drone becomes True only when
    label == 'drone' and score >= thresh for 'consecutive_hits' frames.
    """
    detection = QtCore.Signal(bool, float, str)
    status = QtCore.Signal(str)

    def __init__(self, engine: AudioEngine, model_dir: str,
                 thresh=0.5, window_sec=0.8, hop_sec=0.2,
                 min_rms=0.005, consecutive_hits=2, debug=False):
        super().__init__()
        self.engine = engine
        self.model_dir = model_dir
        self.thresh = float(thresh)
        self.window_sec = float(window_sec)
        self.hop_msec = int(hop_sec * 1000)
        self.min_rms = float(min_rms)
        self.need_hits = int(consecutive_hits)
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._step)
        self._pipe = None
        self._running = False
        self._last_emit = (False, 0.0, "")
        self._hits = 0
        self._target_sr = 16000
        self._debug = bool(debug)

    def start(self):
        if self._running:
            return
        self._running = True
        if self._pipe is None:
            try:
                if not os.path.isdir(self.model_dir):
                    raise RuntimeError(f"Model folder not found: {self.model_dir}")
                self.status.emit(f"Loading drone model from: {self.model_dir}")
                self._pipe = pipeline(
                    "audio-classification",
                    model=self.model_dir,
                    local_files_only=True,
                )
                fe = getattr(self._pipe, "feature_extractor", None)
                cfg_sr = getattr(getattr(self._pipe, "model", None), "config", None)
                self._target_sr = getattr(fe, "sampling_rate",
                                   getattr(cfg_sr, "sampling_rate", 16000))
                self.status.emit(f"Drone model ready (target_sr={self._target_sr}).")
            except Exception as e:
                self.status.emit(f"Model load failed: {e}")
                self._running = False
                return
        self._timer.start(self.hop_msec)

    def stop(self):
        self._timer.stop()
        self._running = False
        self._last_emit = (False, 0.0, "")
        self._hits = 0

    @QtCore.Slot()
    def _step(self):
        try:
            if not self.engine.rings:
                return
            fs = self.engine.fs
            nwin = int(self.window_sec * fs)
            xlist = self.engine.get_aligned_window(display_sec=self.window_sec)
            if not xlist or xlist[0].size == 0:
                return
            sig = xlist[0][-nwin:].astype(np.float32)

            # quick silence guard
            rms = float(np.sqrt(np.mean(sig**2))) if sig.size else 0.0
            if rms < self.min_rms:
                self._hits = 0
                self._emit(False, 0.0, "silence")
                return

            # resample to model's expected SR
            if fs != self._target_sr:
                up, down = self._target_sr, fs
                g = math.gcd(up, down)
                sig = resample_poly(sig, up//g, down//g).astype(np.float32)

            sig = np.ascontiguousarray(np.clip(sig, -1.0, 1.0), dtype=np.float32)

            results = self._pipe(sig, sampling_rate=self._target_sr)
            best = max(results, key=lambda r: float(r.get("score", 0.0)))
            label = str(best.get("label", "")).strip().lower()
            score = float(best.get("score", 0.0))

            is_drone_now = (label == "drone") and (score >= self.thresh)

            # consecutive gating
            if is_drone_now:
                self._hits = min(self._hits + 1, self.need_hits)
            else:
                self._hits = 0
            is_drone = self._hits >= self.need_hits

            if self._debug:
                self.status.emit(f"[det] rms={rms:.3f} label={label} score={score:.2f} hits={self._hits}/{self.need_hits}")

            self._emit(is_drone, score, best.get("label", ""))

        except Exception as e:
            self.status.emit(f"Detect step error: {e}")

    def _emit(self, is_drone: bool, score: float, label: str):
        cur = (is_drone, score, label)
        if cur != self._last_emit:
            self._last_emit = cur
            self.detection.emit(is_drone, score, label)


# ---------------- simple DOA/position estimator ----------------
class SourceLocalizer(QtCore.QObject):
    """
    Estimate 2D source location (normalized canvas coords 0..1) via coarse grid search
    using pairwise TDOA from GCC-PHAT. Works with 2-4 mics. Assumes unit-speed scale
    (relative geometry) — good for direction & relative position.
    """
    estimated = QtCore.Signal(float, float, float)  # x_norm, y_norm, quality(0..1)
    status = QtCore.Signal(str)

    def __init__(self, engine: AudioEngine, poll_hz=5.0, window_sec=0.4, max_tau=0.02):
        super().__init__()
        self.engine = engine
        self.window_sec = float(window_sec)
        self.max_tau = float(max_tau)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._step)
        self.period_ms = int(1000.0 / max(1e-6, poll_hz))
        self._quality = 0.0

    def start(self):
        self.timer.start(self.period_ms)

    def stop(self):
        self.timer.stop()

    def _pairwise_tdoa(self, sigs: List[np.ndarray], fs: int):
        tdoas = {}  # (i,j) with i<j : tau seconds
        m = min([len(s) for s in sigs]) if sigs else 0
        if m < 8:
            return tdoas
        for i in range(len(sigs)):
            for j in range(i+1, len(sigs)):
                tau, _ = gcc_phat(sigs[i][-m:], sigs[j][-m:], fs, max_tau=self.max_tau)
                tdoas[(i, j)] = tau
        return tdoas

    def _grid_search(self, mic_xy: np.ndarray, tdoas: Dict[tuple,int], fs: int, steps=41):
        # mic_xy: (K,2) normalized coords. Assume speed=1 unit/sec (relative); tdoa in sec.
        # Model: (||s-mi|| - ||s-mj||) = c * tdoa_ij, with c=1 in normalized units.
        if mic_xy.shape[0] < 2 or not tdoas:
            return 0.5, 0.5, 0.0
        xs = np.linspace(0.0, 1.0, steps)
        ys = np.linspace(0.0, 1.0, steps)
        best_err = 1e9
        best_xy = (0.5, 0.5)
        for x in xs:
            for y in ys:
                s = np.array([x, y])
                err = 0.0
                for (i, j), tau in tdoas.items():
                    di = np.linalg.norm(s - mic_xy[i])
                    dj = np.linalg.norm(s - mic_xy[j])
                    # speed scale unknown → set c=1; relative fit
                    err += ( (di - dj) - tau )**2
                if err < best_err:
                    best_err = err
                    best_xy = (x, y)
        # normalize quality: rough heuristic
        q = float(np.exp(-best_err * 5.0))
        return best_xy[0], best_xy[1], q

    @QtCore.Slot()
    def _step(self):
        if not self.engine.rings or not self.engine.device_ids:
            return
        fs = self.engine.fs
        nwin = int(self.window_sec * fs)
        sigs = self.engine.get_aligned_window(display_sec=self.window_sec)
        if not sigs:
            return
        # pick only selected devices that we have placement for
        ids = self.engine.device_ids
        mic_pos_dict = self.engine.mic_positions
        have = [(k, mic_pos_dict.get(k, None)) for k in ids]
        if not all(p is not None for _, p in have):
            # not placed yet
            return
        mic_xy = np.array([mic_pos_dict[k] for k in ids], dtype=float)  # Kx2 in 0..1

        # compute TDOA per pair
        tdoas = self._pairwise_tdoa(sigs, fs)
        if not tdoas:
            return
        x, y, q = self._grid_search(mic_xy, tdoas, fs)
        self._quality = q
        self.estimated.emit(x, y, q)
        # optional debug
        # self.status.emit(f"DOA @ ({x:.2f},{y:.2f}) q={q:.2f}")


# ---------------- GUI: DeviceCheckPage (left canvas + right panel) ----------------
class DeviceCheckPage(QtWidgets.QWidget):
    startSoundRequested = QtCore.Signal(list)      # device_ids
    startDroneRequested = QtCore.Signal(list, str) # device_ids, model_dir

    def __init__(self, engine: AudioEngine):
        super().__init__()
        self.engine = engine

        root_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(root_split)

        # Left: mic placement canvas
        left_box = QtWidgets.QWidget()
        left_v = QtWidgets.QVBoxLayout(left_box)
        left_v.addWidget(QtWidgets.QLabel("Mic placement (drag & drop)"))
        self.canvas = MicCanvas()
        left_v.addWidget(self.canvas, 1)

        # Right: device table + VU + controls
        right_box = QtWidgets.QWidget()
        right_v = QtWidgets.QVBoxLayout(right_box)

        right_v.addWidget(QtWidgets.QLabel("Select up to 4 input devices, test input, then choose Start Sound or Start Drone."))

        self.devModel = QtGui.QStandardItemModel(0, 3)
        self.devModel.setHorizontalHeaderLabels(["Use", "Device (index)", "Default SR"])
        self.devView = QtWidgets.QTableView()
        self.devView.setModel(self.devModel)
        self.devView.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        right_v.addWidget(self.devView, 1)

        # VU area
        self.vuLayout = QtWidgets.QHBoxLayout()
        vu_wrap = QtWidgets.QWidget()
        vu_wrap.setLayout(self.vuLayout)
        right_v.addWidget(vu_wrap)

        # Model folder (for drone)
        mrow = QtWidgets.QHBoxLayout()
        mrow.addWidget(QtWidgets.QLabel("Drone model folder:"))
        self.leModel = QtWidgets.QLineEdit()
        self.leModel.setPlaceholderText("e.g. models/drone")
        btnBrowse = QtWidgets.QPushButton("Browse…")
        mrow.addWidget(self.leModel, 1)
        mrow.addWidget(btnBrowse)
        right_v.addLayout(mrow)

        # Buttons
        brow = QtWidgets.QHBoxLayout()
        self.btnRefresh = QtWidgets.QPushButton("Refresh Devices")
        self.btnStartTest = QtWidgets.QPushButton("Start Test")
        self.btnStartSound = QtWidgets.QPushButton("Start Sound")
        self.btnStartDrone = QtWidgets.QPushButton("Start Drone")
        self.btnStartSound.setEnabled(False)
        self.btnStartDrone.setEnabled(False)
        for b in (self.btnRefresh, self.btnStartTest, self.btnStartSound, self.btnStartDrone):
            brow.addWidget(b)
        right_v.addLayout(brow)

        self.status = QtWidgets.QLabel("")
        right_v.addWidget(self.status)

        root_split.addWidget(left_box)
        root_split.addWidget(right_box)
        root_split.setStretchFactor(0, 1)
        root_split.setStretchFactor(1, 1)

        # Signals
        self.btnRefresh.clicked.connect(self.populate_devices)
        self.btnStartTest.clicked.connect(self.start_test)
        self.btnStartSound.clicked.connect(self.emit_start_sound)
        self.btnStartDrone.clicked.connect(self.emit_start_drone)
        btnBrowse.clicked.connect(self._pick_model_dir)

        self.engine.status.connect(self.status.setText)
        self.engine.vu.connect(self.update_vu)

        self.populate_devices()

    def _pick_model_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Drone Model Folder", os.getcwd())
        if d:
            self.leModel.setText(d)

    def populate_devices(self):
        self.devModel.removeRows(0, self.devModel.rowCount())
        self.clear_vu()
        for idx, name, dsr in self.engine.list_input_devices():
            useItem = QtGui.QStandardItem()
            useItem.setCheckable(True)
            useItem.setCheckState(QtCore.Qt.Unchecked)
            idItem  = QtGui.QStandardItem(f"[{idx}] {name}")
            srItem  = QtGui.QStandardItem(str(dsr))
            self.devModel.appendRow([useItem, idItem, srItem])

    def selected_ids(self):
        ids = []
        for row in range(self.devModel.rowCount()):
            if self.devModel.item(row, 0).checkState() == QtCore.Qt.Checked:
                text = self.devModel.item(row, 1).text()
                idx = int(text.split("]")[0][1:])
                ids.append(idx)
        return ids[:4]

    def clear_vu(self):
        for i in reversed(range(self.vuLayout.count())):
            w = self.vuLayout.itemAt(i).widget()
            if w:
                w.setParent(None)

    def build_vu(self, n):
        self.clear_vu()
        self._vuBars: List[pg.PlotWidget] = []
        for i in range(n):
            w = pg.PlotWidget()
            w.setYRange(0, 1.0)
            w.showGrid(y=True)
            w.setTitle(f"ch{i} RMS")
            bar = pg.BarGraphItem(x=[0.5], height=[0], width=0.8)
            w.addItem(bar)
            w._bar = bar
            self.vuLayout.addWidget(w)
            self._vuBars.append(w)

    def start_test(self):
        ids = self.selected_ids()
        if not ids:
            self.status.setText("Select at least one input.")
            return
        if len(ids) > 4:
            self.status.setText("Pick up to 4 inputs.")
            return
        ok = self.engine.start_streams(ids)
        if ok:
            self.build_vu(len(ids))
            self.btnStartSound.setEnabled(True)
            self.btnStartDrone.setEnabled(True)
            # populate left canvas with selected devices
            labels = []
            for row in range(self.devModel.rowCount()):
                if self.devModel.item(row, 0).checkState() == QtCore.Qt.Checked:
                    text = self.devModel.item(row, 1).text()
                    idx = int(text.split("]")[0][1:])
                    labels.append((idx, f"#{idx}"))
            self.canvas.set_mics(labels)
            self.status.setText(self.status.text() + " | Place mics on the left canvas.")

    def update_vu(self, rms_list):
        if not hasattr(self, "_vuBars"): return
        for i, rms in enumerate(rms_list):
            if i >= len(self._vuBars): break
            h = min(1.0, float(rms) * 10.0)
            self._vuBars[i]._bar.setOpts(height=[h])

    def emit_start_sound(self):
        self.engine.mic_positions = self.canvas.get_layout()
        self.startSoundRequested.emit(self.engine.device_ids)

    def emit_start_drone(self):
        model_dir = self.leModel.text().strip()
        if not model_dir:
            self.status.setText("Set drone model folder first.")
            return
        if not os.path.isdir(model_dir):
            self.status.setText(f"Model folder not found: {model_dir}")
            return
        self.engine.mic_positions = self.canvas.get_layout()
        self.startDroneRequested.emit(self.engine.device_ids, model_dir)


# ---------------- GUI: VisualizePage (left runtime canvas + right waveforms) ----------------
class VisualizePage(QtWidgets.QWidget):
    stopRequested = QtCore.Signal()
    clapRequested = QtCore.Signal()

    def __init__(self, engine: AudioEngine):
        super().__init__()
        self.engine = engine

        # Splitter: left canvas (runtime overlay), right waveforms
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        root = QtWidgets.QHBoxLayout(self)
        root.addWidget(split)

        # Left runtime canvas
        left_wrap = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left_wrap)

        top = QtWidgets.QHBoxLayout()
        self.btnClap = QtWidgets.QPushButton("Calibrate (Clap)")
        self.btnStop = QtWidgets.QPushButton("Stop")
        self.lblMode = QtWidgets.QLabel("Mode: —")
        self.lblDetect = QtWidgets.QLabel("—")
        top.addWidget(self.btnClap); top.addWidget(self.btnStop); top.addStretch(1)
        top.addWidget(self.lblMode); top.addWidget(self.lblDetect)
        lv.addLayout(top)

        self.canvas = MicCanvas(title_text="Runtime mic layout + estimated direction/position")
        lv.addWidget(self.canvas, 1)

        self.status = QtWidgets.QLabel("")
        lv.addWidget(self.status)

        # Right: waveforms
        right_wrap = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right_wrap)
        self.glw = pg.GraphicsLayoutWidget()
        rv.addWidget(self.glw)

        split.addWidget(left_wrap)
        split.addWidget(right_wrap)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)

        self.btnClap.clicked.connect(self.clapRequested.emit)
        self.btnStop.clicked.connect(self.stopRequested.emit)

        self.plots: List[pg.PlotItem] = []
        self.curves: List[pg.PlotDataItem] = []
        self.time_axis = np.arange(1)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.redraw)

        # mode / detector / gate / localizer
        self.mode = "sound"  # or "drone"
        self.detector: DroneDetector | None = None
        self._drone_gate = False
        self.localizer = SourceLocalizer(self.engine, poll_hz=5.0, window_sec=0.4, max_tau=0.02)
        self.localizer.estimated.connect(self._on_estimated)
        self.localizer.status.connect(self.status.setText)

        # latest estimate
        self._est_xy = None   # (x_norm, y_norm, q)

    def start(self, mode="sound", model_dir: str | None = None):
        self.mode = mode
        self.lblMode.setText(f"Mode: {mode}")
        self._drone_gate = (mode == "drone") and False
        self._est_xy = None

        # Left canvas: lock mic dots based on saved positions
        self.canvas.clear_overlay()
        self.canvas.set_mics_fixed(self.engine.mic_positions)

        # Right plots
        self.glw.clear()
        self.plots, self.curves = [], []
        n_ch = len(self.engine.device_ids)
        if n_ch == 0:
            self.status.setText("No input devices running.")
            return

        nwin = int(0.5 * self.engine.fs)
        self.time_axis = np.arange(nwin) / self.engine.fs

        for ch in range(n_ch):
            p = self.glw.addPlot(row=ch, col=0)
            p.showGrid(x=True, y=True)
            p.setLabel('left', f'ch{ch}')
            p.setXRange(0, 0.5)
            p.setYRange(-0.5, 0.5)
            p.showAxis('bottom', ch == n_ch - 1)
            if ch > 0:
                p.setXLink(self.plots[0])
            c = p.plot(self.time_axis, np.zeros_like(self.time_axis))
            self.plots.append(p)
            self.curves.append(c)

        self.timer.start(20)  # ~50 FPS
        self.status.setText(f"fs={self.engine.fs} Hz | channels={n_ch}")

        # start/stop detector
        if self.detector:
            self.detector.stop()
            self.detector = None
        if mode == "drone":
            self.detector = DroneDetector(self.engine, model_dir=model_dir or "", thresh=0.90)
            self.detector.status.connect(self.status.setText)
            self.detector.detection.connect(self.on_detection)
            self.detector.start()
            self.lblDetect.setText("loading…")
        else:
            self.lblDetect.setText("—")
            self._clear_bg()

        # start localizer
        self.localizer.start()

    def _clear_bg(self):
        pal = self.palette()
        self.setPalette(pal)
        self.setAutoFillBackground(False)

    @QtCore.Slot(bool, float, str)
    def on_detection(self, is_drone: bool, score: float, label: str):
        self._drone_gate = bool(is_drone)
        self.lblDetect.setText(f"{label}: {score:.2f}")
        if is_drone:
            pal = self.palette()
            pal.setColor(self.backgroundRole(), QtGui.QColor(255, 230, 230))
            self.setAutoFillBackground(True)
            self.setPalette(pal)
        else:
            self._clear_bg()
            # also hide overlay when no drone
            if self.mode == "drone":
                self.canvas.clear_overlay()
                self.canvas.set_mics_fixed(self.engine.mic_positions)

    @QtCore.Slot(float, float, float)
    def _on_estimated(self, x, y, q):
        self._est_xy = (x, y, q)
        if self.mode == "drone" and not self._drone_gate:
            # gating: do not draw when not drone
            return
        # draw overlay
        self.canvas.clear_overlay()
        self.canvas.set_mics_fixed(self.engine.mic_positions)
        label = f"est ~ ({x:.2f}, {y:.2f})  q={q:.2f}"
        self.canvas.draw_estimate(x, y, label)

    def redraw(self):
        aligned = self.engine.get_aligned_window(display_sec=0.5)
        if not aligned:
            return

        # gating: show only when drone detected (drone mode)
        if self.mode == "drone" and not self._drone_gate:
            for i in range(len(self.curves)):
                self.curves[i].setData(self.time_axis, np.zeros_like(self.time_axis))
                self.plots[i].setYRange(-0.5, 0.5)
            self.status.setText(f"fs={self.engine.fs} Hz | waiting drone…")
            return

        global_peak = 0.1
        for i, y in enumerate(aligned):
            self.curves[i].setData(self.time_axis, y)
            peak = max(0.1, float(np.max(np.abs(y))) * 1.2)
            self.plots[i].setYRange(-peak, peak)
            global_peak = max(global_peak, peak)
        mode_note = f"{self.mode}"
        if self._est_xy is not None and (self.mode == "sound" or self._drone_gate):
            mode_note += f" | DOA q≈{self._est_xy[2]:.2f}"
        self.status.setText(f"fs={self.engine.fs} Hz | peak≈{global_peak:.3f} | {mode_note}")

    def stop_everything(self):
        try:
            self.localizer.stop()
        except Exception:
            pass
        try:
            if self.detector:
                self.detector.stop()
        except Exception:
            pass

# ---------------- GUI: MainWindow ----------------
class MainWindow(QtWidgets.QStackedWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MultiMic Visualizer")

        self.engine = AudioEngine()
        self.pageCheck = DeviceCheckPage(self.engine)
        self.pageViz = VisualizePage(self.engine)

        self.addWidget(self.pageCheck)  # 0
        self.addWidget(self.pageViz)    # 1

        self.pageCheck.startSoundRequested.connect(self.goto_viz_sound)
        self.pageCheck.startDroneRequested.connect(self.goto_viz_drone)
        self.pageViz.stopRequested.connect(self.stop_all)
        self.pageViz.clapRequested.connect(self.do_clap)

        self.engine.status.connect(self.status_message)
        self.resize(1200, 700)

    @QtCore.Slot(str)
    def status_message(self, msg: str):
        print(msg)

    @QtCore.Slot(list)
    def goto_viz_sound(self, device_ids):
        self.setCurrentWidget(self.pageViz)
        self.pageViz.start(mode="sound")

    @QtCore.Slot(list, str)
    def goto_viz_drone(self, device_ids, model_dir: str):
        self.setCurrentWidget(self.pageViz)
        self.pageViz.start(mode="drone", model_dir=model_dir)

    @QtCore.Slot()
    def do_clap(self):
        self.engine.calibrate_clap()

    @QtCore.Slot()
    def stop_all(self):
        try:
            self.pageViz.stop_everything()
        except Exception:
            pass
        self.engine.stop_streams()
        self.setCurrentWidget(self.pageCheck)

    def closeEvent(self, e):
        try:
            self.pageViz.stop_everything()
            self.engine.stop_streams()
        except Exception:
            pass
        return super().closeEvent(e)


# ---------------- main ----------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

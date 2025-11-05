#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiMic Visualizer (GUI) — macOS/Windows (head-only, lines removed)
- Up to 4 input devices (USB/earphone mic/built-in) mixed
- Prefers 48 kHz, falls back to 44.1 kHz automatically
- Device check screen: Left canvas (drag&drop mic placement+orientation), Right device table + VU + controls
- Visualize screen: Left runtime canvas (mics + DOA/position), Right live waveforms
- Start Sound: normal visualization + DOA/position overlay (always)
- Start Drone: local model-based drone-only detection (score >= 0.90) with gating
- Clap sync (GCC-PHAT)
- Mic orientation: Alt + Mouse Wheel to rotate (Right-click to toggle omni/cardioid)
- Canvas grid: 0.5 m spacing. Adjustable scale: (1 unit = X m).
- Visualization: **Only the red head dot** (no lines/labels)
"""

import os, sys, time, math, threading
from collections import deque
from typing import Dict, Tuple, List
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

# (Optional) Only needed for "Start Drone"
from transformers import pipeline
from scipy.signal import resample_poly
from scipy.signal import butter, sosfiltfilt

# ===== New: VU bar colors (4-color palette) =====
VU_COLORS = [
    QtGui.QColor("#1f77b4"),  # blue
    QtGui.QColor("#ff7f0e"),  # orange
    QtGui.QColor("#2ca02c"),  # green
    QtGui.QColor("#d62728"),  # red
]

# ---------------- audio helpers ----------------
class Ring:
    def __init__(self, maxlen_samples: int):
        self.maxlen = int(maxlen_samples)
        self.buf = deque()
        self.total = 0
        self.lock = threading.Lock()

    def push(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        with self.lock:
            self.buf.append(x.copy())
            self.total += len(x)
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

def bandpass(sig: np.ndarray, fs: int, lo=300, hi=3000):
    lo = max(1, lo); hi = min(hi, fs//2 - 100)
    if hi <= lo:
        return sig.astype(np.float32)
    sos = butter(4, [lo, hi], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, sig.astype(np.float32))

def gcc_phat_subsample(sig: np.ndarray, ref: np.ndarray, fs: int,
                       max_tau=None, bp: Tuple[int,int] | None = None):
    if bp is not None:
        lo, hi = bp
        sig = bandpass(sig, fs, lo, hi)
        ref = bandpass(ref, fs, lo, hi)
    n = 1 << ((len(sig) + len(ref) - 1).bit_length())
    SIG = np.fft.rfft(sig, n); REF = np.fft.rfft(ref, n)
    R = SIG * np.conj(REF); d = np.abs(R); d[d == 0] = 1e-12; R /= d
    cc = np.fft.irfft(R, n)
    max_shift = n // 2
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    if max_tau is None:
        k0 = int(np.argmax(np.abs(cc)))
    else:
        lim = int(fs * max_tau)
        center = max_shift
        window = cc[center - lim : center + lim + 1]
        k0 = center - lim + int(np.argmax(np.abs(window)))
    if 1 <= k0 < len(cc) - 1:
        y1, y2, y3 = cc[k0-1], cc[k0], cc[k0+1]
        denom = (y1 - 2*y2 + y3)
        frac = 0.0 if denom == 0 else 0.5 * (y1 - y3) / denom
    else:
        frac = 0.0
    shift = (k0 - max_shift) + float(frac)
    tau = shift / float(fs)
    peak = float(np.max(np.abs(cc)) + 1e-12)
    return tau, cc, peak

# ---------------- Mic graphics ----------------
class MicDot(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, dev_id: int, label: str, r=14, color="#3aa3ff"):
        super().__init__(-r, -r, 2*r, 2*r)
        self.setBrush(QtGui.QBrush(QtGui.QColor(color)))
        self.setPen(QtGui.QPen(QtGui.QColor("#0b6bd6"), 2))
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.dev_id = dev_id
        self.angle_deg = 0.0
        self.pattern = "cardioid"

        self.text = QtWidgets.QGraphicsSimpleTextItem(label, self)
        self.text.setBrush(QtGui.QBrush(QtGui.QColor("#102a43")))
        self.text.setPos(-r, -r-18)

        self._forward = QtWidgets.QGraphicsLineItem(self)
        self._forward.setPen(QtGui.QPen(QtGui.QColor("#0b6bd6"), 2))
        self._update_forward()

        self.setToolTip("Alt + Mouse Wheel = rotate\nRight-click to toggle pattern (omni/cardioid)")

    def _update_forward(self):
        L = self.rect().width() * 0.9
        th = math.radians(self.angle_deg)
        x2 = math.cos(th) * L * 0.6
        y2 = math.sin(th) * L * 0.6
        self._forward.setLine(0, 0, x2, y2)

    def wheelEvent(self, event: QtWidgets.QGraphicsSceneWheelEvent):
        if event.modifiers() & QtCore.Qt.AltModifier:
            delta = 5.0 if event.delta() > 0 else -5.0
            self.angle_deg = (self.angle_deg + delta) % 360.0
            self._update_forward()
            event.accept()
            return
        return super().wheelEvent(event)

    def contextMenuEvent(self, event: QtWidgets.QGraphicsSceneContextMenuEvent):
        menu = QtWidgets.QMenu()
        act1 = menu.addAction("Pattern: omni")
        act2 = menu.addAction("Pattern: cardioid")
        chosen = menu.exec(event.screenPos())
        if chosen == act1:
            self.pattern = "omni"
        elif chosen == act2:
            self.pattern = "cardioid"
        event.accept()

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
    Clock-hand DOA visualization whose hand always starts at the green pivot.
    Pivot is computed as the least-squares intersection of all mic-to-mic lines.
    - If lines are degenerate/parallel, falls back to mic centroid (or scene center).
    - Public API is compatible with previous versions.
    """
    def __init__(self, parent=None, title_text=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHint(QtGui.QPainter.Antialiasing, True)

        self.border = self.scene().addRect(
            0, 0, 640, 480, QtGui.QPen(QtGui.QColor("#9aa5b1"), 2)
        )
        self._dots: Dict[int, MicDot] = {}

        txt = title_text or "Drag mics (Alt+Wheel to rotate). Grid=0.5 m; set scale on the right."
        self.hint = self.scene().addText(txt)
        self.hint.setDefaultTextColor(QtGui.QColor("#6b7c93"))
        self.hint.setPos(10, 10)

        self.setMinimumSize(380, 300)
        self.setBackgroundBrush(QtGui.QColor("#f7f7fa"))
        self._meters_per_unit = 1.0
        self._sync_scene_rect()

        # Pivot (green dot) & clock-hand (red line)
        self._pivot: QtWidgets.QGraphicsEllipseItem | None = None
        self._hand: QtWidgets.QGraphicsLineItem | None = None

        # Recompute pivot periodically (in case mics moved by user)
        self._pivot_timer = QtCore.QTimer(self)
        self._pivot_timer.setInterval(150)
        self._pivot_timer.timeout.connect(self._update_pivot_from_crossing)
        self._pivot_timer.start()

    # ---------- helpers ----------
    def _ensure_pivot(self):
        if self._pivot is None:
            r = 8
            self._pivot = QtWidgets.QGraphicsEllipseItem(-r, -r, 2*r, 2*r)
            self._pivot.setBrush(QtGui.QBrush(QtGui.QColor("#22c55e")))   # green
            self._pivot.setPen(QtGui.QPen(QtGui.QColor("#166534"), 2))
            self._pivot.setZValue(1000)
            self.scene().addItem(self._pivot)

    def _ensure_hand(self):
        if self._hand is None:
            self._hand = QtWidgets.QGraphicsLineItem()
            pen = QtGui.QPen(QtGui.QColor("#ef4444"))   # red
            pen.setWidth(4)
            pen.setCapStyle(QtCore.Qt.RoundCap)
            self._hand.setPen(pen)
            self._hand.setZValue(900)
            self.scene().addItem(self._hand)

    def _move_pivot_to(self, x: float, y: float):
        self._ensure_pivot()
        self._pivot.setPos(float(x), float(y))
        self._pivot.setVisible(True)

    def _mic_points(self) -> List[QtCore.QPointF]:
        return [d.pos() for d in self._dots.values()]

    def _centroid(self) -> Tuple[float, float]:
        pts = self._mic_points()
        rect = self.scene().sceneRect()
        if not pts:
            return rect.width()/2.0, rect.height()/2.0
        xs = [p.x() for p in pts]
        ys = [p.y() for p in pts]
        return float(np.mean(xs)), float(np.mean(ys))

    def _all_pair_lines(self):
        """
        Build lines from every mic pair (i<j).
        Each line represented by (p, v) where p is midpoint, v is unit direction.
        """
        pts = self._mic_points()
        lines = []
        n = len(pts)
        for i in range(n):
            for j in range(i+1, n):
                p1 = np.array([pts[i].x(), pts[i].y()], dtype=float)
                p2 = np.array([pts[j].x(), pts[j].y()], dtype=float)
                v = p2 - p1
                nv = np.linalg.norm(v)
                if nv < 1e-6:
                    continue
                v = v / nv
                pmid = 0.5 * (p1 + p2)
                lines.append((pmid, v))
        return lines

    def _least_squares_intersection(self) -> Tuple[float, float] | None:
        """
        Solve x that minimizes Σ || (I - v v^T) (x - p) ||^2
        A = Σ (I - v v^T),  b = Σ (I - v v^T) p  =>  A x = b
        Return None if ill-conditioned.
        """
        lines = self._all_pair_lines()
        if len(lines) < 2:
            return None

        A = np.zeros((2, 2), dtype=float)
        b = np.zeros((2,), dtype=float)
        I = np.eye(2, dtype=float)

        for (p, v) in lines:
            v = v.reshape(2, 1)
            P = I - (v @ v.T)   # projection onto normal to the line direction
            A += P
            b += (P @ p)

        det = float(np.linalg.det(A))
        if abs(det) < 1e-8:
            return None
        x = np.linalg.solve(A, b)
        return float(x[0]), float(x[1])

    def _update_pivot_from_crossing(self):
        """
        Compute pivot from least-squares crossing of mic pair lines.
        Fallback to centroid if not solvable.
        """
        if not self._dots:
            rect = self.scene().sceneRect()
            self._move_pivot_to(rect.width()/2.0, rect.height()/2.0)
            return

        xy = self._least_squares_intersection()
        if xy is None:
            xy = self._centroid()
        self._move_pivot_to(*xy)

    # Public: expose current pivot (updates first to keep fresh)
    def pivot_xy(self) -> Tuple[float, float]:
        self._update_pivot_from_crossing()
        self._ensure_pivot()
        return float(self._pivot.pos().x()), float(self._pivot.pos().y())

    # ---------- public API (compatible) ----------
    def update_head(self, angle_rad: float, length_ratio: float = 0.35,
                    center: Tuple[float, float] | None = None):
        """
        Draw the clock hand so that it ALWAYS starts at the pivot (green dot).
        - If 'center' is provided, we move the pivot to that location first.
        - Otherwise, the pivot is the least-squares intersection.
        """
        self._ensure_pivot()
        self._ensure_hand()

        rect = self.scene().sceneRect()
        if center is None:
            # ensure pivot is at latest intersection
            self._update_pivot_from_crossing()
            cx, cy = self._pivot.pos().x(), self._pivot.pos().y()
        else:
            cx, cy = float(center[0]), float(center[1])
            self._move_pivot_to(cx, cy)

        # Keep legacy rule: reverse 180° + scale by canvas shorter side
        angle = angle_rad + math.pi
        L = 2.0 * float(length_ratio) * float(min(rect.width(), rect.height()))
        x2 = cx + L * math.cos(angle)
        y2 = cy + L * math.sin(angle)

        # Hand ALWAYS starts at pivot
        self._hand.setLine(cx, cy, x2, y2)
        self._hand.setVisible(True)

    def update_clock_hand(self, angle_rad: float, length_ratio: float = 0.35,
                          text: str = "", center: Tuple[float, float] | None = None):
        self.update_head(angle_rad, length_ratio, center)

    def draw_clock_hand(self, angle_rad: float, length_ratio: float = 0.35,
                        text: str = "", center: Tuple[float, float] | None = None):
        self.update_head(angle_rad, length_ratio, center)

    def set_head_visible(self, visible: bool):
        if self._hand is not None:
            self._hand.setVisible(visible)
        if self._pivot is not None:
            self._pivot.setVisible(visible)

    # ---------- layout / scale / paint ----------
    def set_scale(self, meters_per_unit: float):
        self._meters_per_unit = max(1e-6, float(meters_per_unit))
        self.viewport().update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._sync_scene_rect()
        self.viewport().update()

    def _sync_scene_rect(self):
        w = max(360, self.viewport().width() - 20)
        h = max(260, self.viewport().height() - 20)
        self.scene().setSceneRect(0, 0, w, h)
        self.border.setRect(0, 0, w, h)

    def drawBackground(self, painter: QtGui.QPainter, rect: QtCore.QRectF):
        super().drawBackground(painter, rect)
        step_norm = 0.5 / self._meters_per_unit
        if step_norm <= 0:
            return
        scene_rect = self.sceneRect()
        w, h = scene_rect.width(), scene_rect.height()
        if w <= 0 or h <= 0:
            return
        pen_major = QtGui.QPen(QtGui.QColor(210, 215, 222))
        pen_minor = QtGui.QPen(QtGui.QColor(230, 235, 240))
        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        major_step_norm = 1.0 / self._meters_per_unit
        if major_step_norm < step_norm:
            major_step_norm = step_norm * 2.0

        x = 0.0; idx = 0
        while x <= 1.000001:
            px = x * w
            painter.setPen(pen_major if abs((idx*step_norm) % major_step_norm) < 1e-6 else pen_minor)
            painter.drawLine(QtCore.QPointF(px, 0), QtCore.QPointF(px, h))
            x += step_norm; idx += 1

        y = 0.0; idx = 0
        while y <= 1.000001:
            py = y * h
            painter.setPen(pen_major if abs((idx*step_norm) % major_step_norm) < 1e-6 else pen_minor)
            painter.drawLine(QtCore.QPointF(0, py), QtCore.QPointF(w, py))
            y += step_norm; idx += 1

        painter.setPen(QtGui.QPen(QtGui.QColor("#6b7c93")))
        painter.drawText(6, h - 6,
                         f"grid: 0.5 m  |  scale: 1 unit = {self._meters_per_unit:.2f} m")
        painter.restore()

    # ---------- Mic dots ----------
    def set_mics(self, dev_items: List[tuple]):
        for d in self._dots.values():
            self.scene().removeItem(d)
        self._dots.clear()

        rect = self.scene().sceneRect()
        cx, cy = rect.width()/2, rect.height()/2
        span = 0.22 * min(rect.width(), rect.height())
        cross_offsets = [(-span, 0), (span, 0), (0, -span), (0, span)]

        for i, item in enumerate(dev_items[:4]):
            if len(item) == 3:
                dev_id, label, color = item
                color_hex = color.name() if isinstance(color, QtGui.QColor) else str(color)
            else:
                dev_id, label = item
                color_hex = "#3aa3ff"
            dot = MicDot(dev_id, label, color=color_hex)
            dx, dy = cross_offsets[i] if i < len(cross_offsets) else (0, 0)
            dot.setPos(cx + dx, cy + dy)
            self.scene().addItem(dot)
            self._dots[dev_id] = dot

        self._update_pivot_from_crossing()

    def set_mics_fixed(self, layout: Dict[int, Tuple[float, float, float, str]],
                       labels: Dict[int, str] | None = None,
                       colors: Dict[int, QtGui.QColor] | None = None):
        for d in self._dots.values():
            self.scene().removeItem(d)
        self._dots.clear()

        rect = self.scene().sceneRect()
        for dev_id, (xn, yn, ang, patt) in layout.items():
            x = float(np.clip(xn,0,1)) * rect.width()
            y = float(np.clip(yn,0,1)) * rect.height()
            label = labels.get(dev_id, f"#{dev_id}") if labels else f"#{dev_id}"
            color = colors.get(dev_id, QtGui.QColor("#6ee7b7")) if colors else QtGui.QColor("#6ee7b7")
            dot = MicDot(dev_id, label, r=12, color=color.name())
            dot.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)
            dot.angle_deg = float(ang)
            dot.pattern = str(patt)
            dot.setPos(x, y)
            dot._update_forward()
            self.scene().addItem(dot)
            self._dots[dev_id] = dot

        self._update_pivot_from_crossing()

    def get_layout(self) -> Dict[int, Tuple[float, float, float, str]]:
        rect = self.scene().sceneRect()
        out: Dict[int, Tuple[float, float, float, str]] = {}
        for dev_id, dot in self._dots.items():
            p = dot.pos()
            xn = float(p.x() / max(1.0, rect.width()))
            yn = float(p.y() / max(1.0, rect.height()))
            out[dev_id] = (
                float(np.clip(xn,0,1)),
                float(np.clip(yn,0,1)),
                float(dot.angle_deg),
                str(dot.pattern),
            )
        return out

    def clear_overlay(self):
        """
        Keep only: mic dots, border, hint, green pivot, red hand.
        """
        keep = set(self._dots.values()) | {self.border, self.hint}
        if self._pivot is not None:
            keep.add(self._pivot)
        if self._hand is not None:
            keep.add(self._hand)
        for item in list(self.scene().items()):
            if item not in keep:
                self.scene().removeItem(item)

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
        self.sync_delays_samples: List[int] | None = None

        # mic layout: dev_id -> (x_norm, y_norm, angle_deg, pattern)
        self.mic_positions: Dict[int, Tuple[float, float, float, str]] = {}

        # channel label/color
        self.channel_labels: Dict[int, str] = {}
        self.channel_colors: Dict[int, QtGui.QColor] = {}

        # physical scale: 1 canvas unit == meters_per_unit (default 1 m)
        self.meters_per_unit: float = 1.0

        self._vu_timer = QtCore.QTimer()
        self._vu_timer.timeout.connect(self._emit_vu)

        self._sync_alpha = 0.05

    def _rescan_portaudio(self):
        if self.streams:
            return
        try:
            sd._terminate()
            time.sleep(0.05)
            sd._initialize()
        except Exception as e:
            self.status.emit(f"PortAudio reinit failed: {e}")

    def refine_sync_offsets(self, window_sec=0.5, bp: Tuple[int,int] | None = None, max_tau=0.02):
        if not self.rings or len(self.rings) < 2:
            return
        n = int(window_sec * self.fs)
        ref = self.rings[0].get(n)
        if ref.size < n // 2:
            return
        for k in range(1, len(self.rings)):
            x = self.rings[k].get(n)
            m = min(len(ref), len(x))
            if m < n // 2:
                continue
            tau, _, _ = gcc_phat_subsample(x[-m:], ref[-m:], self.fs, max_tau=max_tau, bp=bp)
            est = int(round(tau * self.fs))
            cur = int(self.sync_delays_samples[k] if self.sync_delays_samples else 0)
            self.sync_delays_samples[k] = int(round((1 - self._sync_alpha) * cur + self._sync_alpha * est))

    def list_input_devices(self, hotplug: bool = False):
        if hotplug:
            self._rescan_portaudio()
        devs = sd.query_devices()
        items = []
        for i, d in enumerate(devs):
            if d["max_input_channels"] >= 1:
                items.append((i, d["name"], int(d.get("default_samplerate") or 0)))
        return items

    def _callback_factory(self, idx):
        def cb(indata, frames, tinfo, status):
            if status:
                pass
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
                self.streams, self.rings = [], []
                needed_blocks = int(self.live_seconds * self.fs / self.blocksize) + 10
                for i, dev in enumerate(self.device_ids):
                    info = sd.query_devices(dev)
                    if info["max_input_channels"] < 1:
                        raise RuntimeError(f"Device {dev} has no input")
                    self.rings.append(Ring(maxlen_samples=needed_blocks * self.blocksize))
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
        self._vu_timer.start(200)  # 5 Hz
        self.status.emit(f"Started {len(self.streams)} streams @ {self.fs} Hz (block {self.blocksize})")
        return True

    # ===== Added: restart at a specific sample rate (e.g., 16 kHz for drone mode) =====
    def restart_streams_with_fs(self, device_ids: List[int], fs_override: int = 16000):
        self.stop_streams()
        self.device_ids = device_ids[: self.max_devices]
        if not self.device_ids:
            self.status.emit("No input devices selected.")
            return False
        try:
            self.fs = int(fs_override)
            self.streams, self.rings = [], []
            needed_blocks = int(self.live_seconds * self.fs / self.blocksize) + 10
            for i, dev in enumerate(self.device_ids):
                info = sd.query_devices(dev)
                if info["max_input_channels"] < 1:
                    raise RuntimeError(f"Device {dev} has no input")
                self.rings.append(Ring(maxlen_samples=needed_blocks * self.blocksize))
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
            self.sync_delays_samples = [0 for _ in self.device_ids]
            self._vu_timer.start(200)
            self.status.emit(f"Restarted {len(self.streams)} streams @ {self.fs} Hz (block {self.blocksize})")
            return True
        except Exception as e:
            self.status.emit(f"Restart @ {fs_override} Hz failed: {e}")
            # Fallback to original logic
            return self.start_streams(self.device_ids)

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
class SimpleDroneDetector(QtCore.QObject):
    """
    Minimal, robust drone detector:
    - Loads local HF pipeline from models/drone (or user path)
    - States: SILENCE / DRONE / NO DRONE
    - Emits: stateChanged(str), scoreChanged(float), status(str)
    - New: requires sufficiently loud signal (RMS >= loud_rms) to declare DRONE
    """
    stateChanged = QtCore.Signal(str)   # "SILENCE" | "DRONE" | "NO DRONE"
    scoreChanged = QtCore.Signal(float) # best score (for display/telemetry)
    status = QtCore.Signal(str)

    def __init__(self, engine: AudioEngine, model_dir: str,
                 # ===== relaxed defaults for real-world drone =====
                 thresh: float = 0.40,
                 window_sec: float = 0.8,
                 hop_sec: float = 0.25,
                 min_rms: float = 1e-6,
                 loud_rms: float = 0.0,
                 debug: bool = False):
        super().__init__()
        self.engine = engine
        self.model_dir = model_dir or "models/drone"
        self.thresh = float(thresh)
        self.window_sec = float(window_sec)
        self.hop_ms = int(hop_sec * 1000)
        self.min_rms = float(min_rms)
        self.loud_rms = float(loud_rms)
        self._drone_keys   = ("drone", "uav", "quadcopter")
        self._silence_keys = ("silence",)
        self._no_keys      = ("no_drone", "no drone", "background", "other")
        self.debug = bool(debug)

        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._step)
        self._pipe = None
        self._target_sr = 16000  # typical; updated from feature_extractor if available
        self._running = False

    def start(self):
        if self._running:
            return
        try:
            if not os.path.isdir(self.model_dir):
                raise RuntimeError(f"Model folder not found: {self.model_dir}")
            self.status.emit(f"Loading drone model: {self.model_dir}")
            self._pipe = pipeline("audio-classification",
                                  model=self.model_dir,
                                  local_files_only=True)
            fe = getattr(self._pipe, "feature_extractor", None)
            cfg_sr = getattr(getattr(self._pipe, "model", None), "config", None)
            self._target_sr = getattr(fe, "sampling_rate",
                                getattr(cfg_sr, "sampling_rate", 16000))
            self.status.emit(f"Model ready (target_sr={self._target_sr})")
            self._running = True
            self._timer.start(self.hop_ms)
        except Exception as e:
            self.status.emit(f"Detector init failed: {e}")
            self._running = False

    def stop(self):
        self._timer.stop()
        self._running = False

    @QtCore.Slot()
    def _step(self):
        try:
            if not self.engine.rings or self._pipe is None:
                return

            fs = self.engine.fs
            nwin = int(self.window_sec * fs)
            chans = self.engine.get_aligned_window(display_sec=self.window_sec)
            if not chans or chans[0].size == 0:
                return

            # single-channel inference (first channel)
            x = chans[0][-nwin:].astype(np.float32)
            rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0

            # 무음 게이트(1차)
            if rms < self.min_rms:
                self._emit_state("SILENCE", score=0.0)
                return

            # 필요 시 리샘플
            if fs != self._target_sr:
                up, down = int(self._target_sr), int(fs)
                g = math.gcd(up, down) or 1
                x = resample_poly(x, up // g, down // g).astype(np.float32)

            x = np.ascontiguousarray(np.clip(x, -1.0, 1.0), dtype=np.float32)

            # 파이프라인 호출 (원 코드 흐름 유지)
            results = self._pipe(x, sampling_rate=self._target_sr)
            if isinstance(results, dict):
                results = [results]

            def _norm(s: str) -> str:
                # 소문자 + 공백/언더스코어/하이픈 제거
                return s.strip().lower().replace("_", "").replace(" ", "").replace("-", "")
            results = self._pipe(x, sampling_rate=self._target_sr)
            if isinstance(results, dict):
                results = [results]

            # label -> score (정규화된 라벨 키로 '정확 일치' 집계)
            probs = {}
            for r in results:
                lab = _norm(str(r.get("label", "")))
                sc  = float(r.get("score", 0.0))
                probs[lab] = max(probs.get(lab, 0.0), sc)

            # 키 집합도 미리 정규화해서 equality로만 매칭
            _keys_drone   = {_norm(k) for k in self._drone_keys}
            _keys_silence = {_norm(k) for k in self._silence_keys}
            _keys_nodrone = {_norm(k) for k in self._no_keys}

            def _max_by_keys(keys_normed: set[str]) -> float:
                return max([probs[k] for k in probs if k in keys_normed] or [0.0])

            p_drone   = _max_by_keys(_keys_drone)
            p_silence = _max_by_keys(_keys_silence)
            p_nodrone = _max_by_keys(_keys_nodrone)

            # 표시용 스코어: 드론 확률만
            self.scoreChanged.emit(p_drone)

            # 무음 게이트(2차): 모델이 silence를 강하게 낼 때
            if p_silence > 0.5:
                self._emit_state("SILENCE", p_drone)
                return

            # 판정 (완화된 조건)
            margin = getattr(self, "margin", 0.10)
            if (rms >= self.loud_rms) and (p_drone >= self.thresh) and (p_drone >= p_nodrone + margin):
                self._emit_state("DRONE", p_drone)
            else:
                self._emit_state("NO DRONE", p_drone)

            if self.debug:
                self.status.emit(f"[det] rms={rms:.3f} p_drone={p_drone:.2f} p_no={p_nodrone:.2f} p_sil={p_silence:.2f}")

        except Exception as e:
            self.status.emit(f"Detector step error: {e}")

    def _emit_state(self, state: str, score: float):
        self.stateChanged.emit(state)
        self.scoreChanged.emit(score)

# ---------------- DOA/position estimator with directivity ----------------
class SourceLocalizer(QtCore.QObject):
    estimated = QtCore.Signal(float, float, float)
    status = QtCore.Signal(str)

    def __init__(self, engine: AudioEngine, poll_hz=5.0, window_sec=0.4, max_tau=0.02,
                 meters_per_unit=1.0, alpha=1.5, level_weight=0.5, cardioid_p=1.0):
        super().__init__()
        self.engine = engine
        self.window_sec = float(window_sec)
        self.max_tau = float(max_tau)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._step)
        self.period_ms = int(1000.0 / max(1e-6, poll_hz))
        self.meters_per_unit = float(meters_per_unit)
        self.alpha = float(alpha)
        self.level_weight = float(level_weight)
        self.cardioid_p = float(cardioid_p)
        self.c = 343.0  # m/s

        self.bp: Tuple[int,int] | None = (300, 3000)  # 일반 사운드 기본
        self._ema_xy: Tuple[float,float] | None = None
        self._ema_beta: float = 0.2

        self.rms_gamma = 1.6
        self.adaptive_level = True
        self.min_coh = 0.20

    def set_scale(self, meters_per_unit: float):
        self.meters_per_unit = max(1e-6, float(meters_per_unit))

    def start(self):
        self.timer.start(self.period_ms)

    def stop(self):
        self.timer.stop()

    def _pairwise_tdoa(self, sigs: List[np.ndarray], fs: int):
        tdoas: Dict[Tuple[int,int], Tuple[float, float]] = {}
        m = min([len(s) for s in sigs]) if sigs else 0
        if m < 8:
            return tdoas
        peaks = []
        for i in range(len(sigs)):
            for j in range(i+1, len(sigs)):
                tau, cc, peak = gcc_phat_subsample(sigs[i][-m:], sigs[j][-m:], fs,
                                                   max_tau=self.max_tau, bp=self.bp)
                tdoas[(i, j)] = (float(tau), float(peak))
                peaks.append(float(peak))
        self._avg_coh = float(np.mean(peaks)) if peaks else 0.0
        return tdoas

    def _directivity_gain(self, mic_dir_deg, mic_xy, src_xy, pattern="cardioid"):
        th = math.radians(mic_dir_deg)
        u = np.array([math.cos(th), math.sin(th)], dtype=float)
        v = (src_xy - mic_xy)
        nv = np.linalg.norm(v) + 1e-12
        v /= nv
        cos_phi = float(np.clip(u @ v, -1.0, 1.0))
        if pattern == "omni":
            return 1.0
        return float((0.5*(1.0 + cos_phi))**self.cardioid_p)

    def _grid_search(self, mic_meta, tdoas, rms_obs, steps=41):
        if len(mic_meta) < 2 or not tdoas:
            return 0.5, 0.5, 0.0

        xs = np.linspace(0.0, 1.0, steps)
        ys = np.linspace(0.0, 1.0, steps)
        best_err, best_xy = 1e12, (0.5, 0.5)

        r = np.array(rms_obs, dtype=float) + 1e-12
        r = r ** self.rms_gamma
        r = r / (r.sum() + 1e-12)
        r_log = np.log(r + 1e-12)
        loud_idx = int(np.argmax(r))

        mic_xy = np.array([[x, y] for (x, y, _, _) in mic_meta], dtype=float)
        tdoa_to_unit = self.c / max(1e-6, self.meters_per_unit)

        w_list = np.array([w for (_, w) in tdoas.values()], dtype=float)
        w_max = float(np.max(w_list)) if w_list.size else 1.0

        for x in xs:
            for y in ys:
                s = np.array([x, y], dtype=float)

                tdoa_err = 0.0
                for (i, j), (tau, w) in tdoas.items():
                    di = np.linalg.norm(s - mic_xy[i])
                    dj = np.linalg.norm(s - mic_xy[j])
                    pred = (di - dj)
                    obs = float(tau) * tdoa_to_unit
                    ww = (w / (w_max + 1e-12))
                    tdoa_err += ww * (pred - obs) ** 2

                g = []
                for (mx, my, deg, patt) in mic_meta:
                    mk = np.array([mx, my], dtype=float)
                    dist = np.linalg.norm(s - mk) + 1e-6
                    D = self._directivity_gain(deg, mk, s, pattern=patt)
                    g.append(D / (dist ** self.alpha))

                g = np.array(g, dtype=float) + 1e-12
                g = g ** self.rms_gamma
                g = g / (g.sum() + 1e-12)
                g_log = np.log(g + 1e-12)

                level_err = float(np.mean(np.abs(g_log - r_log)))

                ord_pen = 0.0
                for k in range(len(g)):
                    if k == loud_idx:
                        continue
                    ord_pen += max(0.0, (g[k] - g[loud_idx] + 1e-3))

                lw = self.level_weight
                if getattr(self, "_avg_coh", 0.0) < self.min_coh and self.adaptive_level:
                    lack = (self.min_coh - float(getattr(self, "_avg_coh", 0.0)))
                    lw *= (1.0 + 2.5 * max(0.0, lack))

                err = tdoa_err + lw * level_err + (0.5 * lw) * ord_pen
                if err < best_err:
                    best_err, best_xy = err, (x, y)

        q = float(np.exp(-best_err * 2.0))
        return best_xy[0], best_xy[1], q

    def _grid_search_window(self, mic_meta, tdoas, rms_obs, cx, cy, half=0.12, steps=41):
        x0, x1 = max(0.0, cx - half), min(1.0, cx + half)
        y0, y1 = max(0.0, cy - half), min(1.0, cy + half)
        xs = np.linspace(x0, x1, steps)
        ys = np.linspace(y0, y1, steps)
        best_err, best_xy = 1e12, (cx, cy)

        r = np.array(rms_obs, dtype=float) + 1e-12
        r = r ** self.rms_gamma
        r = r / (r.sum() + 1e-12)
        r_log = np.log(r + 1e-12)
        loud_idx = int(np.argmax(r))

        mic_xy = np.array([[x, y] for (x, y, _, _) in mic_meta], dtype=float)
        tdoa_to_unit = self.c / max(1e-6, self.meters_per_unit)

        w_list = np.array([w for (_, w) in tdoas.values()], dtype=float)
        w_max = float(np.max(w_list)) if w_list.size else 1.0

        for x in xs:
            for y in ys:
                s = np.array([x, y], dtype=float)

                tdoa_err = 0.0
                for (i, j), (tau, w) in tdoas.items():
                    di = np.linalg.norm(s - mic_xy[i])
                    dj = np.linalg.norm(s - mic_xy[j])
                    pred = (di - dj)
                    obs = float(tau) * tdoa_to_unit
                    ww = (w / (w_max + 1e-12))
                    tdoa_err += ww * (pred - obs) ** 2

                g = []
                for (mx, my, deg, patt) in mic_meta:
                    mk = np.array([mx, my], dtype=float)
                    dist = np.linalg.norm(s - mk) + 1e-6
                    D = self._directivity_gain(deg, mk, s, pattern=patt)
                    g.append(D / (dist ** self.alpha))

                g = np.array(g, dtype=float) + 1e-12
                g = g ** self.rms_gamma
                g = g / (g.sum() + 1e-12)
                g_log = np.log(g + 1e-12)

                level_err = float(np.mean(np.abs(g_log - r_log)))

                ord_pen = 0.0
                for k in range(len(g)):
                    if k == loud_idx:
                        continue
                    ord_pen += max(0.0, (g[k] - g[loud_idx] + 1e-3))

                lw = self.level_weight
                if getattr(self, "_avg_coh", 0.0) < self.min_coh and self.adaptive_level:
                    lack = (self.min_coh - float(getattr(self, "_avg_coh", 0.0)))
                    lw *= (1.0 + 2.5 * max(0.0, lack))

                err = tdoa_err + lw * level_err + (0.5 * lw) * ord_pen
                if err < best_err:
                    best_err, best_xy = err, (x, y)

        q = float(np.exp(-best_err * 2.0))
        return best_xy[0], best_xy[1], q

    @QtCore.Slot()
    def _step(self):
        if not self.engine.rings or not self.engine.device_ids:
            return
        fs = self.engine.fs
        sigs = self.engine.get_aligned_window(display_sec=self.window_sec)
        if not sigs:
            return

        ids = self.engine.device_ids
        meta_d = self.engine.mic_positions
        if not all(k in meta_d for k in ids):
            return
        mic_meta = [meta_d[k] for k in ids]

        # Auto max_tau by array geometry
        mic_xy = np.array([[x,y] for (x,y,_,_) in mic_meta], dtype=float)
        if len(mic_xy) >= 2:
            dmax_units = np.max([np.linalg.norm(mic_xy[i]-mic_xy[j])
                                 for i in range(len(mic_xy)) for j in range(i+1, len(mic_xy))])
            dmax_meters = float(dmax_units) * float(self.meters_per_unit)
            self.max_tau = min(0.1, dmax_meters / self.c)

        tdoas = self._pairwise_tdoa(sigs, fs)
        if not tdoas:
            return
        rms_obs = [float(np.sqrt(np.mean(s.astype(np.float32)**2)) + 1e-12) for s in sigs]

        x0, y0, _ = self._grid_search(mic_meta, tdoas, rms_obs, steps=41)
        x, y, q = self._grid_search_window(mic_meta, tdoas, rms_obs, x0, y0, half=0.12, steps=41)

        if self._ema_xy is None:
            ex, ey = x, y
        else:
            ex = (1 - self._ema_beta) * self._ema_xy[0] + self._ema_beta * x
            ey = (1 - self._ema_beta) * self._ema_xy[1] + self._ema_beta * y
        self._ema_xy = (ex, ey)

        self.estimated.emit(ex, ey, q)

# ---------------- GUI: DeviceCheckPage ----------------
class DeviceCheckPage(QtWidgets.QWidget):
    startSoundRequested = QtCore.Signal()
    startDroneRequested = QtCore.Signal(str)  # model_dir

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

        # Scale (meters per unit)
        scale_row = QtWidgets.QHBoxLayout()
        scale_row.addWidget(QtWidgets.QLabel("Scale (m per unit):"))
        self.spnScale = QtWidgets.QDoubleSpinBox()
        self.spnScale.setRange(0.05, 50.0)
        self.spnScale.setSingleStep(0.05)
        self.spnScale.setDecimals(2)
        self.spnScale.setValue(self.engine.meters_per_unit)
        self.spnScale.setKeyboardTracking(False)
        scale_row.addWidget(self.spnScale, 1)
        right_v.addLayout(scale_row)

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
        self.btnRefresh.clicked.connect(lambda: self.populate_devices(hotplug=True))
        self.btnStartTest.clicked.connect(self.start_test)
        self.btnStartSound.clicked.connect(self.emit_start_sound)
        self.btnStartDrone.clicked.connect(self.emit_start_drone)
        btnBrowse.clicked.connect(self._pick_model_dir)

        self.engine.status.connect(self.status.setText)
        self.engine.vu.connect(self.update_vu)
        self.spnScale.valueChanged.connect(self._on_scale_changed)

        # auto refresh state
        self._dev_snapshot: List[tuple] = []
        self.refreshTimer = QtCore.QTimer(self)
        self.refreshTimer.setInterval(1000)
        self.refreshTimer.timeout.connect(self._auto_refresh)
        self.refreshTimer.start()

        self.populate_devices()

    def showEvent(self, e: QtGui.QShowEvent):
        self.refreshTimer.start()
        return super().showEvent(e)

    def hideEvent(self, e: QtGui.QHideEvent):
        self.refreshTimer.stop()
        return super().hideEvent(e)

    def _on_scale_changed(self, v: float):
        self.engine.meters_per_unit = float(v)
        self.canvas.set_scale(v)

    def _current_checkmap(self) -> Dict[int, bool]:
        checked: Dict[int, bool] = {}
        for row in range(self.devModel.rowCount()):
            text = self.devModel.item(row, 1).text()
            try:
                idx = int(text.split("]")[0][1:])
            except Exception:
                continue
            checked[idx] = self.devModel.item(row, 0).checkState() == QtCore.Qt.Checked
        return checked

    def _auto_refresh(self):
        try:
            devs = self.engine.list_input_devices(hotplug=True)
        except Exception as e:
            self.status.setText(f"Device query failed: {e}")
            return
        if devs != self._dev_snapshot:
            preserve = self._current_checkmap()
            self._dev_snapshot = devs
            self.populate_devices(preserve)
            self.status.setText("Device list updated.")

    def _pick_model_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Drone Model Folder", os.getcwd())
        if d:
            self.leModel.setText(d)

    def populate_devices(self, preserve: Dict[int, bool] | None = None, hotplug: bool = False):
        self.devModel.removeRows(0, self.devModel.rowCount())
        self.clear_vu()
        try:
            devs = self.engine.list_input_devices(hotplug=hotplug)
        except Exception as e:
            self.status.setText(f"Device query failed: {e}")
            return
        self._dev_snapshot = devs
        for idx, name, dsr in devs:
            useItem = QtGui.QStandardItem()
            useItem.setCheckable(True)
            if preserve and preserve.get(idx, False):
                useItem.setCheckState(QtCore.Qt.Checked)
            else:
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
        self._vuBars = []
        for i in range(n):
            w = pg.PlotWidget()
            w.setYRange(0, 1.0)
            w.showGrid(y=True)

            dev = self.engine.device_ids[i]
            w.setTitle(f"ch{i} • [{dev}]")

            bar = pg.BarGraphItem(x=[0.5], height=[0], width=0.8)
            # 색만 지정 (펜/기타 옵션 변경하지 않기)
            color = VU_COLORS[i % len(VU_COLORS)]
            bar.setOpts(brush=color, pen=None)

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
        self.startSoundRequested.emit()

    def emit_start_drone(self):
        model_dir = self.leModel.text().strip()
        # allow empty → default models/drone
        if model_dir and not os.path.isdir(model_dir):
            self.status.setText(f"Model folder not found: {model_dir}")
            return
        self.engine.mic_positions = self.canvas.get_layout()
        self.startDroneRequested.emit(model_dir or "models/drone")

# ---------------- GUI: VisualizePage ----------------
class VisualizePage(QtWidgets.QWidget):
    stopRequested = QtCore.Signal()
    clapRequested = QtCore.Signal()

    def __init__(self, engine: AudioEngine):
        super().__init__()
        self.engine = engine

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        root = QtWidgets.QHBoxLayout(self)
        root.addWidget(split)

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

        # BIG state label
        self.lblStateBig = QtWidgets.QLabel("—")
        big_font = self.lblStateBig.font()
        big_font.setPointSize(24)
        big_font.setBold(True)
        self.lblStateBig.setFont(big_font)
        self.lblStateBig.setAlignment(QtCore.Qt.AlignCenter)
        self.lblStateBig.setStyleSheet("color: #444;")
        lv.addWidget(self.lblStateBig)

        # --- Hotkeys: Shift+0 show / Shift+9 hide ---
        self.setFocusPolicy(QtCore.Qt.StrongFocus)  # 키 입력 받을 수 있게
        self._sc_show = QtGui.QShortcut(QtGui.QKeySequence("Shift+0"), self)
        self._sc_show.activated.connect(self._show_drone_banner)
        self._sc_hide = QtGui.QShortcut(QtGui.QKeySequence("Shift+9"), self)
        self._sc_hide.activated.connect(self._hide_drone_banner)


        self.canvas = MicCanvas(title_text="Runtime mic layout + DOA head (red dot)")
        lv.addWidget(self.canvas, 1)

        self.status = QtWidgets.QLabel("")
        lv.addWidget(self.status)

        right_wrap = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right_wrap)
        self.glw = pg.GraphicsLayoutWidget()

        self.vuWrap = QtWidgets.QWidget()
        self.vuLayout = QtWidgets.QHBoxLayout(self.vuWrap)
        self.vuLayout.setContentsMargins(0, 0, 0, 0)

        self.rightSplit = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.rightSplit.addWidget(self.vuWrap)
        self.rightSplit.addWidget(self.glw)
        self.rightSplit.setStretchFactor(0, 1)
        self.rightSplit.setStretchFactor(1, 1)
        rv.addWidget(self.rightSplit)

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

        self.mode = "sound"
        self.detector: SimpleDroneDetector | None = None
        self._drone_gate = False
        self.localizer = SourceLocalizer(self.engine, poll_hz=5.0, window_sec=0.4, max_tau=0.02,
                                         meters_per_unit=self.engine.meters_per_unit,
                                         alpha=1.5, level_weight=0.5, cardioid_p=1.0)
        self.localizer.estimated.connect(self._on_estimated)
        self.localizer.status.connect(self.status.setText)

        self._est_xy = None

        top.addSpacing(12)
        top.addWidget(QtWidgets.QLabel("Scale (m/unit):"))
        self.spnScaleViz = QtWidgets.QDoubleSpinBox()
        self.spnScaleViz.setRange(0.05, 50.0)
        self.spnScaleViz.setSingleStep(0.05)
        self.spnScaleViz.setDecimals(2)
        self.spnScaleViz.setValue(self.engine.meters_per_unit)
        top.addWidget(self.spnScaleViz)
        self.spnScaleViz.valueChanged.connect(self._on_scale_viz_changed)

        self._syncTimer = QtCore.QTimer(self)
        self._syncTimer.setInterval(1200)
        self._syncTimer.timeout.connect(self._refine_sync_tick)

        self._vuBarsViz = []
        self.engine.vu.connect(self._update_vu_viz)

    def _show_drone_banner(self):
        # 상단 큰 라벨에 '드론탐지' 빨간색으로 표시
        self.lblStateBig.setText("드론탐지")
        self.lblStateBig.setStyleSheet("color:#d40000; font-weight: bold;")

    def _hide_drone_banner(self):
        # 원래 상태로 복귀
        self.lblStateBig.setText("—")
        self.lblStateBig.setStyleSheet("color:#444;")


    def _on_scale_viz_changed(self, v: float):
        self.engine.meters_per_unit = float(v)
        self._sync_scale()

    def _sync_scale(self):
        self.canvas.set_scale(self.engine.meters_per_unit)
        self.localizer.set_scale(self.engine.meters_per_unit)

    def _clear_vu_viz(self):
        for i in reversed(range(self.vuLayout.count())):
            w = self.vuLayout.itemAt(i).widget()
            if w:
                w.setParent(None)
        self._vuBarsViz = []

    def _build_vu_viz(self, n: int):
        self._clear_vu_viz()
        for i in range(n):
            w = pg.PlotWidget()
            w.setYRange(0, 1.0)
            w.showGrid(y=True)

            if i < len(self.engine.device_ids):
                dev = self.engine.device_ids[i]
                w.setTitle(f"ch{i} • [{dev}]")
            else:
                w.setTitle(f"ch{i}")

            bar = pg.BarGraphItem(x=[0.5], height=[0], width=0.8)
            # 색만 지정 (펜은 없앰)
            color = VU_COLORS[i % len(VU_COLORS)]
            bar.setOpts(brush=color, pen=None)

            w.addItem(bar)
            w._bar = bar
            self.vuLayout.addWidget(w)
            self._vuBarsViz.append(w)


    @QtCore.Slot(list)
    def _update_vu_viz(self, rms_list):
        if not self._vuBarsViz:
            return
        for i, rms in enumerate(rms_list):
            if i >= len(self._vuBarsViz): break
            h = min(1.0, float(rms) * 10.0)
            self._vuBarsViz[i]._bar.setOpts(height=[h])

    def _equalize_right_split(self):
        if hasattr(self, "rightSplit") and self.rightSplit is not None:
            size = self.rightSplit.size()
            h = max(2, size.height())
            self.rightSplit.setSizes([h // 2, h // 2])

    def start(self, mode="sound", model_dir: str | None = None):
        self.mode = mode
        self.lblMode.setText(f"Mode: {mode}")
        self._drone_gate = (mode == "drone") and False
        self._est_xy = None

        if hasattr(self, "spnScaleViz"):
            self.spnScaleViz.blockSignals(True)
            self.spnScaleViz.setValue(self.engine.meters_per_unit)
            self.spnScaleViz.blockSignals(False)

        self._sync_scale()

        labels = getattr(self.engine, "channel_labels", {})
        colors = getattr(self.engine, "channel_colors", {})
        self.canvas.clear_overlay()
        self.canvas.set_mics_fixed(self.engine.mic_positions, labels=labels, colors=colors)

        self.glw.clear()
        self.plots, self.curves = [], []
        n_ch = len(self.engine.device_ids)
        if n_ch == 0:
            self.status.setText("No input devices running.")
            return

        self._build_vu_viz(n_ch)

        nwin = int(0.5 * self.engine.fs)
        self.time_axis = np.arange(nwin) / self.engine.fs

        for ch in range(n_ch):
            p = self.glw.addPlot(row=ch, col=0)
            p.showGrid(x=True, y=True)
            dev = self.engine.device_ids[ch]
            label = labels.get(dev, f"ch{ch} • [{dev}]")
            p.setLabel('left', label)
            p.setXRange(0, 0.5)
            p.setYRange(-0.5, 0.5)
            p.showAxis('bottom', ch == n_ch - 1)
            if ch > 0:
                p.setXLink(self.plots[0] if self.plots else None)
            pen = pg.mkPen(colors.get(dev, QtGui.QColor("#3aa3ff")))
            c = p.plot(self.time_axis, np.zeros_like(self.time_axis), pen=pen)
            self.plots.append(p)
            self.curves.append(c)

        self.timer.start(20)
        self.status.setText(f"fs={self.engine.fs} Hz | channels={n_ch} | scale {self.engine.meters_per_unit:.2f} m/unit")

        # reset previous detector
        if self.detector:
            self.detector.stop()
            self.detector = None

        if mode == "drone":
            self.localizer.bp = (80, 600)
            model_dir = model_dir or "models/drone"

            # ===== NEW: force capture at 16 kHz for stable inference =====
            _ok = self.engine.restart_streams_with_fs(self.engine.device_ids, fs_override=16000)
            if not _ok:
                self.status.setText("Failed to restart at 16 kHz; using current fs.")

            # new detector with relaxed thresholds & long window
            self.detector = SimpleDroneDetector(self.engine,
                                                model_dir=model_dir,
                                                thresh=0.40,
                                                window_sec=2.5,
                                                hop_sec=0.25,
                                                min_rms=1e-6,
                                                loud_rms=0.0,
                                                debug=False)
            self.detector.status.connect(self.status.setText)
            self.detector.stateChanged.connect(self._on_detector_state)
            self.detector.scoreChanged.connect(self._on_detector_score)
            self.lblDetect.setText("loading…")
            self.lblStateBig.setText("loading…")
            self.lblStateBig.setStyleSheet("color:#666;")
            self._drone_gate = False
            self.detector.start()
        else:
            self.localizer.bp = (300, 3000)
            self.lblDetect.setText("—")
            self.lblStateBig.setText("—")
            self.lblStateBig.setStyleSheet("color:#444;")
            self._clear_bg()

        self._syncTimer.start()
        self.localizer.start()
        QtCore.QTimer.singleShot(0, self._equalize_right_split)
        self.setFocus()
    def _clear_bg(self):
        pal = self.palette()
        self.setPalette(pal)
        self.setAutoFillBackground(False)

    @QtCore.Slot(str)
    def _on_detector_state(self, state: str):
        # Big label + background + DOA gate
        self.lblStateBig.setText(state)
        if state == "SILENCE":
            self.lblStateBig.setStyleSheet("color:#666;")
            self._drone_gate = False
            self._clear_bg()
            if self.mode == "drone":
                self.canvas.clear_overlay()
                self.canvas.set_mics_fixed(self.engine.mic_positions,
                                           labels=self.engine.channel_labels,
                                           colors=self.engine.channel_colors)
        elif state == "DRONE":
            self.lblStateBig.setStyleSheet("color:#d40000;")
            self._drone_gate = True
            pal = self.palette()
            pal.setColor(self.backgroundRole(), QtGui.QColor(255, 230, 230))
            self.setAutoFillBackground(True)
            self.setPalette(pal)
        else:  # NO DRONE
            self.lblStateBig.setStyleSheet("color:#1565c0;")
            self._drone_gate = False
            self._clear_bg()
            if self.mode == "drone":
                self.canvas.clear_overlay()
                self.canvas.set_mics_fixed(self.engine.mic_positions,
                                           labels=self.engine.channel_labels,
                                           colors=self.engine.channel_colors)

    @QtCore.Slot(float)
    def _on_detector_score(self, score: float):
        self.lblDetect.setText(f"score: {score:.2f}")

    @QtCore.Slot(float, float, float)
    def _on_estimated(self, x, y, q):
        self._est_xy = (x, y, q)
        if self.mode == "drone" and not self._drone_gate:
            return

        self.canvas.clear_overlay()
        self.canvas.set_mics_fixed(self.engine.mic_positions,
                                   labels=self.engine.channel_labels,
                                   colors=self.engine.channel_colors)

        rect = self.canvas.scene().sceneRect()
        px = float(np.clip(x, 0, 1)) * rect.width()
        py = float(np.clip(y, 0, 1)) * rect.height()

        aligned = self.engine.get_aligned_window(display_sec=0.3)
        cx, cy = rect.width() / 2.0, rect.height() / 2.0
        if aligned and self.engine.device_ids:
            rms_list = [float(np.sqrt(np.mean(s.astype(np.float32)**2)) + 1e-12) for s in aligned]
            if len(rms_list) == len(self.engine.device_ids):
                loud_idx = int(np.argmax(rms_list))
                loud_dev = self.engine.device_ids[loud_idx]
                if loud_dev in self.engine.mic_positions:
                    mx, my, _, _ = self.engine.mic_positions[loud_dev]
                    cx = float(np.clip(mx, 0, 1)) * rect.width()
                    cy = float(np.clip(my, 0, 1)) * rect.height()
        else:
            if self.engine.mic_positions and self.engine.device_ids:
                pts = []
                for dev_id in self.engine.device_ids:
                    if dev_id in self.engine.mic_positions:
                        mx, my, _, _ = self.engine.mic_positions[dev_id]
                        pts.append((float(np.clip(mx, 0, 1)) * rect.width(),
                                    float(np.clip(my, 0, 1)) * rect.height()))
                if pts:
                    cx = float(np.mean([p[0] for p in pts]))
                    cy = float(np.mean([p[1] for p in pts]))

        angle = math.atan2(py - cy, px - cx)
        # Hand always from pivot (MicCanvas handles pivot internally)
        self.canvas.draw_clock_hand(angle_rad=angle, length_ratio=0.35, center=None)

    def redraw(self):
        aligned = self.engine.get_aligned_window(display_sec=0.5)
        if not aligned:
            return

        if self.mode == "drone" and not self._drone_gate:
            for i in range(len(self.curves)):
                self.curves[i].setData(self.time_axis, np.zeros_like(self.time_axis))
                self.plots[i].setYRange(-0.5, 0.5)
            self.status.setText(f"fs={self.engine.fs} Hz | waiting drone… | scale {self.engine.meters_per_unit:.2f} m/unit")
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
        self.status.setText(f"fs={self.engine.fs} Hz | peak≈{global_peak:.3f} | {mode_note} | scale {self.engine.meters_per_unit:.2f} m/unit")

    def stop_everything(self):
        try:
            self._syncTimer.stop()
        except Exception:
            pass
        try:
            self.localizer.stop()
        except Exception:
            pass
        try:
            if self.detector:
                self.detector.stop()
        except Exception:
            pass

    def _refine_sync_tick(self):
        bp = self.localizer.bp
        self.engine.refine_sync_offsets(window_sec=0.6, bp=bp, max_tau=self.localizer.max_tau)

# ---------------- GUI: MainWindow ----------------
class MainWindow(QtWidgets.QStackedWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MultiMic Visualizer")

        self.engine = AudioEngine()
        self.pageCheck = DeviceCheckPage(self.engine)
        self.pageViz = VisualizePage(self.engine)

        self.addWidget(self.pageCheck)
        self.addWidget(self.pageViz)

        self.pageCheck.startSoundRequested.connect(self.goto_viz_sound)
        self.pageCheck.startDroneRequested.connect(self.goto_viz_drone)
        self.pageViz.stopRequested.connect(self.stop_all)
        self.pageViz.clapRequested.connect(self.do_clap)

        self.engine.status.connect(self.status_message)
        self.resize(1200, 700)

    @QtCore.Slot(str)
    def status_message(self, msg: str):
        print(msg)

    @QtCore.Slot()
    def goto_viz_sound(self):
        self.setCurrentWidget(self.pageViz)
        self.pageViz.start(mode="sound")

    @QtCore.Slot(str)
    def goto_viz_drone(self, model_dir: str):
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

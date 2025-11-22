#!/usr/bin/env python3
"""
TwiNet-like traffic monitoring experiment and plotting (Figure 3 style)

Two modes:
  1) demo     : run a local "real" sender and a "DT" receiver over UDP loopback,
                schedule six rate changes across 60s, measure delay, plot.
  2) logs     : read your own logs (ns-3 as real world, Mininet-WiFi as DT),
                compute average delay, and plot the same style figure.

CSV LOG FORMAT (for --mode logs):
  Each file should be CSV with headers: time_s,pps
  - time_s : float seconds from start (0 at experiment start)
  - pps    : packets per second (integer or float)

Example:
  time_s,pps
  0.0,200
  10.0,800
  20.0,1500
  30.0,500
  45.0,2000
  55.0,1000

INTEGRATION HINTS
-----------------
ns-3 (real world):
  - Add a trace or printf when you change app SendRate / OnOffApplication data rate:
      std::cout << std::fixed << std::setprecision(6)
                << Simulator::Now().GetSeconds() << "," << pps << std::endl;
  - Pipe this to a file and convert to CSV headers or write CSV directly.

Mininet-WiFi (digital twin):
  - In your Python controller loop (or a small agent), whenever you detect/apply
    the mirrored rate change, log a line with the same format time_s,pps.
  - Use time.monotonic() when the DT “state” (rate) actually updates.

USAGE
-----
Demo (end-to-end delay on localhost, then plot):
  python3 twinet_traffic_monitor.py --mode demo

Logs (read your real CSVs and plot):
  python3 twinet_traffic_monitor.py --mode logs --real real_ns3.csv --dt dt_mininet.csv

Outputs:
  - figure: fig3_like.png
  - console: average delay (ms) per change event
"""

import argparse
import socket
import threading
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- Utilities ----------

@dataclass
class RateEvent:
    t: float     # time (s) when rate becomes active
    pps: float   # packets per second


def step_series(events: List[RateEvent], T: float, dt: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn a set of (time, pps) change points into a step-like time series up to T seconds.
    """
    ts = np.arange(0.0, T + 1e-9, dt)
    vals = np.zeros_like(ts)
    current = 0.0
    e_idx = 0
    events_sorted = sorted(events, key=lambda e: e.t)
    for i, t in enumerate(ts):
        while e_idx < len(events_sorted) and t >= events_sorted[e_idx].t:
            current = events_sorted[e_idx].pps
            e_idx += 1
        vals[i] = current
    return ts, vals


def find_change_points(events: List[RateEvent]) -> List[float]:
    """Return just the change times (excluding t=0 if present twice)."""
    times = sorted({e.t for e in events})
    return times


def align_and_delay(real_events: List[RateEvent], dt_events: List[RateEvent]) -> List[float]:
    """
    For each real change time, find the first DT event at the same pps that occurs after it,
    compute delay (dt_time - real_time). Returns list of delays (s).
    """
    delays = []
    real_map = [(e.t, e.pps) for e in sorted(real_events, key=lambda x: x.t)]
    dt_sorted = sorted(dt_events, key=lambda x: x.t)
    for rt, rpps in real_map:
        # Find first DT event with same pps and t >= rt
        cand = [e.t for e in dt_sorted if e.pps == rpps and e.t >= rt]
        if cand:
            delays.append(cand[0] - rt)
    return delays


def save_plot(real_events: List[RateEvent], dt_events: List[RateEvent],
              T: float, zoom_window: Optional[Tuple[float, float]] = None,
              outfile: str = "fig3_like.png", title_left="60s experiment", title_right="Zoomed delay window"):
    ts_r, ys_r = step_series(real_events, T)
    ts_d, ys_d = step_series(dt_events, T)

    plt.figure(figsize=(10, 4.5))
    gs = plt.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.3)

    # Left: full experiment
    ax1 = plt.subplot(gs[0, 0])
    ax1.step(ts_d, ys_d, where='post', label="Digital Twin Update")
    ax1.step(ts_r, ys_r, where='post', linestyle='--', label="Real-World Change")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Packets per second")
    ax1.set_title(title_left)
    ax1.legend(loc="upper right")

    # Right: zoomed area
    if zoom_window is None:
        # Pick a default: around the 2nd change if it exists
        cps = find_change_points(real_events)
        if len(cps) >= 2:
            t0 = cps[1] - 1.0
            t1 = cps[1] + 1.0
        else:
            t0, t1 = 9.0, 11.0
        zoom_window = (max(0.0, t0), min(T, t1))

    ax2 = plt.subplot(gs[0, 1])
    ax2.step(ts_d, ys_d, where='post', label="DT")
    ax2.step(ts_r, ys_r, where='post', linestyle='--', label="Real")
    ax2.set_xlim(zoom_window[0], zoom_window[1])
    ax2.set_xlabel("Time (s)")
    ax2.set_title(title_right)
    # Keep y label off to reduce clutter
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    print(f"[OK] Saved plot: {outfile}")


# ---------- DEMO MODE (UDP loopback to measure real delay) ----------

class DTDReceiver(threading.Thread):
    """
    Receives 'rate-change' UDP messages and records DT event times.
    Message format: "RATE,<pps>,<send_ts_epoch>"
    """
    def __init__(self, port: int, start_epoch: float, dt_events: List[RateEvent]):
        super().__init__(daemon=True)
        self.port = port
        self.start_epoch = start_epoch
        self.dt_events = dt_events
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.running = True

    def run(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(1024)
                now = time.time()  # epoch
                msg = data.decode().strip()
                if not msg.startswith("RATE"):
                    continue
                _, pps_str, send_epoch_str = msg.split(",")
                pps = float(pps_str)
                # arrival time in experiment seconds:
                t_rel = now - self.start_epoch
                # simulate small processing time if desired:
                # time.sleep(0.005)  # 5ms
                self.dt_events.append(RateEvent(t_rel, pps))
            except Exception:
                continue

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except Exception:
            pass


def run_demo():
    """
    Schedules six rate changes in 60s and sends them via UDP to a DT receiver.
    We record "real" change times (send time) and "DT" update times (receive time).
    """
    T = 60.0
    # (t, pps) like the paper: 6 changes
    schedule = [
        RateEvent(0.0, 500),
        RateEvent(10.0, 1000),
        RateEvent(20.0, 2000),
        RateEvent(30.0, 800),
        RateEvent(45.0, 2500),
        RateEvent(55.0, 1200),
    ]

    # Real and DT event logs
    real_events: List[RateEvent] = []
    dt_events: List[RateEvent] = []

    start_epoch = time.time()
    recv = DTDReceiver(port=9999, start_epoch=start_epoch, dt_events=dt_events)
    recv.start()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print("[demo] Running for 60s with scheduled rate changes...")
    for ev in schedule:
        # Sleep until it's time for the change
        while True:
            now = time.time()
            if now - start_epoch >= ev.t:
                break
            time.sleep(0.01)

        # Record real event time (just before send)
        t_rel = time.time() - start_epoch
        real_events.append(RateEvent(t_rel, ev.pps))

        # Send the UDP message to the DT
        msg = f"RATE,{ev.pps},{time.time():.6f}".encode()
        sock.sendto(msg, ("127.0.0.1", 9999))
        print(f"[demo] t={t_rel:5.3f}s  real->DT: pps={ev.pps}")

    # Let any pending packets arrive
    time.sleep(1.0)
    recv.stop()

    # Compute delays by matching same pps after each real change
    delays = align_and_delay(real_events, dt_events)
    if delays:
        avg_ms = np.mean(delays) * 1000.0
        print(f"[demo] Average DT delay over {len(delays)} changes: {avg_ms:.2f} ms")
    else:
        print("[demo] No matching changes detected to compute delay")

    # Choose a zoom window around the 2nd change
    zoom = (9.0, 11.0)
    save_plot(real_events, dt_events, T, zoom_window=zoom)


# ---------- LOG MODE (use your ns-3 + Mininet-WiFi logs) ----------

def load_csv(path: str) -> List[RateEvent]:
    df = pd.read_csv(path)
    # ensure columns exist
    if "time_s" not in df.columns or "pps" not in df.columns:
        raise ValueError(f"{path} must have headers: time_s,pps")
    events = [RateEvent(float(r["time_s"]), float(r["pps"])) for _, r in df.iterrows()]
    return events


def run_logs(real_csv: str, dt_csv: str, zoom_center: Optional[float] = None):
    real_events = load_csv(real_csv)
    dt_events = load_csv(dt_csv)

    # End time: use max of both
    T = max(max(e.t for e in real_events), max(e.t for e in dt_events))
    # Compute delays
    delays = align_and_delay(real_events, dt_events)
    if delays:
        avg_ms = np.mean(delays) * 1000.0
        print(f"[logs] Average DT delay over {len(delays)} changes: {avg_ms:.2f} ms")
    else:
        print("[logs] No matching changes detected to compute delay")

    if zoom_center is None:
        cps = find_change_points(real_events)
        zoom_center = cps[1] if len(cps) >= 2 else (cps[0] if cps else 10.0)
    zoom = (max(0.0, zoom_center - 1.0), min(T, zoom_center + 1.0))

    save_plot(real_events, dt_events, T, zoom_window=zoom)


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["demo", "logs"], required=True)
    ap.add_argument("--real", help="CSV file from ns-3 (time_s,pps) for --mode logs")
    ap.add_argument("--dt", help="CSV file from Mininet-WiFi (time_s,pps) for --mode logs")
    ap.add_argument("--zoom_center", type=float, default=None, help="Center of zoom window in seconds (logs mode)")
    args = ap.parse_args()

    if args.mode == "demo":
        run_demo()
    else:
        if not args.real or not args.dt:
            raise SystemExit("In logs mode, you must provide --real and --dt CSV files.")
        run_logs(args.real, args.dt, args.zoom_center)


if __name__ == "__main__":
    main()


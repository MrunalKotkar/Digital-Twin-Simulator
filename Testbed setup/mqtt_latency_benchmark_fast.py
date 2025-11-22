#!/usr/bin/env python3
"""
Fast MQTT one-way latency benchmark (Table I style) + plot

- Much faster defaults than the original:
    * Fewer sizes by default
    * 10 samples per size (configurable)
    * Shorter timeouts with fail-fast checks
- Adds --fast (very quick) and --full (paper-like) presets.

Requirements:
  pip install paho-mqtt matplotlib pandas

Run:
  python3 mqtt_latency_benchmark_fast.py --fast
or
  python3 mqtt_latency_benchmark_fast.py --samples 20 --sizes 1,100,1024,10240,102400
"""

import argparse
import time
import struct
import threading
import queue
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import paho.mqtt.client as mqtt
except Exception as e:
    raise SystemExit("paho-mqtt is required: pip install paho-mqtt")


# ----------------------------- Defaults -----------------------------

BROKER_HOST = "127.0.0.1"
BROKER_PORT = 1883

TOPIC_R2T = "twinet/real2twin"
TOPIC_T2R = "twinet/twin2real"

# Smaller default set to speed up
DEFAULT_SIZES = [1, 100, 1024, 10*1024, 100*1024]  # up to 100 KB
FULL_SIZES     = [1, 100, 1024, 10*1024, 100*1024, 1024*1024]  # includes 1 MB

DEFAULT_SAMPLES = 10
FAST_SAMPLES    = 5
FULL_SAMPLES    = 100

QOS = 0
WARMUP = 2  # keep very small to avoid long runs

RX_TIMEOUT_S = 0.5   # per message
CONNECT_TIMEOUT_S = 3.0


# ----------------------------- Utils -----------------------------

def now_ns() -> int:
    return time.monotonic_ns()

def build_payload(payload_size: int, seq: int) -> bytes:
    """
    Payload format:
      [8 bytes: send_time_ns][4 bytes: seq][padding ...] -> total length == payload_size (>= 12)
    """
    send_ns = now_ns()
    header = struct.pack("!QI", send_ns, seq)  # 12 bytes
    if payload_size <= len(header):
        # ensure at least header fits; if smaller requested, trim (still works)
        return header[:payload_size]
    return header + b"\x00" * (payload_size - len(header))

def parse_payload(payload: bytes):
    if len(payload) < 12:
        return None, None
    send_ns, seq = struct.unpack("!QI", payload[:12])
    return send_ns, seq


# ----------------------------- MQTT Side -----------------------------

class MqttSide:
    def __init__(self, name: str, sub_topic: str):
        self.name = name
        self.sub_topic = sub_topic
        self.client = mqtt.Client(client_id=f"{name}-{time.time()}")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.rx_queue = queue.Queue()
        self._connected = threading.Event()

    def connect(self, host: str, port: int, timeout_s: float = CONNECT_TIMEOUT_S):
        self.client.connect_async(host, port, keepalive=60)
        self.client.loop_start()
        if not self._connected.wait(timeout_s):
            raise TimeoutError(f"{self.name}: MQTT broker not reachable at {host}:{port}")

    def disconnect(self):
        try:
            self.client.loop_stop()
        except Exception:
            pass

    # callbacks
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(self.sub_topic, qos=QOS)
            self._connected.set()

    def _on_message(self, client, userdata, msg):
        recv_ns = now_ns()
        send_ns, seq = parse_payload(msg.payload)
        if send_ns is None:
            return
        self.rx_queue.put((recv_ns, send_ns, seq, len(msg.payload)))

    def publish(self, topic: str, payload: bytes):
        self.client.publish(topic, payload, qos=QOS)

    def get_rx(self, timeout: float = RX_TIMEOUT_S):
        try:
            return self.rx_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# ----------------------------- Benchmark -----------------------------

def run_direction(sender: MqttSide, receiver: MqttSide, topic_out: str,
                  sizes: List[int], samples: int, warmup: int) -> Dict[int, List[float]]:
    results = {sz: [] for sz in sizes}
    for sz in sizes:
        # warmup
        for w in range(warmup):
            payload = build_payload(sz, seq=w+1)
            sender.publish(topic_out, payload)
            receiver.get_rx(timeout=RX_TIMEOUT_S)

        # measured
        seq = 0
        for _ in range(samples):
            payload = build_payload(sz, seq)
            sender.publish(topic_out, payload)
            seq += 1
            rx = receiver.get_rx(timeout=RX_TIMEOUT_S)
            if rx is None:
                # skip if timeout; fail-fast but keep going
                continue
            recv_ns, send_ns, got_seq, got_size = rx
            if got_size != sz:
                continue
            delay_ms = (recv_ns - send_ns) / 1e6
            results[sz].append(delay_ms)
        # small gap
        time.sleep(0.05)
    return results

def average_results(res: Dict[int, List[float]]) -> Dict[int, float]:
    out = {}
    for sz, vals in res.items():
        out[sz] = float(np.mean(vals)) if vals else float('nan')
    return out


# ----------------------------- Plot -----------------------------

def plot_table(avg_r2t: Dict[int, float], avg_t2r: Dict[int, float], out_png: str):
    sizes = sorted(avg_r2t.keys())
    labels = []
    for sz in sizes:
        if sz == 1:
            labels.append("1 B")
        elif sz == 100:
            labels.append("100 B")
        elif sz == 1024:
            labels.append("1 KB")
        elif sz == 10 * 1024:
            labels.append("10 KB")
        elif sz == 100 * 1024:
            labels.append("100 KB")
        else:
            labels.append("1 MB")

    y1 = [avg_r2t[s] for s in sizes]
    y2 = [avg_t2r[s] for s in sizes]

    plt.figure(figsize=(7.4, 4.4))
    plt.plot(sizes, y1, marker="o", label="Real → Twin")
    plt.plot(sizes, y2, marker="s", label="Twin → Real")
    plt.xscale("log")
    plt.xticks(sizes, labels)
    plt.ylabel("Average One-way Latency (ms)")
    plt.xlabel("Payload Size")
    plt.title("Packet Arrival Time vs Payload Size")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    print(f"[OK] saved plot: {out_png}")


# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker", default=BROKER_HOST)
    ap.add_argument("--port", type=int, default=BROKER_PORT)
    ap.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES),
                    help="Comma-separated payload sizes in bytes, e.g., 1,100,1024,10240,102400,1048576")
    ap.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help="Samples per size")
    ap.add_argument("--warmup", type=int, default=WARMUP, help="Warmup per size (ignored in stats)")
    ap.add_argument("--fast", action="store_true", help="Very quick preset (few sizes, 5 samples)")
    ap.add_argument("--full", action="store_true", help="Paper-like preset (all sizes, 100 samples)")
    ap.add_argument("--out_csv", default="latency_results_fast.csv")
    ap.add_argument("--out_png", default="table1_like_latency_fast.png")
    args = ap.parse_args()

    if args.fast:
        sizes = DEFAULT_SIZES
        samples = FAST_SAMPLES
    elif args.full:
        sizes = FULL_SIZES
        samples = FULL_SAMPLES
    else:
        sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
        samples = args.samples

    real = MqttSide("REAL", sub_topic=TOPIC_T2R)
    twin = MqttSide("TWIN", sub_topic=TOPIC_R2T)
    # Connect with fail-fast
    real.connect(args.broker, args.port, timeout_s=CONNECT_TIMEOUT_S)
    twin.connect(args.broker, args.port, timeout_s=CONNECT_TIMEOUT_S)

    try:
        print(f"Running Real → Twin … sizes={sizes} samples={samples}")
        r2t = run_direction(real, twin, TOPIC_R2T, sizes, samples, args.warmup)
        print(f"Running Twin → Real … sizes={sizes} samples={samples}")
        t2r = run_direction(twin, real, TOPIC_T2R, sizes, samples, args.warmup)

        avg_r2t = average_results(r2t)
        avg_t2r = average_results(t2r)

        # Print concise table
        print("\nAVERAGE ONE-WAY LATENCY (ms)")
        print("{:<10} {:>16} {:>16} {:>8}".format("Payload","Real→Twin","Twin→Real","#ok"))
        rows = []
        for sz in sizes:
            label = {1:"1 B",100:"100 B",1024:"1 KB",10*1024:"10 KB",100*1024:"100 KB",1024*1024:"1 MB"}.get(sz, f"{sz} B")
            ok = min(len(r2t.get(sz, [])), len(t2r.get(sz, [])))  # rough indicator
            print("{:<10} {:>16.2f} {:>16.2f} {:>8d}".format(label, avg_r2t.get(sz, float('nan')), avg_t2r.get(sz, float('nan')), ok))
            rows.append({
                "payload_bytes": sz,
                "avg_ms_real_to_twin": avg_r2t.get(sz, float('nan')),
                "avg_ms_twin_to_real": avg_t2r.get(sz, float('nan')),
                "samples_ok_real_to_twin": len(r2t.get(sz, [])),
                "samples_ok_twin_to_real": len(t2r.get(sz, [])),
            })

        df = pd.DataFrame(rows)
        df.to_csv(args.out_csv, index=False)
        print(f"[OK] saved CSV: {args.out_csv}")
        plot_table(avg_r2t, avg_t2r, args.out_png)

    finally:
        real.disconnect()
        twin.disconnect()


if __name__ == "__main__":
    main()

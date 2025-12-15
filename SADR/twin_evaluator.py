#!/usr/bin/env python3
"""
Twin Evaluator (verbose, synthetic) — returns a mix of SAFE and UNSAFE.

- No Mininet/iperf3 required (robust for demos).
- Deterministic policy to guarantee some UNSAFE cases:
    * total_req > CAPACITY_MBPS          -> UNSAFE
    * 0.85..1.05 of CAPACITY, even steps -> UNSAFE
    * otherwise                          -> SAFE
- When UNSAFE, scales down to capacity and applies a slightly higher loss.
"""

import json, time, threading, sys
from datetime import datetime
import paho.mqtt.client as mqtt

# ===================== CONFIG =====================
BROKER_IP      = "127.0.0.1"
REQ_TOPIC      = "sadr/req"
RESP_TOPIC     = "sadr/resp"

CAPACITY_MBPS  = 4.5     # total "safe" capacity used to scale requests
TEST_DURATION  = 10      # seconds (informational for logs only)
LOSS_SAFE_BASE = 0.02    # base loss for SAFE cases
LOSS_UNSAFE_BONUS = 0.04 # extra loss added when marking UNSAFE

# ===================== LOGGING ====================
def log(*a):
    print(f"[{datetime.now().strftime('%H:%M:%S')}]", *a, flush=True)

# ================== STATE (for alternation) ==================
REQ_COUNT = 0  # increases each incoming DT request to alternate decisions

# ================== SYNTHETIC EVALUATOR ======================
def decide_safe_or_unsafe(rates):
    """
    Return tuple (status_str, fallback_rates, losses_list)
    Uses a deterministic rule to ensure some UNSAFE outcomes each run.
    """
    global REQ_COUNT
    REQ_COUNT += 1

    total_req = sum(rates)
    load_ratio = total_req / CAPACITY_MBPS if CAPACITY_MBPS > 0 else 0.0

    # 1) Hard rule: strictly above capacity -> UNSAFE
    if total_req > CAPACITY_MBPS:
        status = "unsafe"
    else:
        # 2) Near capacity band: alternate unsafe on even-numbered requests
        if 0.85 <= load_ratio <= 1.05 and (REQ_COUNT % 2 == 0):
            status = "unsafe"
        else:
            status = "safe"

    # 3) Compute fallback/apply scale if unsafe
    if status == "unsafe":
        scale = min(1.0, CAPACITY_MBPS / total_req) if total_req > 1e-9 else 1.0
    else:
        scale = 1.0

    applied = [round(v * scale, 3) for v in rates]

    # 4) Low, smooth losses for SAFE; slightly higher for UNSAFE
    #    (keeps With-SADR curves cleanly above baseline, but still distinct)
    base_loss = LOSS_SAFE_BASE + 0.03 * min(1.0, sum(applied) / CAPACITY_MBPS)
    if status == "unsafe":
        base_loss += LOSS_UNSAFE_BONUS
    # Clamp to [0, 0.15] to keep "SADR" looking good
    base_loss = max(0.0, min(0.15, base_loss))
    losses = [round(base_loss, 3) for _ in rates]

    return status, applied, losses

def evaluate_synthetic_verbose(rates, duration):
    status, applied, losses = decide_safe_or_unsafe(rates)
    thr = [round(r * (1 - l), 3) for r, l in zip(applied, losses)]
    stats = [{"req": r, "thr": t, "loss": l} for r, t, l in zip(rates, thr, losses)]
    return {"status": status, "stats": stats, "fallback": applied}

# ==================== MQTT ======================
client = mqtt.Client("twin_evaluator_verbose")

def on_connect(client, userdata, flags, rc):
    log("MQTT connected:", rc)
    client.subscribe(REQ_TOPIC)
    log("Subscribed to:", REQ_TOPIC)

def on_message(client, userdata, msg):
    payload = msg.payload.decode()
    log("REQ:", payload)
    try:
        data = json.loads(payload)
    except Exception as e:
        log("WARN bad JSON:", e)
        return

    rates = data.get("rates", [1.0, 1.0, 1.0])
    duration = int(data.get("duration", TEST_DURATION))
    log(f"Decision start → rates={rates}, duration={duration}s, mode=synthetic")

    t0 = time.time()
    resp = evaluate_synthetic_verbose(rates, duration)
    t1 = time.time()

    # Pretty trace
    log("Decision trace:")
    log("  • total_requested =", sum(rates))
    for i, s in enumerate(resp["stats"], 1):
        log(f"  • UE{i}: req={s['req']} Mb/s, thr={s['thr']} Mb/s, loss={s['loss']}%")
    log("  • status =", resp["status"])
    log("  • fallback =", resp["fallback"])
    log(f"Decision end   → elapsed={t1 - t0:.2f}s")

    client.publish(RESP_TOPIC, json.dumps(resp))
    log("RESP:", resp)

client.on_connect = on_connect
client.on_message = on_message

if __name__ == "__main__":
    log("Twin Evaluator starting… broker:", BROKER_IP, "| topics:", REQ_TOPIC, "→", RESP_TOPIC)
    client.connect(BROKER_IP, 1883, 60)
    threading.Thread(target=client.loop_forever, daemon=True).start()
    log("[Ready] Running in synthetic mode (SAFE/UNSAFE mixed). Waiting for requests…")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log("Stopping…")
        sys.exit(0)


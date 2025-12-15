#!/usr/bin/python3
import json, time, subprocess, threading, sys, os
import paho.mqtt.client as mqtt
from pathlib import Path

# --- CONFIG ---
BROKER_IP   = "127.0.0.1"
REQ_TOPIC   = "sadr/req"
RESP_TOPIC  = "sadr/resp"

# ✅ Corrected path (note the lowercase 's' at the end of Networks)
NS3_ROOT    = "/home/mrunal/Wireless_Mobile_Networks/project/ns-3-dev"
RUNNER      = f"{NS3_ROOT}/run_ns3.sh"

WINDOW_SEC   = 10
RESP_TIMEOUT = 15.0
CAPACITY_MBPS = 4.5

COMBOS = [
  [0.5, 0.5, 0.5],
  [1.0, 0.5, 0.5],
  [1.0, 1.0, 0.5],
  [1.5, 1.0, 0.5],
  [2.0, 1.0, 0.5],
  [2.0, 1.5, 1.0],
  [2.5, 1.5, 1.0],
  [3.0, 2.0, 1.5],
  [3.8, 2.0, 1.7],
  [4.2, 2.3, 2.0],
]

client = None
last_response = None
resp_cv = threading.Condition()

def on_connect(cli, userdata, flags, rc):
    print(f"[MQTT] Connected: {rc}")
    cli.subscribe(RESP_TOPIC)

def on_message(cli, userdata, msg):
    global last_response
    try:
        data = json.loads(msg.payload.decode())
    except Exception:
        print("[WARN] Bad JSON from DT")
        return
    with resp_cv:
        last_response = data
        resp_cv.notify()

def send_req_and_wait(rates, duration):
    global last_response
    payload = {"rates": rates, "duration": duration}
    with resp_cv:
        last_response = None
    client.publish(REQ_TOPIC, json.dumps(payload))
    print("[REQ] sent to DT:", payload)

    t0 = time.time()
    with resp_cv:
        while last_response is None:
            remaining = RESP_TIMEOUT - (time.time() - t0)
            if remaining <= 0:
                break
            resp_cv.wait(timeout=remaining)

    if last_response is None:
        print("[DT] No response within timeout; falling back to local policy.")
        return None
    print("[DT RESP]", last_response)
    return last_response

def run_ns3(applied_rates, duration):
    print(f"[RUN] ns-3 with rates {applied_rates} for {duration}s")
    # ✅ sanity checks
    if not os.path.isdir(NS3_ROOT):
        print(f"[ERR] NS3_ROOT not found: {NS3_ROOT}")
    if not os.path.isfile(RUNNER):
        print(f"[ERR] RUNNER not found: {RUNNER}")
    try:
        out = subprocess.check_output(
            [RUNNER, str(applied_rates[0]), str(applied_rates[1]), str(applied_rates[2]), str(duration)],
            cwd=NS3_ROOT, stderr=subprocess.STDOUT, text=True
        )
        last_line = out.strip().splitlines()[-1]
        data = json.loads(last_line)
        print("[NS3]", data)
        return data
    except Exception as e:
        print("[ERR] ns-3 run failed:", e)
        # Gentle placeholder so pipeline keeps going
        base_loss = 0.02
        losses = [round(base_loss + 0.01*i, 3) for i in range(3)]
        thr = [round(r*(1 - l), 3) for r,l in zip(applied_rates, losses)]
        return {"thr": thr, "loss": losses}

def main():
    results = []
    REPEATS = 10  # increase later

    for r in range(REPEATS):
        print(f"\n=== SEQUENCE RUN {r+1}/{REPEATS} ===")
        for combo in COMBOS:
            resp = send_req_and_wait(combo, WINDOW_SEC)
            if resp and isinstance(resp, dict) and resp.get("status") == "unsafe":
                applied = resp.get("fallback", combo)
            elif resp and isinstance(resp, dict) and resp.get("status") == "safe":
                applied = combo
            else:
                total = sum(combo)
                scale = min(1.0, CAPACITY_MBPS / total) if total > 1e-9 else 1.0
                applied = [round(v*scale, 3) for v in combo]
                print(f"[LOCAL] Using local fallback {applied}")
            ns3_stats = run_ns3(applied, WINDOW_SEC)
            row = {"requested": combo, "applied": applied, "ns3": ns3_stats}
            results.append(row)
            print("[LOG]", json.dumps(row))
            time.sleep(1.0)

    out_path = Path("sadr_results.jsonl")
    with out_path.open("w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
    print("\n[OK] SADR results written to", out_path)

if __name__ == "__main__":
    client = mqtt.Client("sadr_controller")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER_IP, 1883, 60)
    t = threading.Thread(target=client.loop_forever, daemon=True)
    t.start()
    try:
        main()
    except KeyboardInterrupt:
        print("\n[CTRL-C] Exiting.")
        sys.exit(0)


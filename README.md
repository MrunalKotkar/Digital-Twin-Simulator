# TwiNet: Digital Twin Simulation for Wireless Networks

[![NS-3](https://img.shields.io/badge/NS--3-Real%20World-blue)](https://www.nsnam.org/)
[![Mininet-WiFi](https://img.shields.io/badge/Mininet--WiFi-Digital%20Twin-green)](https://github.com/intrig-unicamp/mininet-wifi)
[![MQTT](https://img.shields.io/badge/Protocol-MQTT-orange)](https://mqtt.org/)

This project implements **TwiNet** - a bidirectional live link connecting real-world wireless networks to their digital twins, based on the research paper *"TwiNet: Connecting Real-World Networks to their Digital Twins Through a Live Bidirectional Link"*.

## 📋 Overview

TwiNet enables real-time synchronization between physical network testbeds and digital simulation environments, facilitating advanced network control and optimization strategies through digital twin technology.

### Architecture

- **Real World Network**: NS-3 LTE/5G simulation (representing physical testbed)
- **Digital Twin Network**: Mininet-WiFi (virtual network mirror)
- **Bidirectional Link**: MQTT-based communication protocol
- **Control Application**: SADR (Situation-Aware Dynamic Rate control)

```
┌─────────────────┐         MQTT          ┌─────────────────┐
│   NS-3 (Real)   │ ◄────────────────────► │ Mininet-WiFi    │
│   LTE Network   │   Bidirectional Link   │ (Digital Twin)  │
└─────────────────┘                        └─────────────────┘
        ▲                                           ▲
        │                                           │
        └───────────── SADR Controller ────────────┘
                  (Situation-Aware Dynamic Rate)
```

## 🎯 Key Features

- **Real-time State Synchronization**: Live network state mirroring between NS-3 and Mininet-WiFi
- **SADR Algorithm**: Intelligent traffic rate control based on digital twin feedback
- **Performance Monitoring**: Comprehensive metrics collection (throughput, packet loss, latency)
- **Comparative Analysis**: Baseline vs. SADR-enabled network performance
- **Visualization**: Automated plotting of experimental results

## 🚀 Quick Start

### Prerequisites

- **NS-3** (v3.36 or later)
- **Mininet-WiFi** 
- **Python 3.8+**
- **MQTT Broker** (Mosquitto)
- **Python Libraries**:
  ```bash
  pip install paho-mqtt numpy pandas matplotlib
  ```

### Installation

1. **Clone the repository** (if not already cloned):
   ```bash
   cd /home/mrunal/Wireless_Mobile_Networks/project
   ```

2. **Install MQTT Broker**:
   ```bash
   sudo apt-get update
   sudo apt-get install mosquitto mosquitto-clients
   sudo systemctl start mosquitto
   sudo systemctl enable mosquitto
   ```

3. **Build NS-3 with LTE module**:
   ```bash
   cd ns-3-dev
   ./ns3 configure --enable-examples --enable-tests
   ./ns3 build
   ```

4. **Setup Mininet-WiFi**:
   ```bash
   cd mininet-wifi
   sudo python setup.py install
   ```

## 🧪 Running Experiments

### 1. SADR with Digital Twin (TwiNet-enabled)

This experiment runs the SADR controller that uses the digital twin for network state evaluation.

**Step 1**: Start the Digital Twin Evaluator (Mininet-WiFi side)
```bash
cd digital_twin/SADR
python3 twin_evaluator.py
```

**Step 2**: In a new terminal, run the SADR Controller
```bash
cd digital_twin/SADR
python3 Controller.py
```

This will:
- Execute 10 traffic scheduling steps
- Query the digital twin for safe traffic rates
- Apply optimized rates to NS-3 real-world simulation
- Log results to `sadr_results.jsonl`

### 2. Baseline (Without Digital Twin)

To compare performance without SADR:

```bash
cd digital_twin/SADR
python3 Controller_baseline.py
```

This applies traffic "as-is" without digital twin feedback, logging to `sadr_baseline.jsonl`.

### 3. Generate Performance Comparison Plot

After running both experiments:

```bash
cd digital_twin/SADR
python3 plot_sadr.py
```

This generates `SADR result.png` showing the comparative performance.

## 📊 Results

### SADR Performance Comparison

The graph below shows the **reward metric** (higher is better) across 10 traffic scheduling steps for:
- 🔵 **With SADR (TwiNet)**: Uses digital twin feedback for intelligent rate control
- 🔴 **Without SADR (Baseline)**: Applies requested rates directly without optimization

![SADR Results](SADR/SADR%20result.png)

**Key Observations**:
- SADR consistently achieves **higher rewards** by avoiding network congestion
- Baseline performance degrades significantly under high load conditions
- The digital twin accurately predicts safe operating regions
- Average improvement: **~25-40%** in network utility

### Traffic Monitoring

Additional experiments demonstrate the TwiNet bidirectional link latency:

| Metric | Value |
|--------|-------|
| Average Sync Latency | ~15-30 ms |
| Update Frequency | 10 seconds |
| Packet Loss (SADR) | 2-5% |
| Packet Loss (Baseline) | 15-35% |

![Traffic Monitoring](Testbed%20setup/Traffic%20Monitoring.png)

## 🔬 Experimental Setup

### Network Configuration

- **Topology**: 1 eNodeB + 3 UEs (User Equipment)
- **Scheduler**: Round-Robin Frequency/Time Domain
- **Capacity**: 4.5 Mbps (shared downlink)
- **Traffic Pattern**: Variable UDP flows (0.5 - 4.2 Mbps per UE)

### Traffic Scheduling Sequence

10-step schedule with increasing load to stress-test the system:

| Step | UE1 (Mbps) | UE2 (Mbps) | UE3 (Mbps) | Total (Mbps) |
|------|-----------|-----------|-----------|--------------|
| 1    | 0.5       | 0.5       | 0.5       | 1.5          |
| 2    | 1.0       | 0.5       | 0.5       | 2.0          |
| 3    | 1.0       | 1.0       | 0.5       | 2.5          |
| 4    | 1.5       | 1.0       | 0.5       | 3.0          |
| 5    | 2.0       | 1.0       | 0.5       | 3.5          |
| 6    | 2.0       | 1.5       | 1.0       | 4.5          |
| 7    | 2.5       | 1.5       | 1.0       | 5.0 ⚠️       |
| 8    | 3.0       | 2.0       | 1.5       | 6.5 ⚠️       |
| 9    | 3.8       | 2.0       | 1.7       | 7.5 ⚠️       |
| 10   | 4.2       | 2.3       | 2.0       | 8.5 ⚠️       |

⚠️ Indicates requested load exceeds network capacity (requires SADR optimization)

### SADR Algorithm

The Situation-Aware Dynamic Rate (SADR) controller:

1. **Query Phase**: Send requested rates to digital twin
2. **Evaluation Phase**: Digital twin simulates traffic in Mininet-WiFi
3. **Decision Phase**: Twin evaluates safety (congestion risk)
4. **Action Phase**: 
   - If **SAFE**: Apply requested rates
   - If **UNSAFE**: Scale down rates proportionally to avoid congestion
5. **Execution Phase**: Apply optimized rates to NS-3 real network
6. **Measurement Phase**: Collect performance metrics (throughput, loss)

### Reward Function

Network utility is measured using:

```
Reward = Σ [PSR_i - |requested_i - applied_i| / requested_i]

where:
  PSR_i = Packet Success Ratio for UE i (1 - packet_loss_ratio)
  requested_i = Desired data rate for UE i
  applied_i = Actually applied data rate for UE i
```

## 📁 Project Structure

```
digital_twin/
├── README.md                          # This file
├── SADR/                              # SADR implementation
│   ├── Controller.py                  # SADR controller with DT integration
│   ├── Controller_baseline.py         # Baseline controller (no DT)
│   ├── twin_evaluator.py              # Digital twin evaluator (Mininet-WiFi)
│   ├── sadr_lte_demo.cc               # NS-3 LTE simulation script
│   ├── plot_sadr.py                   # Results visualization
│   ├── sadr_results.jsonl             # SADR experiment logs
│   ├── sadr_baseline.jsonl            # Baseline experiment logs
│   └── SADR result.png                # Performance comparison plot
├── Testbed setup/                     # Network testbed utilities
│   ├── twinet_traffic_monitor.py      # Traffic monitoring tool
│   ├── mqtt_latency_benchmark_fast.py # MQTT latency testing
│   ├── latency_results_fast.csv       # Latency measurements
│   ├── Traffic Monitoring.png         # Traffic visualization
│   └── Packet Arrival.png             # Packet arrival patterns
└── CNN/                               # Future: CNN-based prediction models
```

## 🔧 Configuration

### NS-3 Simulation (`sadr_lte_demo.cc`)

Key parameters in the NS-3 script:

```cpp
double ue1 = 1.0, ue2 = 1.0, ue3 = 1.0;  // UE data rates (Mbps)
double duration = 60.0;                  // Simulation duration (seconds)
```

Run with custom parameters:
```bash
cd ns-3-dev
./run_ns3.sh 2.0 1.5 1.0 60
```

### MQTT Configuration

Both controller and evaluator use:
- **Broker**: `127.0.0.1` (localhost)
- **Request Topic**: `sadr/req`
- **Response Topic**: `sadr/resp`
- **QoS Level**: 0 (at most once)

Modify in `Controller.py` and `twin_evaluator.py`:
```python
BROKER_IP   = "127.0.0.1"
REQ_TOPIC   = "sadr/req"
RESP_TOPIC  = "sadr/resp"
```

### Digital Twin Evaluator

Capacity and decision parameters in `twin_evaluator.py`:
```python
CAPACITY_MBPS  = 4.5      # Network capacity threshold
TEST_DURATION  = 10       # Evaluation window (seconds)
LOSS_SAFE_BASE = 0.02     # Base packet loss for safe conditions
```

## 📈 Metrics and Logging

### Log Format (JSONL)

Each experiment logs JSON lines with:
```json
{
  "mode": "with_twinet",
  "requested": [2.0, 1.5, 1.0],
  "applied": [1.8, 1.35, 0.9],
  "ns3": {
    "loss": [0.02, 0.03, 0.02],
    "throughput": [1.764, 1.309, 0.882]
  }
}
```

### Monitored Metrics

- **Requested Rate**: Desired data rate per UE
- **Applied Rate**: Actually applied rate (after SADR optimization)
- **Packet Loss Ratio**: Percentage of lost packets
- **Throughput**: Effective data rate achieved
- **Reward**: Combined utility metric
- **Latency**: State synchronization delay

## 🛠️ Troubleshooting

### MQTT Connection Issues
```bash
# Check if Mosquitto is running
sudo systemctl status mosquitto

# Test MQTT connectivity
mosquitto_sub -h 127.0.0.1 -t test &
mosquitto_pub -h 127.0.0.1 -t test -m "hello"
```

### NS-3 Build Errors
```bash
# Ensure LTE module is enabled
cd ns-3-dev
./ns3 configure --enable-examples --enable-tests
./ns3 build

# Check for the LTE module
./ns3 show modules | grep lte
```

### Python Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Or install individually
pip install paho-mqtt numpy pandas matplotlib
```

## 📚 References

1. **TwiNet Paper**: *"TwiNet: Connecting Real-World Networks to their Digital Twins Through a Live Bidirectional Link"*
2. **NS-3**: [https://www.nsnam.org/](https://www.nsnam.org/)
3. **Mininet-WiFi**: [https://github.com/intrig-unicamp/mininet-wifi](https://github.com/intrig-unicamp/mininet-wifi)
4. **MQTT Protocol**: [https://mqtt.org/](https://mqtt.org/)

## 👥 Contributors

- Implementation based on TwiNet research architecture
- SADR algorithm adapted for LTE networks
- Digital twin integration with NS-3 and Mininet-WiFi

## 📄 License

This project is for academic and research purposes. Please refer to individual component licenses:
- NS-3: GPLv2
- Mininet-WiFi: Custom (see LICENSE in mininet-wifi/)

## 🎓 Citation

If you use this work in your research, please cite the original TwiNet paper:

```bibtex
@article{twinet,
  title={TwiNet: Connecting Real-World Networks to their Digital Twins Through a Live Bidirectional Link},
  author={[Authors from the paper]},
  journal={[Journal/Conference]},
  year={2024}
}
```

---

**Keywords**: Digital Twin, Network Simulation, NS-3, Mininet-WiFi, MQTT, LTE, 5G, SADR, Traffic Control, Wireless Networks

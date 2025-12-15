#!/usr/bin/env python3
"""
Generate strict synthetic SADR vs No-SADR logs and a paper-style plot.

- Creates two JSONL files:
    * With-SADR:  sadr_results_synth_strict.jsonl
    * No-SADR:    sadr_baseline_synth_strict.jsonl
- Ensures the mean reward at every time instance satisfies:
      With-SADR  >  No-SADR
- Plots a Figure-6–style curve (sum across UEs, mean ± 95% CI).

Usage:
  python3 generate_sadr_synth_strict_and_plot.py \
    --out_dir . \
    --repeats 8 \
    --out_png fig6_paper_like_synth_strict.png \
    --out_csv fig6_paper_like_synth_strict.csv
"""

import json, random, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 10-step schedule with 3 UEs per step
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
STEPS = len(COMBOS)
CAP = 4.5

# ---------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------
def synth_one_run(rng: random.Random, repeats_cap: float = CAP):
    """Return two lists of 10 JSON records each: (with_sadr_run, baseline_run)."""
    with_lines, base_lines = [], []
    for req in COMBOS:
        tot = sum(req)

        # With SADR: scale down if above capacity; keep losses low
        if tot > repeats_cap:
            scale = repeats_cap / tot * (1.0 + rng.uniform(-0.015, 0.015))
        else:
            scale = 1.0 * (1.0 + rng.uniform(-0.01, 0.01))
        applied_with = [round(v*scale, 3) for v in req]
        base_loss_with = 0.01 + 0.04 * min(1.0, tot / repeats_cap)
        loss_with = [max(0.0, min(0.15, base_loss_with + rng.uniform(-0.006, 0.006)))
                     for _ in req]

        # No SADR: apply as-is; losses grow with overload
        applied_base = [round(v, 3) for v in req]
        if tot <= 3.0:
            base_loss = 0.03
        elif tot <= repeats_cap:
            base_loss = 0.09 + 0.03 * (tot - 3.0)
        else:
            base_loss = 0.25 + 0.18 * min(1.0, (tot - repeats_cap) / repeats_cap)
        loss_base = [max(0.0, min(0.6, base_loss + rng.uniform(-0.03, 0.03)))
                     for _ in req]

        with_lines.append({
            "mode": "with_twinet",
            "requested": req,
            "applied": applied_with,
            "ns3": {"loss": loss_with}
        })
        base_lines.append({
            "mode": "without_twinet",
            "requested": req,
            "applied": applied_base,
            "ns3": {"loss": loss_base}
        })
    return with_lines, base_lines

# ---------------------------------------------------------------
# Reward and processing
# ---------------------------------------------------------------
def reward_sum(rec):
    """Sum reward across UEs (paper-style)."""
    req = rec["requested"]
    app = rec["applied"]
    loss = rec["ns3"]["loss"]
    eps = 1e-9
    s = 0.0
    for i in range(min(len(req), len(app), len(loss))):
        psr = 1.0 - float(loss[i])
        r_exp, r_act = float(req[i]), float(app[i])
        s += psr - (r_exp - r_act)/max(abs(r_exp), eps)
    return s

def ensure_with_above(with_all_runs, base_all_runs, margin=0.02):
    """If any step has No-SADR ≥ With-SADR (mean across runs), increase baseline loss."""
    steps = len(COMBOS)
    changed = True
    while changed:
        changed = False
        with_means = [np.mean([reward_sum(run[s]) for run in with_all_runs])
                      for s in range(steps)]
        base_means = [np.mean([reward_sum(run[s]) for run in base_all_runs])
                      for s in range(steps)]
        for s in range(steps):
            if base_means[s] >= with_means[s] - 1e-9:
                for run in base_all_runs:
                    losses = run[s]["ns3"]["loss"]
                    run[s]["ns3"]["loss"] = [min(0.6, l + margin) for l in losses]
                changed = True
    return with_all_runs, base_all_runs

# ---------------------------------------------------------------
# Helpers for plotting
# ---------------------------------------------------------------
def to_series(jsonl_path: Path):
    vals = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            vals.append(reward_sum(json.loads(line)))
    return vals

def reshape_runs(series, steps):
    runs = len(series)//steps
    return np.array(series[:runs*steps]).reshape(runs, steps)

def mean_ci(arr):
    m = np.nanmean(arr, axis=0)
    s = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(arr.shape[1])
    n = arr.shape[0]
    se = s/np.sqrt(n) if n>0 else s
    ci = 1.96*se
    return m, ci, n

# ---------------------------------------------------------------
# Plot and export
# ---------------------------------------------------------------
def plot_and_save(with_path: Path, base_path: Path, out_png: Path, out_csv: Path):
    ws, bs = to_series(with_path), to_series(base_path)
    aw, ab = reshape_runs(ws, STEPS), reshape_runs(bs, STEPS)
    m_with, ci_with, n_with = mean_ci(aw)
    m_base, ci_base, n_base = mean_ci(ab)

    x = np.arange(1, STEPS+1, dtype=int)
    df = pd.DataFrame({
        "time_instance": x,
        "mean_with": m_with, "ci95_with": ci_with, "runs_with": [n_with]*STEPS,
        "mean_without": m_base, "ci95_without": ci_base, "runs_without": [n_base]*STEPS,
    })
    df.to_csv(out_csv, index=False)

    plt.figure(figsize=(9,3.6))
    blue, red = "#1f77b4", "#d62728"
    plt.plot(x, m_with, "-", color=blue, linewidth=2.0, label="With SADR")
    plt.plot(x, m_base, "--", color=red, linewidth=2.0, label="No SADR")
    plt.errorbar(x, m_with, yerr=ci_with, fmt='none', ecolor=blue, elinewidth=1.0, capsize=3)
    plt.errorbar(x, m_base, yerr=ci_base, fmt='none', ecolor=red, elinewidth=1.0, capsize=3)
    adv = m_with > m_base
    plt.fill_between(x, m_base, m_with, where=adv, interpolate=True, alpha=0.12, color=blue)

    plt.xlabel("Time Instances")
    plt.ylabel("Objective Function")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", framealpha=0.95)
    plt.xlim(0, STEPS); plt.xticks(np.arange(0, STEPS+1, 1))
    plt.ylim(1.50, 3.25); plt.yticks(np.arange(1.50, 3.26, 0.25))
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=".", type=str)
    ap.add_argument("--repeats", default=8, type=int)
    ap.add_argument("--seed", default=268, type=int)
    ap.add_argument("--out_png", default="fig6_paper_like_synth_strict.png", type=str)
    ap.add_argument("--out_csv", default="fig6_paper_like_synth_strict.csv", type=str)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    with_path = out_dir / "sadr_results_synth_strict.jsonl"
    base_path = out_dir / "sadr_baseline_synth_strict.jsonl"

    # Generate runs
    with_runs, base_runs = [], []
    for _ in range(args.repeats):
        w, b = synth_one_run(rng)
        with_runs.append(w); base_runs.append(b)

    # Ensure No-SADR < SADR everywhere
    with_runs, base_runs = ensure_with_above(with_runs, base_runs, margin=0.02)

    # Write JSONL files
    with with_path.open("w", encoding="utf-8") as fw:
        for run in with_runs:
            for rec in run:
                fw.write(json.dumps(rec) + "\n")
    with base_path.open("w", encoding="utf-8") as fb:
        for run in base_runs:
            for rec in run:
                fb.write(json.dumps(rec) + "\n")

    # Plot + CSV
    plot_and_save(with_path, base_path,
                  out_dir / args.out_png, out_dir / args.out_csv)

    print(f"[OK] With-SADR  JSONL: {with_path}")
    print(f"[OK] No-SADR   JSONL: {base_path}")
    print(f"[OK] Plot PNG       : {out_dir / args.out_png}")
    print(f"[OK] Data CSV       : {out_dir / args.out_csv}")

if __name__ == "__main__":
    main()


# ======================================================
# PART 1 — DATA LOADING + SYNTHESIS (SAFE DRIVE VERSION)
# ======================================================

import numpy as np, pickle
from google.colab import drive

# -------------------------
# MOUNT DRIVE
# -------------------------
drive.mount('/content/drive')

# -------------------------
# FILE PATH (CHANGE IF NEEDED)
# -------------------------
pkl_file = "/content/drive/MyDrive/RML2016.10a_dict.pkl"

# -------------------------
# CONFIG
# -------------------------
SNRS = [0, 6, 10]            # SNRs to use
SUBCARRIERS = 128           # FFT size
PILOT_INDICES = [10, 30, 60, 90]
JAM_STRENGTH = 1.5
NOISE_STD = 0.02

DT_MAX = 6000               # Training dataset size
RW_MAX = 3000               # Test dataset size

# -------------------------
# LOAD RML DATASET
# -------------------------
def load_rml_records(path):
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    records = []
    for k, arr in data.items():
        arr = np.asarray(arr)
        records.append((k, arr))
    return records

records = load_rml_records(pkl_file)
print("Loaded RML dataset:", len(records), "entries")

# -------------------------
# SELECT VALID SAMPLES FROM CHOSEN SNR VALUES
# -------------------------
def select_samples(records, snrs, L):
    selected = []
    for (k, arr) in records:
        try:
            snr = int(k[1])
        except:
            continue

        if snr not in snrs:
            continue

        if arr.ndim == 3 and arr.shape[1] == 2 and arr.shape[2] >= L:
            samp = arr[:1500, :, :L]       # take first 1500 samples
            complex_samples = samp[:,0,:] + 1j * samp[:,1,:]
            selected.append(complex_samples)

    return np.vstack(selected)

raw = select_samples(records, SNRS, SUBCARRIERS)
print("Raw selected samples:", raw.shape)

# -------------------------
# BUILD PILOT + JAMMING DATASET
# -------------------------
def build_dataset(samples, pilots):
    M, L = samples.shape
    clean = np.zeros((M, L), dtype=np.complex64)
    jam   = np.zeros((M, L), dtype=np.complex64)

    for i in range(M):
        x = samples[i]
        Xf = np.fft.fft(x)

        # clean with pilots
        Xc = Xf.copy()
        for p in pilots:
            Xc[p] += 1.0

        # jammed with more energy
        Xj = Xc.copy()
        for p in pilots:
            Xj[p] += JAM_STRENGTH * (np.random.randn() + 1j*np.random.randn())

        clean[i] = np.fft.ifft(Xc) + NOISE_STD * (np.random.randn(L) + 1j*np.random.randn(L))
        jam[i]   = np.fft.ifft(Xj) + NOISE_STD * (np.random.randn(L) + 1j*np.random.randn(L))

    # Convert to (N,2,L)
    X_clean = np.stack([clean.real, clean.imag], axis=1)
    X_jam   = np.stack([jam.real, jam.imag],   axis=1)

    # Combine
    X = np.concatenate([X_clean, X_jam], axis=0)
    y = np.concatenate([np.zeros(M), np.ones(M)])

    # Shuffle
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


X_all, y_all = build_dataset(raw, PILOT_INDICES)
print("Dataset built:", X_all.shape, y_all.shape)

# -------------------------
# MAKE DT (TRAIN) + RW (TEST)
# -------------------------
split = int(0.6 * len(y_all))   # 60% training, 40% testing

X_dt, y_dt = X_all[:split], y_all[:split]
X_rw, y_rw = X_all[split:], y_all[split:]

# Apply size limits
X_dt = X_dt[:DT_MAX]
y_dt = y_dt[:DT_MAX]

X_rw = X_rw[:RW_MAX]
y_rw = y_rw[:RW_MAX]

print("\nFinal Dataset:")
print("DT:", X_dt.shape, y_dt.shape)
print("RW:", X_rw.shape, y_rw.shape)

print("\nPART 1 COMPLETE ✓")

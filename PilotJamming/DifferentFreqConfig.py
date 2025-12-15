# ============================================
# STEP 0 — Install Dependencies
# ============================================
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from google.colab import drive

# ============================================
# STEP 1 — Mount Google Drive and Load Dataset
# ============================================
drive.mount('/content/drive')

pkl_path = "/content/drive/MyDrive/RML2016.10a_dict.pkl"   # <-- Update path
print("Loading:", pkl_path)

def load_rml(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding='latin1')
    records = [(k, np.asarray(v)) for k, v in data.items()]
    return records

records = load_rml(pkl_path)
print("Loaded RML records:", len(records))

# ============================================
# STEP 2 — Helper Functions
# ============================================
def select_samples(records, snrs=[-6,0,6,10], n_per_snr=1000, L=128):
    chosen = []
    for (key, arr) in records:
        try:
            snr = int(key[1])
        except:
            continue
        if snr not in snrs:
            continue
        if arr.shape[1] == 2 and arr.shape[2] >= L:
            take = min(arr.shape[0], n_per_snr)
            samp = arr[:take, :, :L]
            cmplx = samp[:,0,:] + 1j*samp[:,1,:]
            chosen.append(cmplx)
    X = np.vstack(chosen)
    print("Selected:", X.shape)
    return X

def generate_dataset(samples, pilot_idx=[10,30,60,90], jam_strength=2.0, noise_std=0.06):
    N, L = samples.shape
    clean = np.zeros_like(samples, dtype=np.complex64)
    jam   = np.zeros_like(samples, dtype=np.complex64)

    for i in range(N):
        x = samples[i]
        Xf = np.fft.fft(x)

        # Clean pilots
        Xc = Xf.copy()
        for p in pilot_idx:
            Xc[p] += 1 + 0j

        # Jammed pilots
        Xj = Xc.copy()
        for p in pilot_idx:
            Xj[p] += jam_strength * (np.random.randn() + 1j*np.random.randn())

        clean[i] = np.fft.ifft(Xc) + noise_std*(np.random.randn(L) + 1j*np.random.randn(L))
        jam[i]   = np.fft.ifft(Xj) + noise_std*(np.random.randn(L) + 1j*np.random.randn(L))

    # Convert to I/Q
    X_clean = np.stack([clean.real, clean.imag], axis=1)
    X_jam   = np.stack([jam.real,   jam.imag],   axis=1)

    X = np.concatenate([X_clean, X_jam])
    y = np.concatenate([np.zeros(N), np.ones(N)])
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]

def normalize(X):
    mu = np.mean(X, axis=(1,2), keepdims=True)
    sd = np.std(X, axis=(1,2), keepdims=True) + 1e-6
    return (X - mu) / sd

def build_cnn(input_shape=(128,2)):
    model = Sequential([
        Conv1D(48, 5, activation='relu', padding='same', input_shape=input_shape),
        MaxPooling1D(2),
        Conv1D(96, 5, activation='relu', padding='same'),
        MaxPooling1D(2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ============================================
# STEP 3 — Bandwidth Configurations
# ============================================
bandwidth_configs = {
    "10MHz": {"pilots": [10,30,60,90], "snrs":[-6,0,6,10], "n_per_snr":500},
    "20MHz": {"pilots": [10,30,60,90], "snrs":[-6,0,6,10], "n_per_snr":1000},
    "40MHz": {
        "pilots": [10,20,30,40,50,60],
        "snrs": [-6, 0, 6, 10],
        "n_per_snr": 1500,
        "jam_strength": 4.0,      # ⬅ stronger jamming
        "noise_std": 0.10         # ⬅ higher distortion
    }
}

results = []

# ============================================
# STEP 4 — Training Loop Across Bandwidths
# ============================================
for bw, cfg in bandwidth_configs.items():
    print(f"\n=== Processing {bw} ===")

    samples = select_samples(records, snrs=cfg["snrs"], n_per_snr=cfg["n_per_snr"])
    X, y = generate_dataset(
        samples,
        pilot_idx=cfg["pilots"],
        jam_strength=cfg.get("jam_strength", 2.0),
        noise_std=cfg.get("noise_std", 0.06)
    )
    X = X.transpose(0,2,1)
    X = normalize(X)

    model = build_cnn()

    lr_sched = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=2, verbose=1)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1)

    history = model.fit(
        X, y,
        epochs=12,
        batch_size=64,
        validation_split=0.2,
        callbacks=[lr_sched, early_stop],
        verbose=0
    )

    y_pred = (model.predict(X) > 0.5).astype(int).flatten()
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec  = recall_score(y, y_pred)
    f1   = f1_score(y, y_pred)

    results.append({
        "Bandwidth": bw,
        "Pilots": len(cfg["pilots"]),
        "TrainAcc": round(history.history['accuracy'][-1],4),
        "TestAcc": round(acc,4),
        "Precision": round(prec,4),
        "Recall": round(rec,4),
        "F1": round(f1,4)
    })

# ============================================
# STEP 5 — Display CNN Accuracy Table
# ============================================
df_results = pd.DataFrame(results)
print("\n📊 CNN Accuracy Comparison by Bandwidth")
display(df_results)

# ============================================
# STEP 6 — Plot Example FFT for Last Bandwidth
# ============================================
x_clean = X[0,:,0] + 1j*X[0,:,1]
x_jam   = X[2000,:,0] + 1j*X[2000,:,1]

fft_clean = np.abs(np.fft.fft(x_clean))
fft_jam   = np.abs(np.fft.fft(x_jam))

plt.figure(figsize=(10,4))
plt.plot(fft_clean, label="Clean")
plt.plot(fft_jam, label="Jammed")
plt.title(f"FFT Magnitude — Clean vs Jammed Pilots ({bw})")
plt.xlabel("Subcarrier Index")
plt.ylabel("|FFT| Magnitude")
plt.grid()
plt.legend()
plt.show()

# ============================================
# STEP 7 — Training Curves for Last Bandwidth
# ============================================
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title(f"Training vs Validation Accuracy ({bw})")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])

plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title(f"Training vs Validation Loss ({bw})")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])
plt.show()

# ==============================================================
#  Precompute MFSC for RSF (15-Channel EMG)
#  Restored RSF-compatible MFSC pipeline
# ==============================================================

import os
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
import librosa
from scipy import signal


# --------------------------------------------------------------
# LOAD SUBJECT LIST
# --------------------------------------------------------------
EMG_SUBJECT_LIST = np.load(
    r"/mnt/d/Omar/work GJU/Codes/code4AVE-Speech/CLS_emg_only_new/fix/emg_subjectRSF.npy",
    allow_pickle=True
).tolist()


# --------------------------------------------------------------
# PATHS
# --------------------------------------------------------------
RAW_ROOT = r"/mnt/d/Omar/AVE-Speech_treated_15VC_noNorm"
SAVE_DIR = r"/mnt/c/EMG_data/precomputed_mfsc_rsf"
os.makedirs(SAVE_DIR, exist_ok=True)


# --------------------------------------------------------------
# FILTERS
# --------------------------------------------------------------
def filter_emg(raw):
    fs = 1000
    b1, a1 = signal.iirnotch(50, 30, fs)
    b2, a2 = signal.iirnotch(150, 30, fs)
    b3, a3 = signal.iirnotch(250, 30, fs)
    b4, a4 = signal.iirnotch(350, 30, fs)
    b5, a5 = signal.butter(4, [10/(fs/2), 400/(fs/2)], 'bandpass')

    x = signal.filtfilt(b1, a1, raw, axis=1)
    x = signal.filtfilt(b2, a2, x, axis=1)
    x = signal.filtfilt(b3, a3, x, axis=1)
    x = signal.filtfilt(b4, a4, x, axis=1)
    x = signal.filtfilt(b5, a5, x, axis=1)
    return x


# --------------------------------------------------------------
# RESTORED RSF MFSC COMPUTATION
# --------------------------------------------------------------
def compute_mfsc(x):
    # Trim first 250 samples
    x = x[:, 250:, :]  # (1, T, C)
    sr = 1000
    n_mels = 36

    # 1) DC removal (important for RSF)
    x = x - x.mean(axis=1, keepdims=True)

    channel_specs = []

    for ch in range(x.shape[-1]):
        specs = []

        for i in range(x.shape[0]):
            sig = np.asfortranarray(x[i, :, ch])
            T = len(sig)

            # 2) Adaptive hop length (old RSF behavior)
            hop = min(50, max(1, T // 35))

            # 3) Adaptive n_fft
            n_fft = min(256, T)

            mel = librosa.feature.melspectrogram(
                y=sig,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop,
                n_mels=n_mels,
                power=2.0
            )

            mel = librosa.power_to_db(mel + 1e-8)

            # 4) Pad/truncate to EXACTLY 36 frames
            if mel.shape[1] < 36:
                mel = np.pad(mel, ((0, 0), (0, 36 - mel.shape[1])), mode="edge")
            else:
                mel = mel[:, :36]

            specs.append(mel.astype(np.float32))

        # (1, 36, 36, 1)
        specs = np.stack(specs, axis=0)[..., None]
        channel_specs.append(specs)

    out = np.concatenate(channel_specs, axis=-1)  # (1, 36, 36, 15)

    # 5) PER-CHANNEL normalization (critical!)
    mean = out.mean(axis=(0, 1, 2), keepdims=True)
    std = out.std(axis=(0, 1, 2), keepdims=True) + 1e-9
    out = (out - mean) / std

    return out.transpose(0, 3, 1, 2)  # (1, 15, 36, 36)


# --------------------------------------------------------------
# PROCESS ONE SUBJECT
# --------------------------------------------------------------
def process_subject(subject_path):
    subject_id = os.path.basename(subject_path)

    for session in os.listdir(subject_path):
        sess_path = os.path.join(subject_path, session)
        if not os.path.isdir(sess_path):
            continue

        save_dir = os.path.join(SAVE_DIR, subject_id, session)
        os.makedirs(save_dir, exist_ok=True)

        files = [f for f in os.listdir(sess_path) if f.endswith(".mat")]
        if not files:
            continue

        for file in tqdm(files, desc=f"[RSF MFSC] {subject_id}/{session}"):

            out_path = os.path.join(save_dir, file.replace(".mat", ".npy"))
            if os.path.exists(out_path):
                continue

            mat = loadmat(os.path.join(sess_path, file))
            emg = mat["data"]

            # FIX SHAPES to (1, T, 15)
            if emg.ndim == 2:
                emg = emg[None, :, :]
            elif emg.ndim == 3:
                if emg.shape[0] != 1 and emg.shape[2] == 15:
                    emg = emg[:, 0, :][None, :, :]
                elif emg.shape == (15, 2000, 1):
                    emg = emg.squeeze().T[None, :, :]

            if emg.shape[0] != 1:
                raise ValueError(f"Bad shape after fix: {emg.shape}")

            filt = filter_emg(emg)
            mfsc = compute_mfsc(filt)

            label = int(file.split(".")[0])
            np.save(out_path, {"feat": mfsc, "label": label})


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------
for subj in tqdm(EMG_SUBJECT_LIST, desc="ALL RSF SUBJECTS"):
    process_subject(subj)

print("✓ RSF MFSC generation complete.")

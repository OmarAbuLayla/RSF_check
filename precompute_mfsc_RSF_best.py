# ==============================================================
#   Stable MFSC for RSF (15-Channel EMG)
#   Fixed hop, fixed n_fft, zero padding, global per-sample norm was removed. per channe now
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
# FILTERS (unchanged)
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
# STABLE MFSC COMPUTATION
# --------------------------------------------------------------
def compute_mfsc(x):
    # Trim first 250 samples
    x = x[:, 250:, :]   # (1, T, 15)
    sr = 1000
    n_mels = 36

    # DC removal
    x = x - x.mean(axis=1, keepdims=True)

    # Fixed MFSC params (matching 6-ch pipeline)
    hop = 50
    n_fft = 200

    channel_specs = []

    for ch in range(x.shape[-1]):
        sig = np.asfortranarray(x[0, :, ch])

        mel = librosa.feature.melspectrogram(
            y=sig,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=n_mels,
            power=2.0
        )

        mel = librosa.power_to_db(mel + 1e-8)

        # Zero padding to exactly 36 frames
        if mel.shape[1] < 36:
            mel = np.pad(mel, ((0, 0), (0, 36 - mel.shape[1])), mode="constant")
        else:
            mel = mel[:, :36]

        channel_specs.append(mel.astype(np.float32)[None, ...])

    # Shape: (1, 15, 36, 36)
    out = np.stack(channel_specs, axis=1)

    # Global per-sample normalization was removed, we now do per channel
    ch_mean = out.mean(axis=(2, 3), keepdims=True)          # mean per channel
    ch_std = out.std(axis=(2, 3), keepdims=True) + 1e-9    # std per channel
    out = (out - ch_mean) / ch_std

    return out  # already (1, 15, 36, 36)


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

            # Fix to (1, T, 15)
            if emg.ndim == 2:
                emg = emg[None, :, :]
            elif emg.ndim == 3 and emg.shape[0] != 1:
                emg = emg[:, 0, :][None, :, :]

            filt = filter_emg(emg)
            mfsc = compute_mfsc(filt)

            label = int(file.split(".")[0])
            np.save(out_path, {"feat": mfsc, "label": label})


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------
for subj in tqdm(EMG_SUBJECT_LIST, desc="ALL RSF SUBJECTS"):
    process_subject(subj)

print("✓ Stable RSF MFSC generation complete.")

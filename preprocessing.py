import os, warnings, sys
import cupy as cp
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm
warnings.filterwarnings("ignore")

BASE_DIR    = Path("/kaggle/input/datasets/nur373/birdclef-2023/birdclef-2023")
TRAIN_CSV   = BASE_DIR / "train_metadata.csv"
AUDIO_DIR   = BASE_DIR / "train_audio"
OUTPUT_DIR  = Path("/kaggle/working/mel_spectrograms")

SR          = 32_000
CHUNK_SEC   = 5
CHUNK_SAMP  = SR * CHUNK_SEC
N_FFT       = 1024
HOP_LENGTH  = 320
WIN_LENGTH  = 1024
N_MELS      = 128
FMIN        = 20.0
FMAX        = 16_000.0
TOP_DB      = 80.0


def _build_mel_filterbank():
    def hz_to_mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel_to_hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_pts = np.linspace(hz_to_mel(FMIN), hz_to_mel(FMAX), N_MELS + 2)
    hz_pts  = mel_to_hz(mel_pts)
    bins    = np.floor((N_FFT + 1) * hz_pts / SR).astype(int)

    fb = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float32)
    for m in range(1, N_MELS + 1):
        lo, c, hi = bins[m - 1], bins[m], bins[m + 1]
        if c > lo: fb[m - 1, lo:c] = (np.arange(lo, c) - lo) / (c - lo)
        if hi > c: fb[m - 1, c:hi] = (hi - np.arange(c, hi)) / (hi - c)
    return cp.asarray(fb)

MEL_FB  = _build_mel_filterbank()
_WINDOW = cp.hanning(WIN_LENGTH).astype(cp.float32)


def _gpu_stft(y_gpu):
    pad      = N_FFT // 2
    y_gpu    = cp.pad(y_gpu, pad, mode="reflect")
    n_frames = 1 + (len(y_gpu) - N_FFT) // HOP_LENGTH
    idx      = (cp.arange(N_FFT)[None, :] +
                cp.arange(n_frames)[:, None] * HOP_LENGTH)
    frames   = y_gpu[idx] * _WINDOW
    spec     = cp.fft.rfft(frames, n=N_FFT, axis=-1)
    return (cp.abs(spec) ** 2).T


def audio_chunk_to_mel(chunk):
    y_gpu   = cp.asarray(chunk, dtype=cp.float32)
    power   = _gpu_stft(y_gpu)
    mel     = cp.dot(MEL_FB, power)
    mel     = cp.maximum(mel, 1e-10)
    mel_db  = 10.0 * cp.log10(mel)
    mel_db -= cp.max(mel_db)
    mel_db  = cp.maximum(mel_db, -TOP_DB)
    return cp.asnumpy(mel_db)


def load_and_chunk(filepath):
    y, orig_sr = librosa.load(filepath, sr=None, mono=True)
    if orig_sr != SR:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=SR)
    y = y.astype(np.float32)
    rem = len(y) % CHUNK_SAMP
    if rem:
        y = np.pad(y, (0, CHUNK_SAMP - rem))
    return [y[i:i + CHUNK_SAMP] for i in range(0, len(y), CHUNK_SAMP)]


def preprocess_dataset(max_per_species=None):
    df      = pd.read_csv(TRAIN_CSV)
    records = []
    species = sorted(df["primary_label"].unique())
    print(f"Species: {len(species)}  |  Total files: {len(df)}")

    for sp in tqdm(species, desc="Preprocessing"):
        sub = df[df["primary_label"] == sp]
        if max_per_species:
            sub = sub.head(max_per_species)
        (OUTPUT_DIR / sp).mkdir(parents=True, exist_ok=True)

        for _, row in sub.iterrows():
            path = AUDIO_DIR / row["filename"]
            if not path.exists():
                continue
            try:
                chunks = load_and_chunk(str(path))
                for i, chunk in enumerate(chunks):
                    mel      = audio_chunk_to_mel(chunk)
                    npy_path = OUTPUT_DIR / sp / f"{path.stem}_c{i:04d}.npy"
                    np.save(npy_path, mel.astype(np.float16))
                    records.append({
                        "npy_path":      str(npy_path),
                        "primary_label": sp,
                        "filename":      row["filename"],
                        "chunk":         i,
                    })
            except Exception as e:
                print(f"  [WARN] {path.name}: {e}")

    index = pd.DataFrame(records)
    index.to_csv(OUTPUT_DIR / "mel_index.csv", index=False)
    print(f"Saved {len(index)} chunks.")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index_csv = OUTPUT_DIR / "mel_index.csv"

    if index_csv.exists():
        print("Mel cache already exists, skipping.")
    else:
        preprocess_dataset(max_per_species=None)

    print("Preprocessing complete.")
import os, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import librosa
from sklearn.metrics import (
    average_precision_score, precision_score, recall_score, f1_score
)
warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"font.size": 11})


class CFG:
    BASE_DIR    = Path("/kaggle/input/datasets/nur373/birdclef-2023/birdclef-2023")
    AUDIO_DIR   = BASE_DIR / "train_audio"
    OUTPUT_DIR  = Path("/kaggle/working/mel_spectrograms")
    CKPT_DIR    = Path("/kaggle/working/checkpoints")
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
    MODEL_NAME  = "efficientnet_b0"
    PRETRAINED  = False
    PADDING_FACTOR = 5


class BirdEfficientNet(nn.Module):
    def __init__(self, n_classes, model_name=CFG.MODEL_NAME):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=CFG.PRETRAINED, in_chans=3)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def audio_chunk_to_mel(chunk):
    mel = librosa.feature.melspectrogram(
        y=chunk, sr=CFG.SR, n_fft=CFG.N_FFT,
        hop_length=CFG.HOP_LENGTH, win_length=CFG.WIN_LENGTH,
        n_mels=CFG.N_MELS, fmin=CFG.FMIN, fmax=CFG.FMAX, power=2.0)
    mel = np.maximum(mel, 1e-10)
    mel_db = 10.0 * np.log10(mel)
    mel_db -= np.max(mel_db)
    return np.maximum(mel_db, -CFG.TOP_DB)


def load_and_chunk(filepath):
    y, orig_sr = librosa.load(filepath, sr=None, mono=True)
    if orig_sr != CFG.SR:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=CFG.SR)
    y = y.astype(np.float32)
    rem = len(y) % CFG.CHUNK_SAMP
    if rem:
        y = np.pad(y, (0, CFG.CHUNK_SAMP - rem))
    return [y[i:i + CFG.CHUNK_SAMP] for i in range(0, len(y), CFG.CHUNK_SAMP)]


def padded_cmAP(y_true, y_pred, padding_factor=CFG.PADDING_FACTOR):
    n_classes = y_true.shape[1]
    pad = np.ones((padding_factor, n_classes), dtype=np.float32)
    yt = np.vstack([pad, y_true])
    yp = np.vstack([pad, y_pred])
    aps = [average_precision_score(yt[:, c], yp[:, c])
           for c in range(n_classes) if yt[:, c].sum() > 0]
    return float(np.mean(aps))


index_df  = pd.read_csv(CFG.OUTPUT_DIR / "mel_index.csv")
species   = sorted(index_df["primary_label"].unique())
label2idx = {sp: i for i, sp in enumerate(species)}
idx2label = {v: k for k, v in label2idx.items()}
n_classes = len(species)

device = torch.device("cuda:0")
model  = BirdEfficientNet(n_classes).to(device)
model.load_state_dict(
    torch.load(CFG.CKPT_DIR / "fold1_best.pt", map_location=device))
model.eval()

oof_preds = np.load(CFG.CKPT_DIR / "oof_preds.npy")
oof_true  = np.load(CFG.CKPT_DIR / "oof_true.npy")
print(f"Model loaded | {n_classes} classes | OOF shape: {oof_preds.shape}")


pcmap = padded_cmAP(oof_true, oof_preds)

threshold = 0.5
y_pred_bin = (oof_preds >= threshold).astype(int)
y_true_bin = oof_true.astype(int)

precision_micro = precision_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
recall_micro    = recall_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
f1_micro        = f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)

precision_macro = precision_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
recall_macro    = recall_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
f1_macro        = f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)

precision_samples = precision_score(y_true_bin, y_pred_bin, average="samples", zero_division=0)
recall_samples    = recall_score(y_true_bin, y_pred_bin, average="samples", zero_division=0)
f1_samples        = f1_score(y_true_bin, y_pred_bin, average="samples", zero_division=0)

print(f"\n{'='*60}")
print(f"  METRICS (threshold = {threshold})")
print(f"{'='*60}")
print(f"  pcmAP (official)     : {pcmap:.4f}")
print(f"{'─'*60}")
print(f"  {'':20s} {'Micro':>10s} {'Macro':>10s} {'Samples':>10s}")
print(f"  {'Precision':20s} {precision_micro:>10.4f} {precision_macro:>10.4f} {precision_samples:>10.4f}")
print(f"  {'Recall':20s} {recall_micro:>10.4f} {recall_macro:>10.4f} {recall_samples:>10.4f}")
print(f"  {'F1 Score':20s} {f1_micro:>10.4f} {f1_macro:>10.4f} {f1_samples:>10.4f}")
print(f"{'='*60}")


print(f"\nBenchmarking inference speed...")

dummy = torch.randn(1, 3, 128, 500).to(device)

with torch.no_grad(), torch.amp.autocast("cuda"):
    for _ in range(10):
        _ = model(dummy)
torch.cuda.synchronize()

n_runs = 100
torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.no_grad(), torch.amp.autocast("cuda"):
    for _ in range(n_runs):
        _ = model(dummy)
torch.cuda.synchronize()
t1 = time.perf_counter()

ms_per_chunk  = (t1 - t0) / n_runs * 1000
chunks_per_sec = n_runs / (t1 - t0)
realtime_factor = CFG.CHUNK_SEC * chunks_per_sec

meta_df = pd.read_csv(CFG.BASE_DIR / "train_metadata.csv")
sample_file = meta_df["filename"].values[0]
sample_path = str(CFG.AUDIO_DIR / sample_file)

torch.cuda.synchronize()
t_e2e_start = time.perf_counter()
chunks = load_and_chunk(sample_path)
for chunk in chunks:
    mel = audio_chunk_to_mel(chunk).astype(np.float32)
    mel = (mel + CFG.TOP_DB) / CFG.TOP_DB
    x = torch.from_numpy(mel).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        _ = model(x)
torch.cuda.synchronize()
t_e2e = time.perf_counter() - t_e2e_start

audio_duration = len(chunks) * CFG.CHUNK_SEC

print(f"\n{'='*60}")
print(f"  INFERENCE SPEED")
print(f"{'='*60}")
print(f"  Model forward pass   : {ms_per_chunk:.2f} ms/chunk")
print(f"  Throughput            : {chunks_per_sec:.1f} chunks/sec")
print(f"  Real-time factor      : {realtime_factor:.0f}x  "
      f"(processes {realtime_factor:.0f}s of audio per 1s wall time)")
print(f"{'─'*60}")
print(f"  End-to-end (load+mel+infer)")
print(f"    File                : {sample_file}")
print(f"    Audio duration      : {audio_duration}s ({len(chunks)} chunks)")
print(f"    Total time          : {t_e2e*1000:.1f} ms")
print(f"    Per chunk (e2e)     : {t_e2e/len(chunks)*1000:.1f} ms")
print(f"{'='*60}")


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        output[0, class_idx].backward()

        grads = self.gradients[0]
        acts  = self.activations[0]
        weights = grads.mean(dim=(1, 2))

        cam = (weights[:, None, None] * acts).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy(), class_idx, torch.sigmoid(output[0]).detach().cpu().numpy()


target_layer = None
for name, module in model.backbone.named_modules():
    if isinstance(module, nn.Conv2d):
        target_layer = module
        target_name  = name

print(f"\nGrad-CAM target: backbone.{target_name}")
grad_cam = GradCAM(model, target_layer)

report_df = pd.DataFrame({
    "species": [idx2label[c] for c in range(n_classes)],
    "n_samples": [int(oof_true[:, c].sum()) for c in range(n_classes)]
})
sample_species = report_df.query("n_samples >= 5").sample(6, random_state=42)["species"].values

fig, axes = plt.subplots(6, 3, figsize=(18, 24))

for row_idx, sp in enumerate(sample_species):
    sp_files = meta_df[meta_df["primary_label"] == sp]["filename"].values
    audio_path = CFG.AUDIO_DIR / sp_files[0]
    if not audio_path.exists():
        continue

    chunk = load_and_chunk(str(audio_path))[0]
    mel   = audio_chunk_to_mel(chunk).astype(np.float32)

    mel_norm = (mel + CFG.TOP_DB) / CFG.TOP_DB
    x = torch.from_numpy(mel_norm).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
    x.requires_grad_(True)
    model.eval()

    cam, pred_idx, probs = grad_cam.generate(x, class_idx=label2idx[sp])

    cam_resized = F.interpolate(
        torch.from_numpy(cam).unsqueeze(0).unsqueeze(0),
        size=mel.shape, mode="bilinear", align_corners=False
    ).squeeze().numpy()

    top3 = np.argsort(probs)[::-1][:3]
    top3_str = ", ".join([f"{idx2label[i]} ({probs[i]:.2f})" for i in top3])

    t = np.arange(len(chunk)) / CFG.SR
    axes[row_idx, 0].plot(t, chunk, color="#1976D2", linewidth=0.3)
    axes[row_idx, 0].set_title(f"{sp}", fontsize=11, fontweight="bold")
    axes[row_idx, 0].set_xlabel("Time (s)")
    axes[row_idx, 0].set_ylabel("Amplitude")
    axes[row_idx, 0].set_xlim(0, CFG.CHUNK_SEC)

    axes[row_idx, 1].imshow(mel, aspect="auto", origin="lower",
                             cmap="magma", interpolation="nearest")
    axes[row_idx, 1].set_title("Mel Spectrogram")
    axes[row_idx, 1].set_xlabel("Time Frame")
    axes[row_idx, 1].set_ylabel("Mel Bin")

    axes[row_idx, 2].imshow(mel, aspect="auto", origin="lower",
                             cmap="gray", interpolation="nearest", alpha=0.6)
    axes[row_idx, 2].imshow(cam_resized, aspect="auto", origin="lower",
                             cmap="jet", interpolation="bilinear",
                             alpha=0.5, vmin=0, vmax=1)
    axes[row_idx, 2].set_title(f"Grad-CAM | Top3: {top3_str}", fontsize=9)
    axes[row_idx, 2].set_xlabel("Time Frame")
    axes[row_idx, 2].set_ylabel("Mel Bin")

plt.suptitle("Grad-CAM: Where the Model Looks for Each Species",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("/kaggle/working/gradcam_samples.png", dpi=150, bbox_inches="tight")
plt.show()


def gradcam_single(audio_path, target_species=None):
    chunks = load_and_chunk(audio_path)
    n = len(chunks)
    fig, axes = plt.subplots(n, 3, figsize=(18, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, chunk in enumerate(chunks):
        mel = audio_chunk_to_mel(chunk).astype(np.float32)
        mel_norm = (mel + CFG.TOP_DB) / CFG.TOP_DB
        x = torch.from_numpy(mel_norm).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
        x.requires_grad_(True)

        cls_idx = label2idx[target_species] if target_species else None
        cam, _, probs = grad_cam.generate(x, class_idx=cls_idx)

        cam_resized = F.interpolate(
            torch.from_numpy(cam).unsqueeze(0).unsqueeze(0),
            size=mel.shape, mode="bilinear", align_corners=False
        ).squeeze().numpy()

        top3 = np.argsort(probs)[::-1][:3]
        top3_str = ", ".join([f"{idx2label[j]} ({probs[j]:.2f})" for j in top3])

        t = np.arange(len(chunk)) / CFG.SR
        axes[i, 0].plot(t, chunk, color="#1976D2", linewidth=0.4)
        axes[i, 0].set_title(f"Chunk {i} | {i*CFG.CHUNK_SEC}–{(i+1)*CFG.CHUNK_SEC}s")
        axes[i, 0].set_xlim(0, CFG.CHUNK_SEC)

        axes[i, 1].imshow(mel, aspect="auto", origin="lower", cmap="magma")
        axes[i, 1].set_title("Mel Spectrogram")

        axes[i, 2].imshow(mel, aspect="auto", origin="lower", cmap="gray", alpha=0.6)
        axes[i, 2].imshow(cam_resized, aspect="auto", origin="lower",
                          cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[i, 2].set_title(f"Grad-CAM | {top3_str}", fontsize=9)

    plt.suptitle(f"{Path(audio_path).stem}" +
                 (f" (target: {target_species})" if target_species else ""),
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
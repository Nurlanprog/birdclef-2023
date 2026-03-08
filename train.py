import os, warnings, math, time, gc
import numpy as np
import pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")


class CFG:
    BASE_DIR    = Path("/kaggle/input/birdclef-2023")
    TRAIN_CSV   = BASE_DIR / "train_metadata.csv"
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
    PRETRAINED  = True
    N_FOLDS     = 5
    EPOCHS      = 30
    BATCH_SIZE  = 64
    NUM_WORKERS = 2
    LR          = 1e-3
    MIN_LR      = 1e-6
    WEIGHT_DECAY= 1e-2
    LABEL_SMOOTH= 0.05
    MIXUP_ALPHA = 0.4
    GRAD_CLIP   = 5.0
    SEED        = 42

    USE_SPECAUGMENT = True
    FREQ_MASK_PARAM = 15
    TIME_MASK_PARAM = 30
    N_FREQ_MASKS    = 2
    N_TIME_MASKS    = 2

    PADDING_FACTOR  = 5

    NUM_GPUS    = 2
    MIXED_PRECISION = "fp16"


def seed_everything(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

seed_everything(CFG.SEED)
CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CFG.CKPT_DIR.mkdir(parents=True, exist_ok=True)


import librosa


def audio_chunk_to_mel(chunk: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=chunk,
        sr=CFG.SR,
        n_fft=CFG.N_FFT,
        hop_length=CFG.HOP_LENGTH,
        win_length=CFG.WIN_LENGTH,
        n_mels=CFG.N_MELS,
        fmin=CFG.FMIN,
        fmax=CFG.FMAX,
        power=2.0,
    )
    mel = np.maximum(mel, 1e-10)
    mel_db = 10.0 * np.log10(mel)
    mel_db -= np.max(mel_db)
    mel_db = np.maximum(mel_db, -CFG.TOP_DB)
    return mel_db


def load_and_chunk(filepath: str) -> list:
    y, orig_sr = librosa.load(filepath, sr=None, mono=True)
    if orig_sr != CFG.SR:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=CFG.SR)
    y = y.astype(np.float32)
    rem = len(y) % CFG.CHUNK_SAMP
    if rem:
        y = np.pad(y, (0, CFG.CHUNK_SAMP - rem))
    return [y[i:i + CFG.CHUNK_SAMP] for i in range(0, len(y), CFG.CHUNK_SAMP)]


from tqdm import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate import notebook_launcher


class BirdDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label2idx: dict, train: bool = True):
        self.paths     = df["npy_path"].values
        self.labels    = df["primary_label"].values
        self.label2idx = label2idx
        self.train     = train
        self.n_classes = len(label2idx)

    def _specaugment(self, mel: torch.Tensor) -> torch.Tensor:
        _, F_dim, T_dim = mel.shape
        for _ in range(CFG.N_FREQ_MASKS):
            f0 = np.random.randint(0, max(F_dim - CFG.FREQ_MASK_PARAM, 1))
            mel[:, f0:f0 + CFG.FREQ_MASK_PARAM, :] = 0
        for _ in range(CFG.N_TIME_MASKS):
            t0 = np.random.randint(0, max(T_dim - CFG.TIME_MASK_PARAM, 1))
            mel[:, :, t0:t0 + CFG.TIME_MASK_PARAM] = 0
        return mel

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        mel = np.load(self.paths[idx]).astype(np.float32)
        mel = (mel + CFG.TOP_DB) / CFG.TOP_DB
        mel = torch.from_numpy(mel).unsqueeze(0)
        mel = mel.repeat(3, 1, 1)

        if self.train and CFG.USE_SPECAUGMENT:
            mel = self._specaugment(mel)

        label = torch.zeros(self.n_classes, dtype=torch.float32)
        label[self.label2idx[self.labels[idx]]] = 1.0
        return mel, label


def mixup_data(x, y, alpha=CFG.MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    bs  = x.size(0)
    idx = torch.randperm(bs, device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


class BirdEfficientNet(nn.Module):
    def __init__(self, n_classes: int, model_name: str = CFG.MODEL_NAME):
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
        feat = self.backbone(x)
        return self.head(feat)


class SmoothBCELoss(nn.Module):
    def __init__(self, smoothing=CFG.LABEL_SMOOTH):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, targets)


def mixup_loss(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


def padded_cmAP(y_true: np.ndarray, y_pred: np.ndarray,
                padding_factor: int = CFG.PADDING_FACTOR) -> float:
    n_classes = y_true.shape[1]
    pad_true  = np.ones((padding_factor, n_classes), dtype=np.float32)
    pad_pred  = np.ones((padding_factor, n_classes), dtype=np.float32)

    y_true_pad = np.vstack([pad_true, y_true])
    y_pred_pad = np.vstack([pad_pred, y_pred])

    per_class_ap = []
    for c in range(n_classes):
        if y_true_pad[:, c].sum() == 0:
            continue
        ap = average_precision_score(y_true_pad[:, c], y_pred_pad[:, c])
        per_class_ap.append(ap)

    return float(np.mean(per_class_ap))


def train_one_epoch(model, loader, optimizer, criterion, accelerator, scheduler):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc="  Train", leave=False,
                disable=not accelerator.is_main_process)
    for x, y in pbar:
        x, y_a, y_b, lam = mixup_data(x, y)

        optimizer.zero_grad()
        logits = model(x)
        loss   = mixup_loss(criterion, logits, y_a, y_b, lam)

        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, accelerator):
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="  Val  ", leave=False,
                disable=not accelerator.is_main_process)
    for x, y in pbar:
        logits = model(x)
        loss   = criterion(logits, y)
        total_loss += loss.item()
        n_batches  += 1

        probs = torch.sigmoid(logits)

        all_probs, all_y = accelerator.gather_for_metrics((probs, y))
        all_preds.append(all_probs.cpu().numpy())
        all_labels.append(all_y.cpu().numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    pcmap      = padded_cmAP(all_labels, all_preds)

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, pcmap, all_preds, all_labels


def train_kfold(index_df: pd.DataFrame, accelerator: Accelerator):
    species   = sorted(index_df["primary_label"].unique())
    label2idx = {sp: i for i, sp in enumerate(species)}
    n_classes = len(species)

    if accelerator.is_main_process:
        print(f"\nClasses: {n_classes}  |  Total chunks: {len(index_df)}")

    file_df = (index_df.groupby("filename")["primary_label"]
                       .first().reset_index())

    skf       = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True,
                                random_state=CFG.SEED)
    criterion = SmoothBCELoss()
    fold_scores = []

    for fold, (train_files, val_files) in enumerate(
            skf.split(file_df["filename"], file_df["primary_label"])):

        if accelerator.is_main_process:
            print(f"\n{'='*60}")
            print(f"  FOLD {fold + 1}/{CFG.N_FOLDS}")
            print(f"{'='*60}")

        train_fns = set(file_df.iloc[train_files]["filename"])
        val_fns   = set(file_df.iloc[val_files]["filename"])

        train_df = index_df[index_df["filename"].isin(train_fns)]
        val_df   = index_df[index_df["filename"].isin(val_fns)]

        train_ds = BirdDataset(train_df, label2idx, train=True)
        val_ds   = BirdDataset(val_df,   label2idx, train=False)

        train_loader = DataLoader(
            train_ds,
            batch_size=CFG.BATCH_SIZE,
            shuffle=True,
            num_workers=CFG.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if CFG.NUM_WORKERS > 0 else False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=CFG.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=CFG.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if CFG.NUM_WORKERS > 0 else False,
        )

        model = BirdEfficientNet(n_classes)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)

        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader
        )

        total_steps = CFG.EPOCHS * len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=CFG.MIN_LR)

        best_score, best_epoch = 0.0, 0
        ckpt_path = CFG.CKPT_DIR / f"fold{fold+1}_best.pt"

        for epoch in range(CFG.EPOCHS):
            t0 = time.time()

            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, accelerator, scheduler)
            val_loss, pcmap, preds, labels = validate(
                model, val_loader, criterion, accelerator)

            elapsed = time.time() - t0
            lr_now  = optimizer.param_groups[0]["lr"]

            if accelerator.is_main_process:
                print(f"  Ep {epoch+1:02d}/{CFG.EPOCHS}  "
                      f"loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                      f"pcmAP {pcmap:.4f} | lr {lr_now:.2e} | {elapsed:.0f}s")

                if pcmap > best_score:
                    best_score, best_epoch = pcmap, epoch + 1
                    unwrapped = accelerator.unwrap_model(model)
                    accelerator.save(unwrapped.state_dict(), ckpt_path)
                    print(f"    ✓ New best pcmAP={best_score:.4f} saved.")

            accelerator.wait_for_everyone()

        accelerator.wait_for_everyone()

        unwrapped = accelerator.unwrap_model(model)
        if ckpt_path.exists():
            unwrapped.load_state_dict(
                torch.load(ckpt_path, map_location=accelerator.device))

        _, _, oof_p, oof_t = validate(model, val_loader, criterion, accelerator)

        if accelerator.is_main_process:
            np.save(CFG.CKPT_DIR / f"fold{fold+1}_oof_preds.npy", oof_p)
            np.save(CFG.CKPT_DIR / f"fold{fold+1}_oof_true.npy",  oof_t)
            fold_scores.append(best_score)
            print(f"  Fold {fold+1} best pcmAP = {best_score:.4f} "
                  f"(epoch {best_epoch})")

        del model, optimizer, train_loader, val_loader, scheduler
        gc.collect()
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        all_oof_preds = []
        all_oof_true  = []
        for f in range(CFG.N_FOLDS):
            p = np.load(CFG.CKPT_DIR / f"fold{f+1}_oof_preds.npy")
            t = np.load(CFG.CKPT_DIR / f"fold{f+1}_oof_true.npy")
            all_oof_preds.append(p)
            all_oof_true.append(t)
        oof_preds = np.vstack(all_oof_preds)
        oof_true  = np.vstack(all_oof_true)

        oof_score = padded_cmAP(oof_true, oof_preds)
        print(f"\n{'='*60}")
        print(f"  OOF pcmAP  : {oof_score:.4f}")
        print(f"  Fold scores: {[f'{s:.4f}' for s in fold_scores]}")
        print(f"  Mean       : {np.mean(fold_scores):.4f} "
              f"± {np.std(fold_scores):.4f}")
        print(f"{'='*60}")

        np.save(CFG.CKPT_DIR / "oof_preds.npy",  oof_preds)
        np.save(CFG.CKPT_DIR / "oof_true.npy",   oof_true)

    accelerator.wait_for_everyone()
    return label2idx, fold_scores


def evaluate_oof(label2idx: dict):
    oof_preds = np.load(CFG.CKPT_DIR / "oof_preds.npy")
    oof_true  = np.load(CFG.CKPT_DIR / "oof_true.npy")
    idx2label = {v: k for k, v in label2idx.items()}
    n_classes = len(label2idx)

    pad_true = np.ones((CFG.PADDING_FACTOR, n_classes), dtype=np.float32)
    pad_pred = np.ones((CFG.PADDING_FACTOR, n_classes), dtype=np.float32)
    yt_pad   = np.vstack([pad_true, oof_true])
    yp_pad   = np.vstack([pad_pred, oof_preds])

    rows = []
    for c in range(n_classes):
        ap = average_precision_score(yt_pad[:, c], yp_pad[:, c])
        n  = int(oof_true[:, c].sum())
        rows.append({"species": idx2label[c], "AP": ap, "n_samples": n})

    report  = pd.DataFrame(rows).sort_values("AP")
    overall = padded_cmAP(oof_true, oof_preds)

    print(f"\n── OOF Evaluation Report ──────────────────────────")
    print(f"  Overall pcmAP : {overall:.4f}")
    print(f"\n  Worst 10 species:")
    print(report.head(10).to_string(index=False))
    print(f"\n  Best 10 species:")
    print(report.tail(10).to_string(index=False))

    report.to_csv(CFG.CKPT_DIR / "per_species_ap.csv", index=False)
    print(f"\n  Full report saved → {CFG.CKPT_DIR}/per_species_ap.csv")
    return report


@torch.no_grad()
def predict_soundscape(audio_path: str, model: nn.Module,
                       label2idx: dict, device) -> pd.DataFrame:
    idx2label = {v: k for k, v in label2idx.items()}
    model.eval()
    chunks = load_and_chunk(audio_path)
    rows   = []
    stem   = Path(audio_path).stem

    for i, chunk in enumerate(chunks):
        mel = audio_chunk_to_mel(chunk).astype(np.float32)
        mel = (mel + CFG.TOP_DB) / CFG.TOP_DB
        x   = torch.from_numpy(mel).unsqueeze(0).repeat(3, 1, 1)
        x   = x.unsqueeze(0).to(device)

        with torch.amp.autocast("cuda"):
            logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

        t_start = i * CFG.CHUNK_SEC
        row     = {"row_id": f"{stem}_{t_start}"}
        row.update({idx2label[j]: float(probs[j])
                    for j in range(len(idx2label))})
        rows.append(row)

    return pd.DataFrame(rows)


def training_function():
    accelerator = Accelerator(
        mixed_precision=CFG.MIXED_PRECISION,
    )

    set_seed(CFG.SEED)

    if accelerator.is_main_process:
        print(f"Accelerator: {accelerator.distributed_type}, "
              f"device={accelerator.device}, "
              f"num_processes={accelerator.num_processes}, "
              f"mixed_precision={accelerator.mixed_precision}")

    index_csv = CFG.OUTPUT_DIR / "mel_index.csv"
    if not index_csv.exists():
        raise FileNotFoundError(
            f"Mel cache not found at {index_csv}. "
            "Run preprocessing (Part A) first!")
    index_df = pd.read_csv(index_csv)

    label2idx, fold_scores = train_kfold(index_df, accelerator)

    if accelerator.is_main_process:
        report = evaluate_oof(label2idx)

    accelerator.wait_for_everyone()



if __name__ == "__main__":
    import subprocess, sys

    index_csv = CFG.OUTPUT_DIR / "mel_index.csv"
    if index_csv.exists():
        print("Mel cache found, skipping preprocessing.")
    else:
        print("Running CuPy preprocessing in subprocess...")
        result = subprocess.run(
            [sys.executable, "preprocess_cupy.py"],
            cwd="/kaggle/working"
        )
        if result.returncode != 0:
            raise RuntimeError("Preprocessing failed!")

    notebook_launcher(training_function, num_processes=CFG.NUM_GPUS)

    index_df  = pd.read_csv(index_csv)
    species   = sorted(index_df["primary_label"].unique())
    label2idx = {sp: i for i, sp in enumerate(species)}
    n_classes = len(species)

    device = torch.device("cuda:0")
    model  = BirdEfficientNet(n_classes).to(device)

    best_ckpt = CFG.CKPT_DIR / "fold1_best.pt"
    model.load_state_dict(torch.load(best_ckpt, map_location=device))

    sub_rows = []
    test_dir = CFG.BASE_DIR / "test_soundscapes"
    if test_dir.exists():
        for ogg in sorted(test_dir.glob("*.ogg")):
            df_out = predict_soundscape(str(ogg), model, label2idx, device)
            sub_rows.append(df_out)

        submission = pd.concat(sub_rows, ignore_index=True)
        submission.to_csv("/kaggle/working/submission.csv", index=False)
        print(f"\nSubmission saved: {len(submission)} rows.")
    else:
        print("No test_soundscapes directory found. Skipping inference.")

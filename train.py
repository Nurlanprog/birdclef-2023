%%writefile train_multi_gpu.py

import os, warnings, math, time, gc, random
import numpy as np
import pandas as pd
from pathlib import Path
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
warnings.filterwarnings("ignore")


class CFG:
    BASE_DIR    = Path("/kaggle/input/datasets/nur373/birdclef-2023/birdclef-2023")
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

    MODEL_NAME      = "efficientnet_b0"
    PRETRAINED      = True
    N_FOLDS         = 2
    EPOCHS          = 2
    BATCH_SIZE      = 64
    NUM_WORKERS     = 2
    LR              = 1e-3
    MIN_LR          = 1e-6
    WEIGHT_DECAY    = 1e-2
    LABEL_SMOOTH    = 0.05
    MIXUP_ALPHA     = 0.4
    GRAD_CLIP       = 5.0
    SEED            = 42

    USE_SPECAUGMENT = True
    FREQ_MASK_PARAM = 15
    TIME_MASK_PARAM = 30
    N_FREQ_MASKS    = 2
    N_TIME_MASKS    = 2

    PADDING_FACTOR  = 5


CFG.CKPT_DIR.mkdir(parents=True, exist_ok=True)


class BirdDataset(Dataset):
    def __init__(self, df, label2idx, train=True):
        self.paths     = df["npy_path"].values
        self.labels    = df["primary_label"].values
        self.label2idx = label2idx
        self.train     = train
        self.n_classes = len(label2idx)

    def _specaugment(self, mel):
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
        mel = torch.from_numpy(mel).unsqueeze(0).repeat(3, 1, 1)

        if self.train and CFG.USE_SPECAUGMENT:
            mel = self._specaugment(mel)

        label = torch.zeros(self.n_classes, dtype=torch.float32)
        label[self.label2idx[self.labels[idx]]] = 1.0
        return mel, label


def mixup_data(x, y, alpha=CFG.MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


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


class SmoothBCELoss(nn.Module):
    def __init__(self, smoothing=CFG.LABEL_SMOOTH):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, targets)


def mixup_loss(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


def padded_cmAP(y_true, y_pred, padding_factor=CFG.PADDING_FACTOR):
    n_classes = y_true.shape[1]
    pad = np.ones((padding_factor, n_classes), dtype=np.float32)
    yt  = np.vstack([pad, y_true])
    yp  = np.vstack([pad, y_pred])
    aps = [
        average_precision_score(yt[:, c], yp[:, c])
        for c in range(n_classes) if yt[:, c].sum() > 0
    ]
    return float(np.mean(aps))


def train_one_epoch(model, loader, optimizer, criterion, accelerator, scheduler):
    model.train()
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc="  Train", leave=False,
                disable=not accelerator.is_main_process)
    for x, y in pbar:
        x, y_a, y_b, lam = mixup_data(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss = mixup_loss(criterion, model(x), y_a, y_b, lam)
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, criterion, accelerator):
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc="  Val  ", leave=False,
                disable=not accelerator.is_main_process)
    for x, y in pbar:
        logits = model(x)
        total_loss += criterion(logits, y).item()
        n += 1
        probs = torch.sigmoid(logits)
        all_p, all_y = accelerator.gather_for_metrics((probs, y))
        all_preds.append(all_p.cpu().numpy())
        all_labels.append(all_y.cpu().numpy())
    preds  = np.vstack(all_preds)
    labels = np.vstack(all_labels)
    return total_loss / max(n, 1), padded_cmAP(labels, preds), preds, labels


def train_kfold(index_df, accelerator):
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
            print(f"\n{'='*60}\n  FOLD {fold + 1}/{CFG.N_FOLDS}\n{'='*60}")

        train_fns = set(file_df.iloc[train_files]["filename"])
        val_fns   = set(file_df.iloc[val_files]["filename"])
        train_df  = index_df[index_df["filename"].isin(train_fns)]
        val_df    = index_df[index_df["filename"].isin(val_fns)]

        train_loader = DataLoader(
            BirdDataset(train_df, label2idx, train=True),
            batch_size=CFG.BATCH_SIZE, shuffle=True,
            num_workers=CFG.NUM_WORKERS, pin_memory=True,
            drop_last=True, persistent_workers=CFG.NUM_WORKERS > 0,
        )
        val_loader = DataLoader(
            BirdDataset(val_df, label2idx, train=False),
            batch_size=CFG.BATCH_SIZE * 2, shuffle=False,
            num_workers=CFG.NUM_WORKERS, pin_memory=True,
            drop_last=False, persistent_workers=CFG.NUM_WORKERS > 0,
        )

        model     = BirdEfficientNet(n_classes)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)

        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader)

        total_steps = CFG.EPOCHS * len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=CFG.MIN_LR)

        best_score, best_epoch = 0.0, 0
        ckpt_path = CFG.CKPT_DIR / f"fold{fold+1}_best.pt"

        for epoch in range(CFG.EPOCHS):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, accelerator, scheduler)
            val_loss, pcmap, _, _ = validate(
                model, val_loader, criterion, accelerator)

            if accelerator.is_main_process:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"  Ep {epoch+1:02d}/{CFG.EPOCHS}  "
                      f"loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                      f"pcmAP {pcmap:.4f} | lr {lr_now:.2e} | "
                      f"{time.time()-t0:.0f}s")
                if pcmap > best_score:
                    best_score, best_epoch = pcmap, epoch + 1
                    accelerator.save(
                        accelerator.unwrap_model(model).state_dict(), ckpt_path)
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
        oof_preds = np.vstack([
            np.load(CFG.CKPT_DIR / f"fold{f+1}_oof_preds.npy")
            for f in range(CFG.N_FOLDS)])
        oof_true = np.vstack([
            np.load(CFG.CKPT_DIR / f"fold{f+1}_oof_true.npy")
            for f in range(CFG.N_FOLDS)])

        oof_score = padded_cmAP(oof_true, oof_preds)
        print(f"\n{'='*60}")
        print(f"  OOF pcmAP  : {oof_score:.4f}")
        print(f"  Fold scores: {[f'{s:.4f}' for s in fold_scores]}")
        print(f"  Mean       : {np.mean(fold_scores):.4f} "
              f"± {np.std(fold_scores):.4f}")
        print(f"{'='*60}")
        np.save(CFG.CKPT_DIR / "oof_preds.npy", oof_preds)
        np.save(CFG.CKPT_DIR / "oof_true.npy",  oof_true)

    accelerator.wait_for_everyone()
    return label2idx, fold_scores


def evaluate_oof(label2idx):
    oof_preds = np.load(CFG.CKPT_DIR / "oof_preds.npy")
    oof_true  = np.load(CFG.CKPT_DIR / "oof_true.npy")
    idx2label = {v: k for k, v in label2idx.items()}
    n_classes = len(label2idx)
    pad = np.ones((CFG.PADDING_FACTOR, n_classes), dtype=np.float32)
    yt  = np.vstack([pad, oof_true])
    yp  = np.vstack([pad, oof_preds])

    rows = [{"species": idx2label[c],
             "AP": average_precision_score(yt[:, c], yp[:, c]),
             "n_samples": int(oof_true[:, c].sum())}
            for c in range(n_classes)]
    report  = pd.DataFrame(rows).sort_values("AP")
    overall = padded_cmAP(oof_true, oof_preds)

    print(f"\n── OOF Evaluation Report ──────────────────────────")
    print(f"  Overall pcmAP : {overall:.4f}")
    print(f"\n  Worst 10:\n{report.head(10).to_string(index=False)}")
    print(f"\n  Best 10:\n{report.tail(10).to_string(index=False)}")
    report.to_csv(CFG.CKPT_DIR / "per_species_ap.csv", index=False)
    print(f"\n  Full report → {CFG.CKPT_DIR}/per_species_ap.csv")
    return report


def main():
    accelerator = Accelerator(mixed_precision="fp16")
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
            "Run preprocessing first!")

    index_df = pd.read_csv(index_csv)
    label2idx, fold_scores = train_kfold(index_df, accelerator)

    if accelerator.is_main_process:
        evaluate_oof(label2idx)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
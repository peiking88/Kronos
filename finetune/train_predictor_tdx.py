#!/usr/bin/env python3
"""
Single-GPU fine-tuning script for Kronos Predictor using TDX local data.

全参数微调，让模型适配 A 股分布

Uses bf16 AMP (RTX 5080 native bf16 — no GradScaler needed).

Usage:
    python finetune/train_predictor_tdx.py --data-dir data/tdx_import/1d
"""

import os
import sys
import json
import time
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune.config_tdx import TdxFineTuneConfig
from finetune.dataset import QlibDataset
from model.kronos import KronosTokenizer, Kronos


# ---------------------------------------------------------------------------
# Custom collate: 避免 PyTorch 2.12 多进程 storage resize bug
def fast_collate(batch):
    xs, stamps = zip(*batch)
    return (
        torch.stack(xs),
        torch.stack(stamps),
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def create_dataloaders(config: dict, config_obj, data_dir: str):
    """Create train/val dataloaders for predictor training."""
    config['dataset_path'] = data_dir
    config_obj.dataset_path = data_dir

    train_dataset = QlibDataset('train', config=config_obj)
    valid_dataset = QlibDataset('val', config=config_obj)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['predictor_batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=fast_collate,
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=config['predictor_batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=fast_collate,
    )
    return train_loader, val_loader, train_dataset, valid_dataset


def train_phase_full(config: dict, config_obj, data_dir: str, device: torch.device):
    """全参数微调 Kronos Predictor。"""

    epochs = config.get('phase1_epochs', 10)
    use_amp = config.get('use_amp', True)
    save_dir = os.path.join(config['save_path'], config['predictor_save_folder_name'])
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

    # Load models
    tokenizer = KronosTokenizer.from_pretrained(config['finetuned_tokenizer_path'])
    tokenizer.eval().to(device)

    model = Kronos.from_pretrained(config['pretrained_predictor_path'])
    model.train().to(device)

    model_size = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Predictor parameters: {model_size:.1f}M")

    # Data
    train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(
        config, config_obj, data_dir
    )

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['predictor_learning_rate'],
        betas=(config['adam_beta1'], config['adam_beta2']),
        weight_decay=config['adam_weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['predictor_learning_rate'],
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.03,
        div_factor=10,
    )

    best_val_loss = float('inf')
    patience = config.get('early_stop_patience', 5)
    no_improve = 0
    global_step = 0
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        train_dataset.set_epoch_seed(epoch * 10000)
        train_loss_total = 0.0
        train_batches = 0

        for batch_x, batch_x_stamp in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                loss, _, _ = model.head.compute_loss(
                    logits[0], logits[1], token_out[0], token_out[1]
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()

            train_loss_total += loss.item()
            train_batches += 1
            global_step += 1

            if global_step % config.get('log_interval', 50) == 0:
                lr = optimizer.param_groups[0]['lr']
                avg = train_loss_total / train_batches
                print(
                    f"E{epoch+1:2d}/{epochs} "
                    f"B{train_batches:4d}/{len(train_loader)}] "
                    f"LR {lr:.6f}  Loss {avg:.4f}"
                )

        # --- Validation ---
        model.eval()
        val_loss_total = 0.0
        val_samples = 0
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            for batch_x, batch_x_stamp in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                val_loss, _, _ = model.head.compute_loss(
                    logits[0], logits[1], token_out[0], token_out[1]
                )
                val_loss_total += val_loss.item() * batch_x.size(0)
                val_samples += batch_x.size(0)

        avg_val_loss = val_loss_total / val_samples if val_samples > 0 else float('inf')
        elapsed = time.time() - epoch_start

        print(
            f"--- Epoch {epoch+1}/{epochs} "
            f"Train Loss {train_loss_total/train_batches:.4f}  "
            f"Val Loss {avg_val_loss:.4f}  "
            f"Time {elapsed/60:.1f}m ---"
        )

        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
            save_path = f"{save_dir}/checkpoints/best_model"
            model.save_pretrained(save_path)
            print(f"Best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop at epoch {epoch+1}")
                break

    # Save summary
    summary = {
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch + 1,
        'total_time_s': time.time() - start_time,
        'model_params_m': model_size,
    }
    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Training complete. Best Val Loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune Kronos Predictor on TDX data')
    parser.add_argument('--data-dir', default='./data/tdx_import/1d',
                        help='Path to processed data directory')
    parser.add_argument('--device', default='cuda:0',
                        help='Device for training')
    parser.add_argument('--epochs', type=int, default=0,
                        help='Override epochs (0 = use config default)')
    parser.add_argument('--tokenizer-path', default=None,
                        help='Override fine-tuned tokenizer path')
    parser.add_argument('--predictor-path', default=None,
                        help='Override fine-tuned predictor path (Phase 2)')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable bf16 AMP')
    args = parser.parse_args()

    config = TdxFineTuneConfig()
    cfg = config.to_dict()

    if args.epochs > 0:
        cfg['epochs'] = args.epochs
    if args.tokenizer_path:
        cfg['finetuned_tokenizer_path'] = args.tokenizer_path
    if args.predictor_path:
        cfg['finetuned_predictor_path'] = args.predictor_path
    if args.no_amp:
        cfg['use_amp'] = False

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Batch size: {cfg['predictor_batch_size']}, AMP: {cfg['use_amp']}")
    print(f"Epochs: {cfg.get('phase1_epochs', 10)}")
    print(f"Pretrained predictor: {cfg['pretrained_predictor_path']}")
    print(f"Finetuned tokenizer: {cfg['finetuned_tokenizer_path']}")
    train_phase_full(cfg, config, args.data_dir, device)


if __name__ == '__main__':
    main()

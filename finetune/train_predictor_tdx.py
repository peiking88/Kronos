#!/usr/bin/env python3
"""
Single-GPU fine-tuning script for Kronos Predictor using TDX local data.

Uses AMP (fp16) to fit batch_size=12 on RTX 4060 8GB.
Gradient accumulation simulates larger effective batch size.

Usage:
    python finetune/train_predictor_tdx.py [--data-dir DATA_DIR] [--device cuda:0]
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune.config_tdx import TdxFineTuneConfig
from finetune.dataset import QlibDataset
from model.kronos import KronosTokenizer, Kronos


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def create_dataloaders(config: dict, config_obj, data_dir: str):
    """Create train/val dataloaders for single-GPU predictor training."""
    config['dataset_path'] = data_dir
    config_obj.dataset_path = data_dir

    train_dataset = QlibDataset('train', config=config_obj)
    valid_dataset = QlibDataset('val', config=config_obj)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['predictor_batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 2),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=config['predictor_batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 2),
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader, train_dataset, valid_dataset


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: dict, config_obj, data_dir: str, device: torch.device):
    save_dir = os.path.join(config['save_path'], config['predictor_save_folder_name'])
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

    train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(
        config, config_obj, data_dir
    )

    # Load tokenizer (frozen) and predictor (trainable)
    print("Loading tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained(config['finetuned_tokenizer_path'])
    tokenizer.eval().to(device)
    for p in tokenizer.parameters():
        p.requires_grad = False

    print("Loading predictor...")
    model = Kronos.from_pretrained(config['pretrained_predictor_path'])
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Predictor parameters: {n_params:,}")

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
        epochs=config['epochs'],
        pct_start=0.03,
        div_factor=10,
    )

    use_amp = config.get('use_amp', True) and device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_loss = float('inf')
    start_time = time.time()
    accum_steps = config.get('predictor_accumulation', 1)

    for epoch_idx in range(config['epochs']):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        train_dataset.set_epoch_seed(epoch_idx * 10000)
        train_loss_total = 0.0
        train_batches = 0
        optimizer.zero_grad()

        for batch_idx, (batch_x, batch_x_stamp) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

            # Tokenize (no grad, frozen tokenizer)
            with torch.no_grad():
                tok0, tok1 = tokenizer.encode(batch_x, half=True)

            token_in = [tok0[:, :-1], tok1[:, :-1]]
            token_out = [tok0[:, 1:], tok1[:, 1:]]
            stamp_in = batch_x_stamp[:, :-1, :]

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                logits = model(token_in[0], token_in[1], stamp_in)
                loss, s1_loss, s2_loss = model.head.compute_loss(
                    logits[0], logits[1], token_out[0], token_out[1]
                )
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            # Step every accum_steps batches
            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            train_loss_total += loss.item() * accum_steps
            train_batches += 1

            if (batch_idx + 1) % config['log_interval'] == 0:
                lr = optimizer.param_groups[0]['lr']
                avg = train_loss_total / train_batches
                print(f"[E{epoch_idx+1:2d}/{config['epochs']} "
                      f"B{batch_idx+1:4d}/{len(train_loader)}] "
                      f"LR {lr:.6f}  Loss {avg:.4f}  "
                      f"S1 {s1_loss.item():.4f}  S2 {s2_loss.item():.4f}")

        # --- Validation ---
        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

                tok0, tok1 = tokenizer.encode(batch_x, half=True)
                token_in = [tok0[:, :-1], tok1[:, :-1]]
                token_out = [tok0[:, 1:], tok1[:, 1:]]
                stamp_in = batch_x_stamp[:, :-1, :]

                logits = model(token_in[0], token_in[1], stamp_in)
                val_loss, _, _ = model.head.compute_loss(
                    logits[0], logits[1], token_out[0], token_out[1]
                )
                val_loss_total += val_loss.item()
                val_batches += 1

        avg_val_loss = val_loss_total / val_batches if val_batches > 0 else float('inf')
        elapsed = time.time() - epoch_start

        print(f"--- Epoch {epoch_idx+1}/{config['epochs']} "
              f"| Train Loss: {train_loss_total/train_batches:.4f} "
              f"| Val Loss: {avg_val_loss:.4f} "
              f"| Time: {elapsed/60:.1f}m ---")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(save_dir, 'checkpoints', 'best_model')
            model.save_pretrained(ckpt_path)
            print(f"  -> Best model saved to {ckpt_path} (val_loss={best_val_loss:.4f})")

    total_time = time.time() - start_time
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Total time: {total_time/3600:.1f}h")
    print(f"Model saved to: {save_dir}/checkpoints/best_model")

    summary = {
        'best_val_loss': best_val_loss,
        'total_time_h': total_time / 3600,
        'epochs': config['epochs'],
        'n_params': n_params,
        'amp': use_amp,
        'batch_size': config['predictor_batch_size'],
        'accumulation': accum_steps,
    }
    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return best_val_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable AMP mixed precision')
    args = parser.parse_args()

    config = TdxFineTuneConfig()
    cfg = config.to_dict()

    if args.epochs > 0:
        cfg['epochs'] = args.epochs
    if args.tokenizer_path:
        cfg['finetuned_tokenizer_path'] = args.tokenizer_path
    if args.no_amp:
        cfg['use_amp'] = False

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Epochs: {cfg['epochs']}, Batch size: {cfg['predictor_batch_size']}")
    print(f"Accumulation: {cfg['predictor_accumulation']}, AMP: {cfg['use_amp']}")
    print(f"Pretrained predictor: {cfg['pretrained_predictor_path']}")
    print(f"Finetuned tokenizer: {cfg['finetuned_tokenizer_path']}")

    train(cfg, config, args.data_dir, device)


if __name__ == '__main__':
    main()

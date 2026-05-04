#!/usr/bin/env python3
"""
Single-GPU fine-tuning script for KronosTokenizer using TDX local data.

Usage:
    python finetune/train_tokenizer_tdx.py [--config CONFIG] [--device cuda:0]
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
from model.kronos import KronosTokenizer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def create_dataloaders(config, config_obj, data_dir: str):
    """Create train/val dataloaders for single-GPU training."""
    config['dataset_path'] = data_dir
    config_obj.dataset_path = data_dir

    train_dataset = QlibDataset('train', config=config_obj)
    valid_dataset = QlibDataset('val', config=config_obj)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 2),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=config['batch_size'],
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
    save_dir = os.path.join(config['save_path'], config['tokenizer_save_folder_name'])
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

    train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(
        config, config_obj, data_dir
    )

    model = KronosTokenizer.from_pretrained(config['pretrained_tokenizer_path'])
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Tokenizer parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['tokenizer_learning_rate'],
        weight_decay=config['adam_weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['tokenizer_learning_rate'],
        steps_per_epoch=len(train_loader),
        epochs=config['epochs'],
        pct_start=0.03,
        div_factor=10,
    )

    best_val_loss = float('inf')
    start_time = time.time()

    for epoch_idx in range(config['epochs']):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        train_dataset.set_epoch_seed(epoch_idx * 10000)
        train_loss_total = 0.0
        train_batches = 0

        for batch_idx, (batch_x, _) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)

            # Gradient accumulation
            accum_loss = 0.0
            for j in range(config['accumulation_steps']):
                n = batch_x.shape[0] // config['accumulation_steps']
                sub_x = batch_x[j * n:(j + 1) * n]

                zs, bsq_loss, _, _ = model(sub_x)
                z_pre, z = zs

                recon_loss_pre = F.mse_loss(z_pre, sub_x)
                recon_loss_all = F.mse_loss(z, sub_x)
                recon_loss = recon_loss_pre + recon_loss_all
                loss = (recon_loss + bsq_loss) / 2
                (loss / config['accumulation_steps']).backward()
                accum_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_loss_total += accum_loss / config['accumulation_steps']
            train_batches += 1

            if (batch_idx + 1) % config['log_interval'] == 0:
                lr = optimizer.param_groups[0]['lr']
                avg = train_loss_total / train_batches
                print(f"[E{epoch_idx+1:2d}/{config['epochs']} "
                      f"B{batch_idx+1:4d}/{len(train_loader)}] "
                      f"LR {lr:.6f}  Loss {avg:.4f}")

        # --- Validation ---
        model.eval()
        val_loss_total = 0.0
        val_samples = 0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                zs, _, _, _ = model(batch_x)
                _, z = zs
                loss = F.mse_loss(z, batch_x)
                val_loss_total += loss.item() * batch_x.size(0)
                val_samples += batch_x.size(0)

        avg_val_loss = val_loss_total / val_samples if val_samples > 0 else float('inf')
        elapsed = time.time() - epoch_start

        print(f"--- Epoch {epoch_idx+1}/{config['epochs']} "
              f"| Train Loss: {train_loss_total/train_batches:.4f} "
              f"| Val Loss: {avg_val_loss:.4f} "
              f"| Time: {elapsed/60:.1f}m ---")

        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(save_dir, 'checkpoints', 'best_model')
            model.save_pretrained(ckpt_path)
            print(f"  -> Best model saved to {ckpt_path} (val_loss={best_val_loss:.4f})")

    total_time = time.time() - start_time
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Total time: {total_time/3600:.1f}h")
    print(f"Model saved to: {save_dir}/checkpoints/best_model")

    # Save summary
    summary = {
        'best_val_loss': best_val_loss,
        'total_time_h': total_time / 3600,
        'epochs': config['epochs'],
        'n_params': n_params,
    }
    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return best_val_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Fine-tune KronosTokenizer on TDX data')
    parser.add_argument('--data-dir', default='./data/tdx_import/1d',
                        help='Path to processed data directory')
    parser.add_argument('--device', default='cuda:0',
                        help='Device for training')
    parser.add_argument('--epochs', type=int, default=0,
                        help='Override epochs (0 = use config default)')
    args = parser.parse_args()

    config = TdxFineTuneConfig()
    cfg = config.to_dict()

    if args.epochs > 0:
        cfg['epochs'] = args.epochs

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Epochs: {cfg['epochs']}, Batch size: {cfg['batch_size']}")
    print(f"Pretrained: {cfg['pretrained_tokenizer_path']}")

    train(cfg, config, args.data_dir, device)


if __name__ == '__main__':
    main()

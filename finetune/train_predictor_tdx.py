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
    xs, stamps, covs = zip(*batch)
    return (
        torch.stack(xs),
        torch.stack(stamps),
        torch.stack(covs),
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

#!/usr/bin/env python3
"""
Single-GPU fine-tuning script for Kronos Predictor using TDX local data.

两阶段训练模式:
    Phase 1 (--phase full):  全参数微调，无 IIB/CZSC，让模型适配 A 股分布
    Phase 2 (--phase iib):   IIB + CZSC 训练，渐进式解冻

Uses bf16 AMP (RTX 5080 native bf16 — no GradScaler needed).

Usage:
    python finetune/train_predictor_tdx.py --phase full --data-dir data/tdx_import/1d
    python finetune/train_predictor_tdx.py --phase iib  --data-dir data/tdx_import/1d
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
from model.covariate import InputInjectionBlock


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


# ---------------------------------------------------------------------------
# Progressive unfreezing helpers (Phase 2)
# ---------------------------------------------------------------------------

def apply_freeze_stage(model, stage, config):
    """按阶段设置参数冻结状态。"""
    for param in model.parameters():
        param.requires_grad = False

    if stage == 'iib_only':
        for name, param in model.named_parameters():
            if 'iib' in name:
                param.requires_grad = True

    elif stage == 'iib_plus_top':
        n_layers = model.n_layers
        unfreeze_from = n_layers - 4
        for name, param in model.named_parameters():
            if 'iib' in name:
                param.requires_grad = True
            for layer_idx in range(unfreeze_from, n_layers):
                if f'transformer.{layer_idx}.' in name:
                    param.requires_grad = True
            if 'head.' in name or 'norm.' in name:
                param.requires_grad = True

    elif stage == 'all':
        for param in model.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Stage '{stage}': trainable {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


def build_optimizer(model, stage, config):
    """根据阶段构建差异化学习率的优化器。"""
    if stage == 'iib_only':
        iib_params = [p for n, p in model.named_parameters() if 'iib' in n]
        return torch.optim.AdamW([
            {'params': iib_params, 'lr': config['iib_learning_rate'],
             'weight_decay': config.get('iib_weight_decay', 0.2)},
        ], betas=(config['adam_beta1'], config['adam_beta2']))

    elif stage == 'iib_plus_top':
        iib_params, top_params, base_params = [], [], []
        n_layers = model.n_layers
        unfreeze_from = n_layers - 4
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'iib' in name:
                iib_params.append(param)
            elif any(f'transformer.{i}.' in name for i in range(unfreeze_from, n_layers)):
                top_params.append(param)
            elif 'head.' in name or 'norm.' in name:
                top_params.append(param)
            else:
                base_params.append(param)

        groups = [
            {'params': iib_params, 'lr': config['iib_learning_rate'],
             'weight_decay': config.get('iib_weight_decay', 0.2)},
            {'params': top_params, 'lr': config['transformer_top_lr'],
             'weight_decay': config['adam_weight_decay']},
        ]
        return torch.optim.AdamW(groups, betas=(config['adam_beta1'], config['adam_beta2']))

    else:  # 'all'
        iib_params, top_params, base_params = [], [], []
        n_layers = model.n_layers
        unfreeze_from = n_layers - 4
        for name, param in model.named_parameters():
            if 'iib' in name:
                iib_params.append(param)
            elif any(f'transformer.{i}.' in name for i in range(unfreeze_from, n_layers)):
                top_params.append(param)
            elif 'head.' in name or 'norm.' in name:
                top_params.append(param)
            else:
                base_params.append(param)

        groups = [
            {'params': iib_params, 'lr': config['iib_learning_rate'],
             'weight_decay': config.get('iib_weight_decay', 0.2)},
            {'params': top_params, 'lr': config['transformer_top_lr'],
             'weight_decay': config['adam_weight_decay']},
            {'params': base_params, 'lr': config['transformer_base_lr'],
             'weight_decay': config['adam_weight_decay']},
        ]
        return torch.optim.AdamW(groups, betas=(config['adam_beta1'], config['adam_beta2']))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_phase_full(config: dict, config_obj, data_dir: str, device: torch.device):
    """Phase 1: 全参数微调（无 IIB/CZSC）。"""
    phase_epochs = config.get('phase1_epochs', 10)
    save_dir = os.path.join(config['save_path'], config['predictor_save_folder_name'])
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

    # Phase 1 不用协变量
    config_obj.use_iib = False

    train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(
        config, config_obj, data_dir
    )

    # 加载 tokenizer (frozen)
    print("Loading tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained(config['finetuned_tokenizer_path'])
    tokenizer.eval().to(device)
    for p in tokenizer.parameters():
        p.requires_grad = False

    # 加载 predictor (全参数可训练)
    print("Loading predictor (full fine-tuning)...")
    model = Kronos.from_pretrained(config['pretrained_predictor_path'])
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Predictor parameters: {n_params:,} (all trainable)")

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
        epochs=phase_epochs,
        pct_start=0.03,
        div_factor=10,
    )

    use_amp = config.get('use_amp', True) and device.type == 'cuda'
    print(f"AMP (bf16): {use_amp}")

    best_val_loss = float('inf')
    start_time = time.time()
    accum_steps = config.get('predictor_accumulation', 1)
    patience = config.get('early_stop_patience', 5)
    no_improve = 0

    for epoch_idx in range(phase_epochs):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        train_dataset.set_epoch_seed(epoch_idx * 10000)
        train_loss_total = 0.0
        train_batches = 0
        optimizer.zero_grad()

        for batch_idx, (batch_x, batch_x_stamp, _) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

            with torch.no_grad():
                tok0, tok1 = tokenizer.encode(batch_x, half=True)

            token_in = [tok0[:, :-1], tok1[:, :-1]]
            token_out = [tok0[:, 1:], tok1[:, 1:]]
            stamp_in = batch_x_stamp[:, :-1, :]

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                logits = model(token_in[0], token_in[1], stamp_in, past_covariates=None)
                loss, s1_loss, s2_loss = model.head.compute_loss(
                    logits[0], logits[1], token_out[0], token_out[1]
                )
                loss = loss / accum_steps

            loss.backward()

            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss_total += loss.item() * accum_steps
            train_batches += 1

            if (batch_idx + 1) % config['log_interval'] == 0:
                lr = optimizer.param_groups[0]['lr']
                avg = train_loss_total / train_batches
                print(f"[E{epoch_idx+1:2d}/{phase_epochs} "
                      f"B{batch_idx+1:4d}/{len(train_loader)}] "
                      f"LR {lr:.6f}  Loss {avg:.4f}  "
                      f"S1 {s1_loss.item():.4f}  S2 {s2_loss.item():.4f}")

        # --- Validation ---
        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            for batch_x, batch_x_stamp, _ in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

                tok0, tok1 = tokenizer.encode(batch_x, half=True)
                token_in = [tok0[:, :-1], tok1[:, :-1]]
                token_out = [tok0[:, 1:], tok1[:, 1:]]
                stamp_in = batch_x_stamp[:, :-1, :]

                logits = model(token_in[0], token_in[1], stamp_in, past_covariates=None)
                val_loss, _, _ = model.head.compute_loss(
                    logits[0], logits[1], token_out[0], token_out[1]
                )
                val_loss_total += val_loss.item()
                val_batches += 1

        avg_val_loss = val_loss_total / val_batches if val_batches > 0 else float('inf')
        elapsed = time.time() - epoch_start

        print(f"--- Epoch {epoch_idx+1}/{phase_epochs} "
              f"| Train Loss: {train_loss_total/train_batches:.4f} "
              f"| Val Loss: {avg_val_loss:.4f} "
              f"| Time: {elapsed/60:.1f}m ---")

        if avg_val_loss < best_val_loss - 1e-6:
            best_val_loss = avg_val_loss
            no_improve = 0
            ckpt_path = os.path.join(save_dir, 'checkpoints', 'best_model')
            model.save_pretrained(ckpt_path)
            print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  -> Early stopping at epoch {epoch_idx+1}")
                break

    total_time = time.time() - start_time
    print(f"\nPhase 1 complete ({epoch_idx+1}/{phase_epochs} epochs). Best val loss: {best_val_loss:.4f}")
    print(f"Total time: {total_time/3600:.1f}h")

    summary = {
        'phase': 'full',
        'best_val_loss': best_val_loss,
        'total_time_h': total_time / 3600,
        'epochs': epoch_idx + 1,
        'n_params': n_params,
        'amp_bf16': use_amp,
        'batch_size': config['predictor_batch_size'],
        'accumulation': accum_steps,
    }
    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return best_val_loss


def train_phase_iib(config: dict, config_obj, data_dir: str, device: torch.device):
    """Phase 2: IIB + CZSC 渐进式解冻训练。"""
    save_dir = os.path.join(config['save_path'], config['predictor_save_folder_name'])
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

    # Phase 2 使用协变量
    config_obj.use_iib = True

    train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(
        config, config_obj, data_dir
    )

    # 加载 tokenizer (frozen)
    print("Loading tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained(config['finetuned_tokenizer_path'])
    tokenizer.eval().to(device)
    for p in tokenizer.parameters():
        p.requires_grad = False

    # 加载 Phase 1 微调后的 predictor
    predictor_path = config['finetuned_predictor_path']
    print(f"Loading Phase 1 predictor from {predictor_path}...")
    model = Kronos.from_pretrained(predictor_path)

    # 替换 IIB 为升级版（随机初始化）
    model.iib = InputInjectionBlock(
        d_model=model.d_model,
        cov_dim=config['cov_dim'],
        hidden_dim=config['iib_hidden_dim'],
        dropout=config['iib_dropout'],
        n_layers=config['iib_n_layers'],
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    iib_params = sum(p.numel() for n, p in model.named_parameters() if 'iib' in n)
    print(f"Predictor: {n_params:,} total, IIB (upgraded): {iib_params:,} ({100*iib_params/n_params:.2f}%)")

    # 渐进式解冻配置
    stage_a_epochs = config.get('iib_only_epochs', 5)
    stage_b_epochs = config.get('iib_plus_top_epochs', 5)
    total_epochs = config.get('epochs', 30)

    use_amp = config.get('use_amp', True) and device.type == 'cuda'
    print(f"AMP (bf16): {use_amp}")
    print(f"Progressive unfreezing: A={stage_a_epochs}, B={stage_b_epochs}, C={total_epochs-stage_a_epochs-stage_b_epochs}")

    best_val_loss = float('inf')
    start_time = time.time()
    accum_steps = config.get('predictor_accumulation', 1)
    patience = config.get('early_stop_patience', 5)
    no_improve = 0
    prev_stage = None
    optimizer = None
    scheduler = None

    for epoch_idx in range(total_epochs):
        # 确定当前阶段
        if epoch_idx < stage_a_epochs:
            stage = 'iib_only'
        elif epoch_idx < stage_a_epochs + stage_b_epochs:
            stage = 'iib_plus_top'
        else:
            stage = 'all'

        # 阶段切换时重建 optimizer/scheduler
        if stage != prev_stage:
            print(f"\n=== Entering stage: {stage} (epoch {epoch_idx+1}) ===")
            apply_freeze_stage(model, stage, config)
            optimizer = build_optimizer(model, stage, config)
            remaining_epochs = total_epochs - epoch_idx
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[g['lr'] for g in optimizer.param_groups],
                steps_per_epoch=len(train_loader),
                epochs=remaining_epochs,
                pct_start=0.03,
                div_factor=10,
            )
            prev_stage = stage

        epoch_start = time.time()

        # --- Train ---
        model.train()
        train_dataset.set_epoch_seed(epoch_idx * 10000)
        train_loss_total = 0.0
        train_batches = 0
        optimizer.zero_grad()

        for batch_idx, (batch_x, batch_x_stamp, batch_cov) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
            batch_cov = batch_cov.to(device, non_blocking=True)

            with torch.no_grad():
                tok0, tok1 = tokenizer.encode(batch_x, half=True)

            token_in = [tok0[:, :-1], tok1[:, :-1]]
            token_out = [tok0[:, 1:], tok1[:, 1:]]
            stamp_in = batch_x_stamp[:, :-1, :]
            cov_in = batch_cov[:, :-1, :]

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                logits = model(token_in[0], token_in[1], stamp_in, past_covariates=cov_in)
                loss, s1_loss, s2_loss = model.head.compute_loss(
                    logits[0], logits[1], token_out[0], token_out[1]
                )
                loss = loss / accum_steps

            loss.backward()

            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss_total += loss.item() * accum_steps
            train_batches += 1

            if (batch_idx + 1) % config['log_interval'] == 0:
                lr = optimizer.param_groups[0]['lr']
                avg = train_loss_total / train_batches
                print(f"[E{epoch_idx+1:2d}/{total_epochs} ({stage}) "
                      f"B{batch_idx+1:4d}/{len(train_loader)}] "
                      f"LR {lr:.6f}  Loss {avg:.4f}  "
                      f"S1 {s1_loss.item():.4f}  S2 {s2_loss.item():.4f}")

        # --- Validation ---
        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            for batch_x, batch_x_stamp, batch_cov in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
                batch_cov = batch_cov.to(device, non_blocking=True)

                tok0, tok1 = tokenizer.encode(batch_x, half=True)
                token_in = [tok0[:, :-1], tok1[:, :-1]]
                token_out = [tok0[:, 1:], tok1[:, 1:]]
                stamp_in = batch_x_stamp[:, :-1, :]
                cov_in = batch_cov[:, :-1, :]

                logits = model(token_in[0], token_in[1], stamp_in, past_covariates=cov_in)
                val_loss, _, _ = model.head.compute_loss(
                    logits[0], logits[1], token_out[0], token_out[1]
                )
                val_loss_total += val_loss.item()
                val_batches += 1

        avg_val_loss = val_loss_total / val_batches if val_batches > 0 else float('inf')
        elapsed = time.time() - epoch_start

        print(f"--- Epoch {epoch_idx+1}/{total_epochs} ({stage}) "
              f"| Train: {train_loss_total/train_batches:.4f} "
              f"| Val: {avg_val_loss:.4f} "
              f"| Time: {elapsed/60:.1f}m ---")

        if avg_val_loss < best_val_loss - 1e-6:
            best_val_loss = avg_val_loss
            no_improve = 0
            ckpt_path = os.path.join(save_dir, 'checkpoints', 'best_model')
            model.save_pretrained(ckpt_path)
            print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  -> Early stopping at epoch {epoch_idx+1}")
                break

    total_time = time.time() - start_time
    print(f"\nPhase 2 complete ({epoch_idx+1}/{total_epochs} epochs). Best val loss: {best_val_loss:.4f}")
    print(f"Total time: {total_time/3600:.1f}h")

    summary = {
        'phase': 'iib',
        'best_val_loss': best_val_loss,
        'total_time_h': total_time / 3600,
        'epochs': epoch_idx + 1,
        'n_params': n_params,
        'amp_bf16': use_amp,
        'batch_size': config['predictor_batch_size'],
        'iib_n_layers': config['iib_n_layers'],
        'iib_dropout': config['iib_dropout'],
        'iib_lr': config['iib_learning_rate'],
        'stages': f"A={stage_a_epochs}, B={stage_b_epochs}",
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
    parser.add_argument('--phase', choices=['full', 'iib'], default='full',
                        help='Phase 1: full fine-tuning. Phase 2: IIB+CZSC.')
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

    # Phase 控制
    cfg['phase'] = args.phase
    if args.epochs > 0:
        cfg['epochs'] = args.epochs
    if args.tokenizer_path:
        cfg['finetuned_tokenizer_path'] = args.tokenizer_path
    if args.predictor_path:
        cfg['finetuned_predictor_path'] = args.predictor_path
    if args.no_amp:
        cfg['use_amp'] = False

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Phase: {args.phase}")
    print(f"Device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Batch size: {cfg['predictor_batch_size']}, AMP: {cfg['use_amp']}")

    if args.phase == 'full':
        print(f"Epochs: {cfg.get('phase1_epochs', 10)} (Phase 1)")
        print(f"Pretrained predictor: {cfg['pretrained_predictor_path']}")
        print(f"Finetuned tokenizer: {cfg['finetuned_tokenizer_path']}")
        train_phase_full(cfg, config, args.data_dir, device)
    else:
        print(f"Epochs: {cfg['epochs']} (Phase 2, progressive unfreezing)")
        print(f"Finetuned predictor: {cfg['finetuned_predictor_path']}")
        print(f"Finetuned tokenizer: {cfg['finetuned_tokenizer_path']}")
        train_phase_iib(cfg, config, args.data_dir, device)


if __name__ == '__main__':
    main()

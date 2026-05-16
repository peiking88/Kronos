#!/usr/bin/env python3
"""
Predict SSE Composite Index (上证指数 sh000001) using fine-tuned Kronos models.

Loads the TDX fine-tuned tokenizer and predictor, fetches the latest
sh000001 daily data, and generates future price forecasts.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import pandas as pd
import torch

from model.kronos import KronosTokenizer, Kronos, KronosPredictor


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load fine-tuned models
    # ------------------------------------------------------------------
    tokenizer_path = "./outputs/tdx_finetune/tdx_tokenizer/checkpoints/best_model"
    predictor_path = "./outputs/tdx_finetune/tdx_predictor/checkpoints/best_model"

    print(f"Loading fine-tuned tokenizer: {tokenizer_path}")
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path).to(device)
    print(f"  params: {sum(p.numel() for p in tokenizer.parameters()):,}")

    print(f"Loading fine-tuned predictor: {predictor_path}")
    model = Kronos.from_pretrained(predictor_path).to(device)
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    # ------------------------------------------------------------------
    # 2. Load sh000001 data
    # ------------------------------------------------------------------
    with open("./data/tdx_import_sse/1d/data.pkl", "rb") as f:
        data = pickle.load(f)

    df = data["sh000001"]
    print(f"\nsh000001 data: {len(df)} rows, {df.index[0]} ~ {df.index[-1]}")
    print(f"Columns: {list(df.columns)}")

    # Ensure correct column names for predictor
    df = df.rename(columns={"vol": "volume", "amt": "amount"})

    # ------------------------------------------------------------------
    # 3. Prediction parameters
    # ------------------------------------------------------------------
    lookback = 90       # Use last 90 trading days as context
    pred_len = 20       # Predict next 20 trading days (~1 month)
    T = 0.6             # Temperature for sampling
    top_p = 0.9         # Nucleus sampling
    sample_count = 5    # Average over N samples

    context_df = df.iloc[-lookback:]
    x_ts_index = pd.to_datetime(context_df.index)
    x_timestamp = pd.Series(x_ts_index.values, name='timestamps')
    last_date = x_ts_index[-1]
    # Generate future trading day timestamps
    y_timestamp = pd.Series(pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=pred_len, freq="C",
        weekmask="Mon Tue Wed Thu Fri"
    ))

    print(f"\nLookback:  {lookback} days ({x_ts_index[0].date()} ~ {x_ts_index[-1].date()})")
    print(f"Predict:   {pred_len} days ({y_timestamp.iloc[0].date()} ~ {y_timestamp.iloc[-1].date()})")
    print(f"Params:    T={T}, top_p={top_p}, samples={sample_count}")

    # ------------------------------------------------------------------
    # 4. Run prediction
    # ------------------------------------------------------------------
    print("\nGenerating forecast...")
    pred_df = predictor.predict(
        df=context_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=T,
        top_p=top_p,
        sample_count=sample_count,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 5. Display results
    # ------------------------------------------------------------------
    last_close = context_df["close"].iloc[-1]
    print(f"\n{'='*70}")
    print(f"  上证指数 (sh000001) 走势预测")
    print(f"{'='*70}")
    print(f"  基准收盘价: {last_close:.2f}")
    print(f"{'='*70}")
    print(f"  {'日期':<14s} {'开盘':>8s} {'最高':>8s} {'最低':>8s} {'收盘':>8s} {'涨跌幅':>8s}")
    print(f"  {'-'*54}")

    for i, (ts, row) in enumerate(pred_df.iterrows()):
        chg = (row["close"] - last_close) / last_close * 100
        prev = last_close if i == 0 else pred_df.iloc[i-1]["close"]
        daily_chg = (row["close"] - prev) / prev * 100
        print(f"  {str(ts.date()):<14s} "
              f"{row['open']:>8.2f} {row['high']:>8.2f} "
              f"{row['low']:>8.2f} {row['close']:>8.2f} {daily_chg:>+7.2f}%")

    final_close = pred_df["close"].iloc[-1]
    total_chg = (final_close - last_close) / last_close * 100
    print(f"  {'-'*54}")
    print(f"  20日预测涨跌幅: {total_chg:+.2f}%")
    print(f"  预测终点: {final_close:.2f} (起始: {last_close:.2f})")
    print(f"{'='*70}")

    # Direction summary
    up_days = (pred_df["close"].diff().dropna() > 0).sum()
    down_days = pred_len - 1 - up_days
    print(f"\n  涨跌统计: {int(up_days)}涨 / {int(down_days)}跌")
    pred_high = pred_df["high"].max()
    pred_low = pred_df["low"].min()
    print(f"  预测区间: {pred_low:.2f} ~ {pred_high:.2f}")
    print(f"  波动幅度: {(pred_high - pred_low) / last_close * 100:.2f}%")


if __name__ == "__main__":
    main()

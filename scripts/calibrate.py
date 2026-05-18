#!/usr/bin/env python3
"""回测校准模块：计算模型系统性偏差，用于修正预测结果。

用法:
    from scripts.calibrate import backtest_calibrate
    bias = backtest_calibrate(predictor, df, pred_len=30)
"""

import numpy as np
import pandas as pd

BACKTEST_DAYS = 60
BACKTEST_CTX = 400


def backtest_calibrate(predictor, df, pred_len, lookback=BACKTEST_CTX,
                       backtest_days=BACKTEST_DAYS, temperature=1.2,
                       top_p=0.95, sample_count=2, verbose=True):
    """回测校准：用最近 backtest_days 天做回测，计算系统性偏差 ME。

    参数:
        predictor: KronosPredictor 实例（已加载模型）
        df: 历史行情 DataFrame，需含 open/high/low/close/volume/amount 列
        pred_len: 预测天数（仅用于日志）
        lookback: 回测上下文天数
        backtest_days: 回测验证天数
        temperature, top_p, sample_count: 采样参数

    返回:
        bias_correction: 校正值（正值 = 预测偏低，需上调；负值 = 预测偏高，需下调）
    """
    if len(df) < lookback + backtest_days:
        if verbose:
            print(f"  数据不足回测校准 (需 >= {lookback + backtest_days}，实际 {len(df)})，跳过")
        return 0.0

    backtest_df = df.iloc[-(lookback + backtest_days):]
    ctx_df = backtest_df.iloc[:lookback]
    actual_df = backtest_df.iloc[lookback:]

    ctx_ts = pd.Series(pd.to_datetime(ctx_df.index).values, name="timestamps")
    actual_ts = pd.Series(pd.to_datetime(actual_df.index).values, name="timestamps")

    if verbose:
        print(f"  回测校准: 上下文 {ctx_df.index[0]}~{ctx_df.index[-1]} "
              f"→ 预测 {actual_df.index[0]}~{actual_df.index[-1]}")

    try:
        bt_pred = predictor.predict(
            df=ctx_df[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True),
            x_timestamp=ctx_ts,
            y_timestamp=actual_ts,
            pred_len=backtest_days,
            T=temperature,
            top_p=top_p,
            sample_count=sample_count,
            verbose=False,
        )
    except Exception as e:
        if verbose:
            print(f"  回测预测失败: {e}，跳过校准")
        return 0.0

    actual_close = actual_df["close"].values
    pred_close = bt_pred["close"].values[:len(actual_close)]
    me = np.mean(pred_close - actual_close)
    mae = np.mean(np.abs(pred_close - actual_close))
    mape = np.mean(np.abs(pred_close - actual_close) / actual_close * 100)

    if verbose:
        print(f"  回测结果: ME={me:+.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")
        print(f"  偏差校正值: {-me:+.2f}")

    return float(-me)

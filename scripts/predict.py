#!/usr/bin/env python3
"""
Kronos 一键预测脚本

用法:
    python scripts/predict.py 600000
    python scripts/predict.py 002741 600519 000001
    python scripts/predict.py --pred-len 60 600000
    python scripts/predict.py --model base 600000

功能:
    1. 从 TDX 本地数据导入指定股票最新行情
    2. Kronos 模型预测未来 N 日走势
    3. 回测计算模型偏差值（不自动修正）
    4. 应用涨跌停约束
    5. 输出 CSV + 交互式 HTML 图表
"""
import argparse
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

# Pre-import mootdx/opentdx before sys.path modification to avoid
# namespace package conflict with ~/peiking88/opentdx/ directory.
import mootdx  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import Kronos, KronosTokenizer, KronosPredictor
from scripts.calibrate import backtest_calibrate

# ── 常量 ──────────────────────────────────────────────
TDX_DEFAULT = os.path.expanduser("~/.local/share/tdxcfv/drive_c/tc/")
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_MAP = {
    "mini":  "NeoQuasar/Kronos-mini",
    "small": "NeoQuasar/Kronos-small",
    "base":  "NeoQuasar/Kronos-base",
}
LOOKBACK = 400
LIMIT_RATE = 0.10
BACKTEST_DAYS = 60      # 回测校准窗口（约三个月交易日）
BACKTEST_CTX = 400       # 回测上下文天数
FACTOR_DIR = os.path.expanduser("~/.local/share/tdxcfv/drive_c/tc/.factor_cache")


# ── 工具函数 ──────────────────────────────────────────
def _load_factor_df(code):
    """加载后复权因子 DataFrame，优先本地缓存。"""
    cache_file = os.path.join(FACTOR_DIR, f"{code}.pkl")
    if os.path.exists(cache_file):
        try:
            f = pd.read_pickle(cache_file)
            f.index = pd.to_datetime(f.index)
            if not f.empty:
                return f
        except Exception:
            pass
    try:
        from tdxdata.sources.adjust import fetch_factor
        from mootdx.quotes import Quotes
        quotes = Quotes.factory(market='std')
        factor_df = fetch_factor(code, "hfq", quotes_client=quotes)
        if factor_df is not None and not factor_df.empty:
            os.makedirs(FACTOR_DIR, exist_ok=True)
            factor_df.to_pickle(cache_file)
            return factor_df
    except Exception as e:
        print(f"  警告: 获取 {code} 复权因子失败: {e}")
    return None


def normalize_symbol(raw: str) -> str:
    """将用户输入的代码标准化为 (market, symbol, tdx_key) 三元组。

    接受格式: 600000 / sh600000 / sz002741 / 000001
    """
    raw = raw.strip().lower()
    if raw.startswith("sh"):
        market, code = "sh", raw[2:]
    elif raw.startswith("sz"):
        market, code = "sz", raw[2:]
    elif raw.startswith("6") or raw.startswith("9"):
        market, code = "sh", raw
    else:
        market, code = "sz", raw
    return market, code, f"{market}{code}"


def import_from_tdx(tdx_key: str, tdxdir: str, end_date: str):
    """从 TDX 本地数据导入单只股票日线（后复权），返回 (DataFrame, factor)。

    模型使用后复权数据训练，因此推理输入也必须是后复权价格。
    factor 用于将预测结果从后复权空间换算回实际市场价格。
    """
    from mootdx.reader import Reader
    reader = Reader.factory(market="std", tdxdir=tdxdir)
    code = tdx_key[2:]
    df = reader.daily(symbol=code)
    if df is None or df.empty:
        raise RuntimeError(f"TDX 中未找到 {tdx_key} 的数据")
    df = df[df.index <= pd.Timestamp(end_date)]
    df = df[["open", "high", "low", "close", "amount", "volume"]].copy()
    df.index = pd.to_datetime(df.index)

    # 应用后复权（匹配 Kronos 模型训练数据）
    factor = 1.0
    factor_df = _load_factor_df(code)
    if factor_df is not None and not factor_df.empty:
        factor_df = factor_df.sort_index()
        factor = float(factor_df["factor"].iloc[-1])
        common_dtype = "datetime64[us]"
        df_tmp = df.copy()
        df_tmp.index = df_tmp.index.astype(common_dtype)
        fidx = factor_df.index.astype(common_dtype)
        merged = pd.merge_asof(
            df_tmp.reset_index().rename(columns={"index": "date"}),
            factor_df[["factor"]].set_index(fidx),
            left_on="date", right_index=True, direction="forward",
        )
        merged["factor"] = merged["factor"].ffill().bfill().fillna(1.0)
        for col in ["open", "high", "low", "close"]:
            merged[col] = merged[col] * merged["factor"]
        df = merged.drop(columns=["factor"]).set_index("date")
        df.index = pd.to_datetime(df.index)
        print(f"  已应用后复权 (factor={factor:.4f})")
    else:
        print(f"  警告: 无复权因子，使用不复权数据（预测偏差可能较大）")

    return df, factor


def apply_price_limits(pred_df: pd.DataFrame, last_close: float, limit_rate: float):
    """逐日应用涨跌停约束。"""
    lc = last_close
    for i in range(len(pred_df)):
        up, dn = lc * (1 + limit_rate), lc * (1 - limit_rate)
        for col in ["open", "high", "low", "close"]:
            val = pred_df.iat[i, pred_df.columns.get_loc(col)]
            pred_df.iat[i, pred_df.columns.get_loc(col)] = max(min(val, up), dn)
        lc = pred_df.iat[i, pred_df.columns.get_loc("close")]


def run_predict(predictor, df, pred_len, temperature, top_p, sample_count):
    """使用已加载的 predictor 预测。"""
    x_df = df.iloc[-LOOKBACK:][["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
    x_timestamp = pd.Series(df.iloc[-LOOKBACK:].index.to_list())
    last_date = df.index[-1]
    y_timestamp = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=pred_len)

    print(f"预测 {pred_len} 个交易日 (从 {y_timestamp[0].date()} 起)...")

    with torch.no_grad():
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=pd.Series(y_timestamp),
            pred_len=pred_len,
            T=temperature,
            top_p=top_p,
            sample_count=sample_count,
            verbose=True,
        )

    pred_df = pred_df.copy()
    pred_df["date"] = y_timestamp
    pred_df = pred_df[["date", "open", "high", "low", "close", "volume", "amount"]]
    return pred_df


def plot_result(hist_df, pred_df, tdx_key, out_html):
    """生成 plotly 交互图表。"""
    import plotly.graph_objects as go

    hist_tail = hist_df.iloc[-60:]
    last_close = hist_df.iloc[-1]["close"]
    last_date = hist_df.index[-1].strftime("%Y-%m-%d")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_tail.index, y=hist_tail["close"],
        mode="lines+markers", name="历史 (最近60日)",
        line=dict(color="#2563eb", width=2), marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=pred_df["date"], y=pred_df["close"],
        mode="lines+markers", name="预测",
        line=dict(color="#dc2626", width=2, dash="dash"), marker=dict(size=5, symbol="diamond"),
    ))
    fig.add_shape(type="line", x0=hist_df.index[-1], x1=hist_df.index[-1],
                  y0=0, y1=1, yref="paper", line_dash="dot", line_color="gray")
    fig.update_layout(
        title=f"{tdx_key} — 基于 {last_date} 收盘的 {len(pred_df)} 日预测",
        xaxis_title="日期", yaxis_title="收盘价",
        hovermode="x unified", template="plotly_white", height=520,
    )
    fig.write_html(out_html)


# ── 主流程 ────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Kronos 一键预测: 从 TDX 本地数据导入 → 模型预测 → 输出 CSV + 图表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python scripts/predict.py 600000\n"
               "  python scripts/predict.py 002741 600519\n"
               "  python scripts/predict.py --pred-len 60 --model base 600000\n",
    )
    parser.add_argument("symbols", nargs="+", help="股票代码，如 600000 002741 sh000001")
    parser.add_argument("--pred-len", type=int, default=5, help="预测交易日数 (默认 5)")
    parser.add_argument("--model", choices=MODEL_MAP.keys(), default="base",
                        help="模型大小: mini/small/base (默认 small)")
    parser.add_argument("--device", default="cpu", help="计算设备 (默认 cpu)")
    parser.add_argument("--temperature", "-T", type=float, default=1.2, help="采样温度 (默认 1.2)")
    parser.add_argument("--top-p", type=float, default=0.95, help="核采样概率 (默认 0.95)")
    parser.add_argument("--samples", type=int, default=2, help="采样次数 (默认 2)")
    parser.add_argument("--lookback", type=int, default=LOOKBACK, help="回看天数 (默认 400)")
    parser.add_argument("--no-limit", action="store_true", help="不应用涨跌停约束")
    parser.add_argument("--output-dir", default="outputs", help="输出目录 (默认 outputs)")
    parser.add_argument("--tdxdir", default=TDX_DEFAULT, help="TDX 数据目录")
    args = parser.parse_args()

    # 获取 TDX 最新交易日期
    from mootdx.reader import Reader
    reader = Reader.factory(market="std", tdxdir=args.tdxdir)
    ref = reader.daily(symbol="000001")
    end_date = ref.index[-1].strftime("%Y-%m-%d") if ref is not None and not ref.empty else None
    if not end_date:
        print("无法确定 TDX 最新交易日期，请检查 TDX 数据目录")
        sys.exit(1)
    print(f"TDX 最新交易日: {end_date}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    today_str = end_date.replace("-", "")

    # 加载模型（一次）
    model_id = MODEL_MAP[args.model]
    print(f"加载 tokenizer: {TOKENIZER_ID}")
    print(f"加载模型: {model_id} (device={args.device})")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_ID)
    model = Kronos.from_pretrained(model_id)
    model.eval()
    predictor = KronosPredictor(model, tokenizer, device=args.device, max_context=512)
    print()

    for raw_sym in args.symbols:
        market, code, tdx_key = normalize_symbol(raw_sym)
        print(f"{'='*60}")
        print(f"股票: {tdx_key}")
        print(f"{'='*60}")

        # 1. 导入数据（后复权）
        try:
            df, factor = import_from_tdx(tdx_key, args.tdxdir, end_date)
        except Exception as e:
            print(f"导入失败: {e}\n")
            continue

        if len(df) < args.lookback:
            print(f"数据不足: {len(df)} 根 < 回看 {args.lookback}，跳过\n")
            continue

        last_close_hfq = df.iloc[-1]["close"]
        last_close = last_close_hfq / factor
        print(f"数据: {len(df)} 根, 末日期 {df.index[-1].date()}, 收盘 {last_close:.2f}")

        # 2. 预测
        try:
            pred_df = run_predict(
                predictor, df, args.pred_len,
                args.temperature, args.top_p, args.samples,
            )
        except Exception as e:
            print(f"预测失败: {e}\n")
            continue

        # 3. 将预测结果从后复权空间换算为实际价格
        if factor != 1.0:
            for col in ["open", "high", "low", "close"]:
                pred_df[col] = pred_df[col] / factor

        # 4. 涨跌停约束（实际价格空间）
        if not args.no_limit:
            apply_price_limits(pred_df, last_close, LIMIT_RATE)

        # 5. 回测校准 — 仅计算偏差值，不修改预测
        bias_correction = 0.0
        if not args.no_limit:
            bias_correction = backtest_calibrate(
                predictor, df, args.pred_len,
                temperature=args.temperature, top_p=args.top_p,
                sample_count=args.samples,
            )
            # 将 hfq 空间的偏差换算为实际价格空间
            bias_correction = bias_correction / factor

        # 6. 保存 CSV
        out_csv = os.path.join(args.output_dir, f"pred_{tdx_key}_{today_str}.csv")
        pred_df.to_csv(out_csv, index=False, float_format="%.2f")

        # 7. 生成图表（实际价格空间）
        df_actual = df.copy()
        if factor != 1.0:
            for col in ["open", "high", "low", "close"]:
                df_actual[col] = df_actual[col] / factor
        out_html = os.path.join(args.output_dir, f"pred_{tdx_key}_{today_str}_chart.html")
        try:
            plot_result(df_actual, pred_df, tdx_key, out_html)
        except Exception as e:
            print(f"图表生成失败 (可忽略): {e}")
            out_html = None

        # 8. 摘要
        pred_first = pred_df.iloc[0]["close"]
        pred_last = pred_df.iloc[-1]["close"]
        chg_first = (pred_first - last_close) / last_close * 100
        chg_last = (pred_last - last_close) / last_close * 100

        print(f"\n=== {tdx_key} 预测摘要 ===")
        print(f"收盘: {last_close:.2f}")
        if abs(bias_correction) > 0.01:
            print(f"过去一个月模型偏差值: {bias_correction:+.2f} (正值=模型预测偏低，负值=模型预测偏高)")
        print(f"首日 ({pred_df.iloc[0]['date'].date()}): {pred_first:.2f} ({chg_first:+.2f}%)")
        print(f"末日 ({pred_df.iloc[-1]['date'].date()}): {pred_last:.2f} ({chg_last:+.2f}%)")
        print(f"区间: [{pred_df['close'].min():.2f}, {pred_df['close'].max():.2f}]")
        print(f"CSV: {out_csv}")
        if out_html:
            print(f"图表: {out_html}")
        print()

    print("全部完成。")


if __name__ == "__main__":
    main()

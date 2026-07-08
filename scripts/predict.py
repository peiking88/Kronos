#!/usr/bin/env python3
"""
Kronos 一键预测脚本

用法:
    python scripts/predict.py 600000
    python scripts/predict.py 002741 600519 000001
    python scripts/predict.py --pred-len 60 600000
    python scripts/predict.py --model base 600000

功能:
    1. 从 TDengine 导入指定股票最新行情
    2. Kronos 模型预测未来 N 日走势
    3. 回测计算模型偏差值（不自动修正）
    4. 应用涨跌停约束
    5. 输出 CSV + 交互式 HTML 图表
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from taosws import connect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import Kronos, KronosTokenizer, KronosPredictor
from scripts.calibrate import backtest_calibrate

# ── 常量 ──────────────────────────────────────────────
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_MAP = {
    "mini":  "NeoQuasar/Kronos-mini",
    "small": "NeoQuasar/Kronos-small",
    "base":  "NeoQuasar/Kronos-base",
}
LOOKBACK = 400
LIMIT_RATE = 0.10
BACKTEST_DAYS = 60
BACKTEST_CTX = 400


# ── TDengine 数据工具 ─────────────────────────────────
def _query_adjust_events(conn, symbol: str) -> list[dict]:
    """查询分红事件。symbol 形如 'sh600000'。"""
    events = []
    try:
        r = conn.query(
            f"select ts, fenhong, peigujia, songzhuangu, peigu "
            f"from tdx.a_{symbol} order by ts"
        )
        for row in r:
            ts, fh, pj, sz, pg = row
            fh, pj, sz, pg = float(fh), float(pj), float(sz), float(pg)
            if fh > 0 or sz > 0 or pg > 0:
                events.append({
                    'date': pd.Timestamp(ts).tz_localize(None),
                    'fenhong': fh,
                    'peigujia': pj,
                    'songzhuangu': sz,
                    'peigu': pg,
                })
    except Exception:
        pass
    return events


def _compute_back_adjust_factor(df: pd.DataFrame, events: list[dict]) -> np.ndarray:
    """从分红事件计算后复权因子。"""
    n = len(df)
    factor = np.ones(n, dtype=np.float64)
    if not events:
        return factor

    events_sorted = sorted(events, key=lambda e: e['date'])
    df_dates = df.index.values
    raw_close = df['close'].values

    for evt in events_sorted:
        evt_date = np.datetime64(evt['date'])
        event_idx = int(np.searchsorted(df_dates, evt_date))
        if event_idx >= n:
            continue
        prev_idx = event_idx - 1
        if prev_idx < 0:
            continue
        C_before = raw_close[prev_idx]
        if C_before <= 0:
            continue

        D = evt['fenhong'] / 10.0
        S = evt['songzhuangu'] / 10.0
        P = evt['peigu'] / 10.0
        Pp = evt['peigujia']

        denominator = C_before - D + P * Pp
        if denominator <= 0:
            continue
        multiplier = C_before * (1.0 + S + P) / denominator
        if abs(multiplier - 1.0) < 1e-12:
            continue
        factor[:event_idx] *= multiplier
    return factor


def _apply_adjustment(df: pd.DataFrame, factor: np.ndarray) -> pd.DataFrame:
    """应用后复权因子到 OHLC。"""
    df_adj = df.copy()
    for col in ['open', 'high', 'low', 'close']:
        if col in df_adj.columns:
            df_adj[col] = (df_adj[col].values * factor).astype(np.float32)
    df_adj['vol'] = df_adj['vol'].astype(np.float32)
    df_adj['amt'] = df_adj['amt'].astype(np.float32)
    return df_adj


def _get_latest_trading_date() -> str | None:
    """从 TDengine 获取最新交易日期。"""
    conn = connect()
    try:
        r = conn.query(
            "select last_row(ts) from tdx.k_sh000001_1d"
        )
        rows = list(r)
        if rows:
            ts = pd.Timestamp(rows[0][0]).tz_localize(None)
            return ts.strftime("%Y-%m-%d")
    except Exception:
        pass
    finally:
        conn.close()
    return None


def normalize_symbol(raw: str) -> tuple[str, str, str]:
    """将用户输入的代码标准化为 (market, code, tdx_key)。"""
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


def import_from_tdx(tdx_key: str, end_date: str):
    """从 TDengine 导入单只股票日线（后复权），返回 (DataFrame, factor)。

    后复权因子由 adjust 表事件实时计算。factor 恒为 1.0（后复权数据末
    日即实际市场价），保留以兼容下游代码。
    """
    conn = connect()
    try:
        r = conn.query(
            f"select ts, open, high, low, close, volume, amount "
            f"from tdx.k_{tdx_key}_1d order by ts"
        )
        rows = list(r)
        if len(rows) < 100:
            raise RuntimeError(f"TDengine 中未找到 {tdx_key} 的数据")

        df = pd.DataFrame(
            rows,
            columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'amt'],
        )
        df['ts'] = pd.to_datetime(df['ts']).dt.tz_localize(None)
        df = df.set_index('ts').sort_index()
        df = df.astype({c: np.float64 for c in ['open', 'high', 'low', 'close', 'vol', 'amt']})

        # 后复权
        events = _query_adjust_events(conn, tdx_key)
        factor_arr = _compute_back_adjust_factor(df, events)
        df = _apply_adjustment(df, factor_arr)

        # 日期过滤
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]

        print(f"  已应用后复权 (来自 adjust 表事件)")
    finally:
        conn.close()

    # 后复权：末日即市场价，factor=1.0
    factor = 1.0
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
    today = pd.Timestamp.today().normalize()
    pred_start = max(last_date, today) + pd.Timedelta(days=1)
    y_timestamp = pd.bdate_range(start=pred_start, periods=pred_len)

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
        description="Kronos 一键预测: 从 TDengine → 模型预测 → 输出 CSV + 图表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python scripts/predict.py 600000\n"
               "  python scripts/predict.py 002741 600519\n"
               "  python scripts/predict.py --pred-len 60 --model base 600000\n",
    )
    parser.add_argument("symbols", nargs="+", help="股票代码，如 600000 002741 sh000001")
    parser.add_argument("--pred-len", type=int, default=5, help="预测交易日数 (默认 5)")
    parser.add_argument("--model", choices=MODEL_MAP.keys(), default="base",
                        help="模型大小: mini/small/base (默认 base)")
    parser.add_argument("--device", default="cpu", help="计算设备 (默认 cpu)")
    parser.add_argument("--temperature", "-T", type=float, default=1.2, help="采样温度 (默认 1.2)")
    parser.add_argument("--top-p", type=float, default=0.95, help="核采样概率 (默认 0.95)")
    parser.add_argument("--samples", type=int, default=2, help="采样次数 (默认 2)")
    parser.add_argument("--lookback", type=int, default=LOOKBACK, help="回看天数 (默认 400)")
    parser.add_argument("--no-limit", action="store_true", help="不应用涨跌停约束")
    parser.add_argument("--output-dir", default="outputs", help="输出目录 (默认 outputs)")
    args = parser.parse_args()

    # 获取 TDengine 最新交易日期
    end_date = _get_latest_trading_date()
    if not end_date:
        print("无法确定 TDengine 最新交易日期，请检查 TDengine 连接")
        sys.exit(1)
    print(f"TDengine 最新交易日: {end_date}\n")

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
            df, factor = import_from_tdx(tdx_key, end_date)
        except Exception as e:
            print(f"导入失败: {e}\n")
            continue

        # 1.5 盘中追加实时行情
        from scripts.realtime import append_realtime_bars
        append_realtime_bars([tdx_key], {tdx_key: df}, {tdx_key: factor})

        if len(df) < args.lookback:
            print(f"数据不足: {len(df)} 根 < 回看 {args.lookback}，跳过\n")
            continue

        last_close = df.iloc[-1]["close"]
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

        # 3. 涨跌停约束（后复权数据末日即实际价格）
        if not args.no_limit:
            apply_price_limits(pred_df, last_close, LIMIT_RATE)

        # 4. 回测校准
        bias_correction = 0.0
        if not args.no_limit:
            bias_correction = backtest_calibrate(
                predictor, df, args.pred_len,
                temperature=args.temperature, top_p=args.top_p,
                sample_count=args.samples,
            )

        # 5. 保存 CSV
        out_csv = os.path.join(args.output_dir, f"pred_{tdx_key}_{today_str}.csv")
        pred_df.to_csv(out_csv, index=False, float_format="%.2f")

        # 6. 生成图表
        out_html = os.path.join(args.output_dir, f"pred_{tdx_key}_{today_str}_chart.html")
        try:
            plot_result(df, pred_df, tdx_key, out_html)
        except Exception as e:
            print(f"图表生成失败 (可忽略): {e}")
            out_html = None

        # 7. 摘要
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

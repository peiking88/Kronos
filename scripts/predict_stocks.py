#!/usr/bin/env python3
"""
一键预测脚本 — 使用微调模型预测个股未来10日收盘价，输出md报告。

用法:
    python scripts/predict_stocks.py sh600000 sz002741              # 预测2只
    python scripts/predict_stocks.py sz300450 sh600353 sz002741     # 预测3只
    python scripts/predict_stocks.py --help

价格已换算为实际市场价（不复权）。
"""

import argparse, os, sys, pickle, json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.kronos import KronosTokenizer, Kronos, KronosPredictor

# ---------------------------------------------------------------------------
STOCK_NAMES = {
    "sh999999": "上证指数", "sz399006": "创业板指", "sz399001": "深证成指",
    "sh600353": "旭光电子", "sz002741": "光华科技", "sz300450": "先导智能",
}
FACTOR_DIR = os.path.expanduser("~/.local/share/tdxcfv/drive_c/tc/.factor_cache")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tdx_import", "1d")
SSE_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tdx_import_sse", "1d", "data.pkl")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "tdx_finetune")

LOOKBACK = 90
PRED_LEN = 10
T = 0.6
TOP_P = 0.9
SAMPLE_COUNT = 5
BACKTEST_SAMPLE_COUNT = 3
BACKTEST_WINDOWS = 30
TDX_DIR = os.path.expanduser("~/.local/share/tdxcfv/drive_c/tc/")
MAX_STALE_DAYS = 5


def load_factor(code):
    """Return the last 后复权 factor for a stock, or 1.0 if not found."""
    cache_file = os.path.join(FACTOR_DIR, f"{code}.pkl")
    if os.path.exists(cache_file):
        f = pd.read_pickle(cache_file)
        f.index = pd.to_datetime(f.index)
        return float(f.sort_index()["factor"].iloc[-1])
    return 1.0


def get_data(code):
    """Return DataFrame for a stock code (6-field, 后复权)."""
    # Try SSE data first (for indices)
    if code.startswith("sh000") or code.startswith("sz399"):
        if os.path.exists(SSE_DATA):
            with open(SSE_DATA, "rb") as f:
                d = pickle.load(f)
            if code in d:
                return d[code].copy()

    # Try test / val / train / data pickle
    for fname in ["test_data.pkl", "val_data.pkl", "train_data.pkl", "data.pkl"]:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, "rb") as f:
                d = pickle.load(f)
            if code in d:
                return d[code].copy()
    return None


def load_model(device):
    tok = KronosTokenizer.from_pretrained(
        os.path.join(MODEL_DIR, "tdx_tokenizer", "checkpoints", "best_model")
    ).to(device)
    mod = Kronos.from_pretrained(
        os.path.join(MODEL_DIR, "tdx_predictor", "checkpoints", "best_model")
    ).to(device)
    return KronosPredictor(mod, tok, device=device, max_context=512)


def ensure_fresh_data(codes):
    """检查数据新鲜度，过期则从TDX本地文件导入最新数据并合并。

    返回 {code: merged_df}（仅包含需要刷新的股票）。
    """
    stale_codes = []
    existing_data = {}
    for code in codes:
        # 指数代码走 SSE_DATA 路径，不通过 TDX 导入
        if code.startswith("sh000") or code.startswith("sz399"):
            continue
        df = get_data(code)
        if df is None or len(df) == 0:
            stale_codes.append(code)
            continue
        latest = df.index.max().date()
        if (datetime.now().date() - latest).days > MAX_STALE_DAYS:
            stale_codes.append(code)
            existing_data[code] = df

    if not stale_codes:
        print("数据均为最新，无需导入。")
        return {}

    print(f"以下 {len(stale_codes)} 只股票数据需要更新: {stale_codes}")
    print("正在从本地TDX数据导入...")

    try:
        from scripts.tdx_import import TdxDataImporter
        importer = TdxDataImporter(tdxdir=TDX_DIR, dividend_type="back")
        fresh_dataset = importer.build_dataset(
            stale_codes, period="1d", check_continuity=False,
        )
    except Exception as e:
        print(f"导入失败: {e}，将使用现有数据。")
        return {}

    result = {}
    for code in stale_codes:
        if code not in fresh_dataset:
            print(f"  {code}: TDX中无数据，跳过")
            continue
        fresh_df = fresh_dataset[code]
        if code in existing_data:
            merged = pd.concat([existing_data[code], fresh_df])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            result[code] = merged
        else:
            result[code] = fresh_df
        print(f"  {code}: 数据已更新至 {result[code].index.max().strftime('%Y-%m-%d')}")

    return result


def run_prediction(predictor, df, factor):
    """Forward predict next PRED_LEN days. Returns list of dicts."""
    df = df.rename(columns={"vol": "volume", "amt": "amount"})
    context = df.iloc[-LOOKBACK:]
    last_date = pd.to_datetime(context.index[-1])
    last_close = context["close"].iloc[-1]

    future = pd.bdate_range(
        start=last_date + timedelta(days=1), periods=PRED_LEN,
        freq="C", weekmask="Mon Tue Wed Thu Fri"
    )
    x_ts = pd.Series(pd.to_datetime(context.index).values, name="timestamps")
    y_ts = pd.Series(future.values)

    pred = predictor.predict(
        df=context, x_timestamp=x_ts, y_timestamp=y_ts,
        pred_len=PRED_LEN, T=T, top_p=TOP_P,
        sample_count=SAMPLE_COUNT, verbose=False
    )

    rows = []
    for i, (ts, row) in enumerate(pred.iterrows()):
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        rows.append({
            "date": ts.date(),
            "open": round(o / factor, 2),
            "high": round(max(h, l) / factor, 2),
            "low": round(min(h, l) / factor, 2),
            "close": round(c / factor, 2),
            "cum_chg": (c - last_close) / last_close,
        })
    return rows, float(last_close) / factor


def run_backtest(predictor, df, factor):
    """Rolling backtest. Returns list of dicts."""
    df = df.rename(columns={"vol": "volume", "amt": "amount"})
    total_w = len(df) - LOOKBACK - PRED_LEN
    step = max(1, total_w // BACKTEST_WINDOWS)
    results = []

    for i in range(0, total_w, step):
        ctx = df.iloc[i:i + LOOKBACK]
        act = df.iloc[i + LOOKBACK:i + LOOKBACK + PRED_LEN]
        if len(act) < PRED_LEN:
            break
        try:
            p = predictor.predict(
                df=ctx,
                x_timestamp=pd.Series(pd.to_datetime(ctx.index).values),
                y_timestamp=pd.Series(pd.to_datetime(act.index).values),
                pred_len=PRED_LEN, T=T, top_p=TOP_P,
                sample_count=BACKTEST_SAMPLE_COUNT, verbose=False
            )
            for d in range(PRED_LEN):
                pc = p["close"].iloc[d]
                ac = act["close"].iloc[d]
                ao = act["open"].iloc[d]
                ph = max(p["high"].iloc[d], p["low"].iloc[d])
                pl = min(p["high"].iloc[d], p["low"].iloc[d])
                ah = act["high"].iloc[d]
                al = act["low"].iloc[d]
                results.append({
                    "day": d + 1,
                    "pc_hfq": pc, "ac_hfq": ac,
                    "ao_hfq": ao, "ph_hfq": ph, "pl_hfq": pl,
                    "ah_hfq": ah, "al_hfq": al,
                })
        except Exception:
            pass

    # Compute metrics
    r = pd.DataFrame(results)
    n_win = len(r) // PRED_LEN
    metrics = {}
    for d in [1, 3, 5, 7, 10]:
        day = r[r["day"] == d]
        if len(day) == 0:
            continue
        ape = np.abs(day["pc_hfq"] - day["ac_hfq"]) / day["ac_hfq"]
        dr = (np.sign(day["pc_hfq"] - day["ao_hfq"]) == np.sign(day["ac_hfq"] - day["ao_hfq"])).mean()
        hc = (day["ph_hfq"] >= day["ah_hfq"]).mean()
        lc = (day["pl_hfq"] <= day["al_hfq"]).mean()
        metrics[d] = {
            "mape": ape.mean(), "mdape": ape.median(),
            "lt3": (ape < 0.03).mean(), "lt5": (ape < 0.05).mean(),
            "dir": dr, "hi_cov": hc, "lo_cov": lc,
        }
    # Convert actual prices for confidence intervals
    day1 = r[r["day"] == 1]
    day5 = r[r["day"] == 5]
    day10 = r[r["day"] == 10]
    conf = {}
    if len(day1) > 0:
        conf["d1"] = float(np.median(np.abs(day1["pc_hfq"] - day1["ac_hfq"]))) / factor
    if len(day5) > 0:
        conf["d5"] = float(np.median(np.abs(day5["pc_hfq"] - day5["ac_hfq"]))) / factor
    if len(day10) > 0:
        conf["d10"] = float(np.median(np.abs(day10["pc_hfq"] - day10["ac_hfq"]))) / factor

    return metrics, n_win, conf


def format_table(rows):
    """Format prediction rows as markdown table."""
    lines = []
    lines.append("| 日期 | 开盘 | 最高 | 最低 | 收盘 | 累计涨跌 |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['date']} | {r['open']:.2f} | {r['high']:.2f} | "
            f"{r['low']:.2f} | {r['close']:.2f} | {r['cum_chg']*100:+.2f}% |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="一键预测个股未来10日收盘价")
    parser.add_argument("codes", nargs="+", help="股票代码，空格分割，如 sh600000 sz002741")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-o", "--output", default=None, help="输出md文件路径（默认自动生成）")
    parser.add_argument("--no-import", action="store_true", help="跳过自动导入，使用现有数据")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # 检查并导入最新数据
    fresh_cache = {} if args.no_import else ensure_fresh_data(args.codes)

    print(f"Loading fine-tuned model...")
    predictor = load_model(device)

    all_forward = {}
    all_bt = {}
    all_conf = {}
    errors = []

    for code in args.codes:
        print(f"\nProcessing {code}...")
        df = fresh_cache.get(code) or get_data(code)
        if df is None:
            errors.append(f"{code}: 数据未找到")
            continue

        factor = load_factor(code)
        name = STOCK_NAMES.get(code, code)

        # Forward
        rows, last_close_actual = run_prediction(predictor, df, factor)
        all_forward[code] = {"name": name, "rows": rows, "base": last_close_actual, "factor": factor}

        # Backtest
        metrics, n_win, conf = run_backtest(predictor, df, factor)
        all_bt[code] = {"name": name, "metrics": metrics, "windows": n_win}
        all_conf[code] = {"name": name, "conf": conf, "base": last_close_actual}

    # --- Generate report ---
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append(f"# 个股未来10日收盘价预测报告")
    lines.append(f"")
    lines.append(f"**时间**: {now} | **模型**: Kronos-base TDX后复权微调版 | **基准日**: 最近交易日")
    lines.append(f"")
    lines.append(f"> 所有价格为实际市场价（已从后复权换算）。涨跌幅 = (预测收盘 - 基准收盘) / 基准收盘。")
    lines.append(f"")
    lines.append("---")
    lines.append("")

    # Forward predictions
    lines.append("## 一、预测结果")
    lines.append("")
    for code in args.codes:
        if code not in all_forward:
            continue
        info = all_forward[code]
        lines.append(f"### {info['name']} ({code})")
        lines.append(f"")
        lines.append(f"基准收盘: **{info['base']:.2f}**")
        lines.append(f"")
        lines.append(format_table(info["rows"]))
        lines.append("")
        final = info["rows"][-1]
        lines.append(f"10日涨跌: **{final['cum_chg']*100:+.2f}%** → 终点 **{final['close']:.2f}**")
        lines.append("")

    # Accuracy
    lines.append("---")
    lines.append("")
    lines.append("## 二、准确度（历史回测）")
    lines.append("")
    for code in args.codes:
        if code not in all_bt:
            continue
        info = all_bt[code]
        lines.append(f"### {info['name']} ({code})  — {info['windows']} 个窗口")
        lines.append("")
        lines.append("| 周期 | MAPE | MdAPE | <3% | <5% | 方向 | 高覆盖 | 低覆盖 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for d in [1, 3, 5, 7, 10]:
            if d not in info["metrics"]:
                continue
            m = info["metrics"][d]
            lines.append(
                f"| D{d:2d} | {m['mape']:.1%} | {m['mdape']:.1%} | "
                f"{m['lt3']:.0%} | {m['lt5']:.0%} | {m['dir']:.0%} | "
                f"{m['hi_cov']:.0%} | {m['lo_cov']:.0%} |"
            )
        lines.append("")

    # Confidence intervals
    lines.append("---")
    lines.append("")
    lines.append("## 三、预测置信区间（中位误差）")
    lines.append("")
    lines.append("| 标的 | D1 ± | D5 ± | D10 ± |")
    lines.append("|---|---|---|---|")
    for code in args.codes:
        if code not in all_conf:
            continue
        info = all_conf[code]
        c = info["conf"]
        lines.append(
            f"| {info['name']} | "
            f"{c.get('d1', 0):.2f} | {c.get('d5', 0):.2f} | {c.get('d10', 0):.2f} |"
        )
    lines.append("")

    # Errors
    if errors:
        lines.append("---")
        lines.append("")
        lines.append("## 错误")
        lines.append("")
        for e in errors:
            lines.append(f"- {e}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("> **免责**: 预测结果仅供模型能力验证，不构成投资建议。")

    report = "\n".join(lines)

    # Output
    if args.output:
        out_path = args.output
    else:
        codes_str = "_".join(args.codes[:3])
        out_path = f"outputs/pred_{codes_str}_{datetime.now().strftime('%Y%m%d_%H%M')}.md"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)

    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    main()

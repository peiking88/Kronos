#!/usr/bin/env python3
"""
一键预测脚本 — 使用微调模型预测个股/指数未来5日走势。

用法:
    python scripts/predict_stocks.py                                # 读取TDX自选股，输出报告
    python scripts/predict_stocks.py sh600000 sz002741              # 预测个股，输出 MD 报告
    python scripts/predict_stocks.py --format console               # 读取自选股，控制台表格
    python scripts/predict_stocks.py sh600353 -n 8                  # 8线程并发预测

个股价格已换算为实际市场价（不复权）。指数为实际点位。
报告按 指数→看涨→看平→看跌 排列。

特性:
    - 施加10%涨跌停约束，预测价格不会超出日涨跌停范围
    - 模型偏差值仅作参考，不自动修正预测收盘价
    - 与上次预测结果对比，标注稳定性告警（预测跳变/方向翻转）
"""

import argparse, os, sys, pickle, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from taosws import connect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.kronos import KronosTokenizer, Kronos, KronosPredictor
from scripts.calibrate import backtest_calibrate

# ---------------------------------------------------------------------------
FACTOR_CACHE_1D = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", ".factor_1d.json",
)
FACTOR_CACHE_DAYS = 30
FACTOR_WORKERS = 4


def _cache_month_key(date_str):
    """从 'YYYY-MM-DD' 字符串提取 'YYYY-MM' 月份键。"""
    try:
        key = date_str[:7]
        return key if len(key) == 7 else None
    except Exception:
        return None


def prefetch_factors(codes, fresh_cache):
    """批量获取复权因子（一个月缓存，4线程并发）。

    从 TDengine 查询原始收盘价，与后复权收盘价对比推导因子。
    """
    cache = {}
    if os.path.exists(FACTOR_CACHE_1D):
        try:
            with open(FACTOR_CACHE_1D, "r") as f:
                cache = json.load(f)
        except Exception:
            cache = {}
    today_str = str(datetime.now().date())
    this_month = _cache_month_key(today_str)

    result = {}
    need_fetch = []

    for code in codes:
        if classify_code(code):
            result[code] = (1.0, True)
            continue

        cached = cache.get(code)
        if cached and _cache_month_key(cached.get("date", "")) == this_month:
            factor = cached["factor"]
            ok = factor != 1.0
            result[code] = (factor, ok)
            print(f"  {code}: 复权因子(缓存)={factor:.4f}")
            continue

        need_fetch.append(code)

    if need_fetch:
        print(f"  并发获取复权因子: {len(need_fetch)} 只, {FACTOR_WORKERS} 线程")

        def _fetch_one(code):
            """Worker: 从 TDengine 查询原始收盘价推导因子。"""
            try:
                df = fresh_cache.get(code) if fresh_cache else None
                if df is None:
                    df = get_data(code)
                factor = derive_factor(code, df, verbose=False) if df is not None else 1.0
            except Exception as e:
                print(f"  {code}: 因子推导异常 {type(e).__name__}: {e}", file=sys.stderr)
                factor = 1.0
            return code, factor

        with ThreadPoolExecutor(max_workers=FACTOR_WORKERS) as executor:
            futures = {executor.submit(_fetch_one, code): code for code in need_fetch}
            for future in as_completed(futures):
                code, factor = future.result()
                if factor == 1.0:
                    old = cache.get(code)
                    if old and old.get("factor", 1.0) != 1.0:
                        print(f"  {code}: 最新因子获取失败，使用旧缓存 {old['factor']:.4f}",
                              file=sys.stderr)
                        factor = old["factor"]
                result[code] = (factor, factor != 1.0)
                cache[code] = {"factor": factor, "date": today_str}

    os.makedirs(os.path.dirname(FACTOR_CACHE_1D), exist_ok=True)
    try:
        with open(FACTOR_CACHE_1D, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass

    return result


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tdx_import", "1d")
SSE_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tdx_import_sse", "1d", "data.pkl")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "tdx_finetune")

LOOKBACK = 90
PRED_LEN = 5
T = 0.6
TOP_P = 0.9
SAMPLE_COUNT = 5
BACKTEST_SAMPLE_COUNT = 3
BACKTEST_WINDOWS = 30
ZXG_BLK = os.path.expanduser("~/.local/share/tdxcfv/drive_c/tc/T0002/blocknew/zxg.blk")
MAX_STALE_DAYS = 0

BULL_THRESHOLD = 0.03
BEAR_THRESHOLD = -0.03

LIMIT_RATE = 0.10
CONSENSUS_RUNS = 3

STABILITY_THRESHOLD = 0.15
STABILITY_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "outputs", ".pred_last.json")


def derive_factor(code, df_hfq=None, verbose=True):
    """从数据本身推导复权因子。

    方法：hfq 收盘价 / TDengine 原始收盘价 = 因子。
    后复权数据末日即实际市场价，因子通常为 1.0。
    """
    if classify_code(code):
        return 1.0

    # 从 TDengine 查询原始收盘价，与后复权收盘价对比推导
    if df_hfq is not None and not df_hfq.empty:
        try:
            conn = connect()
            try:
                r = conn.query(
                    f"select ts, close from tdx.k_{code}_1d order by ts"
                )
                rows = list(r)
                if rows:
                    raw_df = pd.DataFrame(rows, columns=['ts', 'close'])
                    raw_df['ts'] = pd.to_datetime(raw_df['ts']).dt.tz_localize(None)
                    raw_df = raw_df.set_index('ts').sort_index()
                    last_date = df_hfq.index[-1]
                    raw_before = raw_df[raw_df.index <= pd.Timestamp(last_date)]
                    if not raw_before.empty:
                        raw_close = float(raw_before.iloc[-1]["close"])
                        hfq_close = float(df_hfq.iloc[-1]["close"])
                        if raw_close > 0 and hfq_close > 0:
                            factor = hfq_close / raw_close
                            if verbose:
                                print(f"  复权因子(推导): {factor:.4f} (hfq={hfq_close:.2f} / raw={raw_close:.2f})")
                            return factor
            finally:
                conn.close()
        except Exception:
            pass

    if verbose:
        print(f"[derive_factor] ⚠️ {code} 复权因子获取失败，输出为后复权价格", file=sys.stderr)
    return 1.0


# sh999999 是通达信自选股中对上证综指的别名，实际数据在 sh000001
CODE_ALIASES = {"sh999999": "sh000001"}


def _resolve_alias(code):
    return CODE_ALIASES.get(code, code)


def _fetch_index_from_tdengine(code: str) -> "pd.DataFrame | None":
    """从 TDengine 拉取指数 1d K 线，返回 6 字段 DataFrame (open, high, low, close, vol, amt)。

    code 为带前缀原始代码（如 sh999999、sh000688、sz399006），表名 k_{code}_1d。
    """
    try:
        conn = connect()
        try:
            r = conn.query(
                f"select ts, open, high, low, close, volume, amount "
                f"from tdx.k_{code}_1d order by ts"
            )
            rows = list(r)
            if len(rows) < 50:
                return None
            df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "vol", "amt"])
            df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)
            df = df.set_index("ts").sort_index()
            df = df.astype({c: np.float64 for c in ["open", "high", "low", "close", "vol", "amt"]})
            return df
        finally:
            conn.close()
    except Exception:
        return None


def get_data(code):
    """Return DataFrame for a stock code (6-field, 后复权)。

    指数（sh000/sh999/sz399）直接从 TDengine 拉取最新数据；
    个股仍从本地 pickle 读取（同旧行为）。
    """
    resolved = _resolve_alias(code)
    if resolved.startswith("sh000") or resolved.startswith("sh999") or resolved.startswith("sz399"):
        df = _fetch_index_from_tdengine(code)
        if df is not None and len(df) > 0:
            return df
        # 兜底：本地 pickle（兼容旧工作流）
        if os.path.exists(SSE_DATA):
            with open(SSE_DATA, "rb") as f:
                d = pickle.load(f)
            if resolved in d:
                return d[resolved].copy()

    for fname in ["test_data.pkl", "val_data.pkl", "train_data.pkl", "data.pkl"]:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, "rb") as f:
                d = pickle.load(f)
            if resolved in d:
                return d[resolved].copy()
    return None


def load_model(device):
    tok = KronosTokenizer.from_pretrained(
        os.path.join(MODEL_DIR, "tdx_tokenizer", "checkpoints", "best_model")
    ).to(device)
    mod = Kronos.from_pretrained(
        os.path.join(MODEL_DIR, "tdx_predictor", "checkpoints", "best_model")
    ).to(device)
    return KronosPredictor(mod, tok, device=device, max_context=512)


def parse_zxg_blk(path=ZXG_BLK):
    """解析通达信自选股 blk 文件。"""
    if not os.path.exists(path):
        print(f"自选股文件不存在: {path}", file=sys.stderr)
        return []
    with open(path, "rb") as f:
        text = f.read().decode("ascii", errors="replace")
    codes = []
    for line in text.split():
        line = line.strip()
        if len(line) < 7:
            continue
        market = line[0]
        code_num = line[1:7]
        if not code_num.isdigit() or market not in ("0", "1"):
            continue
        prefix = "sh" if market == "1" else "sz"
        codes.append(f"{prefix}{code_num}")
    return codes



def classify_code(code):
    """判断是否为指数代码。"""
    return code.startswith("sh000") or code.startswith("sh999") or code.startswith("sz399")


def _get_stock_names(codes):
    """批量从 TDengine 获取股票名称。

    查询 tdx.stock_name 表，按 6 位数字代码匹配。
    支持别名解析（如 sh999999 → sh000001）。
    对 index 代码取首条（通常为指数名），对个股尝试取末条（避免指数/债券同名覆盖）。
    返回 {full_code: name}。
    """
    if not codes:
        return {}
    try:
        conn = connect()
        names = {}
        for code in codes:
            resolved = _resolve_alias(code)
            code_num = resolved[2:]  # strip sh/sz/bj prefix
            market = resolved[:2]
            try:
                # stock_name 同 code 跨市场重复（sh000001=上证指数/sz000001=平安银行），
                # 按 market 精确过滤（v0.13.7+）。
                r = conn.query(
                    f"select name from tdx.stock_name "
                    f"where code = '{code_num}' and market = '{market}'"
                )
                rows = list(r)
                names[code] = rows[0][0] if rows else code
            except Exception:
                names[code] = code
        conn.close()
        return names
    except Exception as e:
        print(f"  [名称获取] 连接失败: {e}", file=sys.stderr)
        return {c: c for c in codes}


def classify_prediction(cum_chg):
    """根据10日累计涨跌幅分类。"""
    if cum_chg > BULL_THRESHOLD:
        return "bull"
    elif cum_chg < BEAR_THRESHOLD:
        return "bear"
    return "neutral"


def _sort_key_for_code(code, all_forward):
    """排序键：0=指数, 1=看涨, 2=看平, 3=看跌。"""
    if classify_code(code):
        category = 0
    else:
        info = all_forward.get(code)
        chg = info["rows"][-1]["cum_chg"] if info and info.get("rows") else 0
        cls = classify_prediction(chg)
        category = {"bull": 1, "neutral": 2, "bear": 3}.get(cls, 2)
    chg_val = 0.0
    info = all_forward.get(code)
    if info and info.get("rows"):
        chg_val = -info["rows"][-1]["cum_chg"]
    return (category, chg_val)


def process_single(code, predictor, fresh_cache, factor=None):
    """处理单只股票：数据获取 → 因子 → 校准 → 三次预测 → 回测。"""
    from collections import Counter

    print(f"  处理 {code}...", flush=True)
    # 仅使用 fresh_cache 中的数据；missing_codes 已在 main() 中剔除并记入 errors，
    # 此处不再回退到本地 pickle。
    if code not in fresh_cache:
        return {"code": code, "error": "数据未在 fresh_cache 中（TDengine 无 1d 数据或导入异常，已跳过）"}
    df = fresh_cache[code]
    if df is None or len(df) == 0:
        return {"code": code, "error": "数据为空"}

    is_index = classify_code(code)
    if factor is None:
        factor = derive_factor(code, df)
    factor_ok = is_index or factor != 1.0
    if not factor_ok:
        print(f"  {code}: 复权因子获取失败，输出为后复权价格", file=sys.stderr)

    df_for_cal = df.rename(columns={"vol": "volume", "amt": "amount"})
    bias_correction = backtest_calibrate(
        predictor, df_for_cal, PRED_LEN,
        lookback=LOOKBACK, temperature=T, top_p=TOP_P,
        sample_count=SAMPLE_COUNT,
    )
    if factor_ok:
        bias_correction = bias_correction / factor

    all_rows = []
    directions = []
    for i in range(CONSENSUS_RUNS):
        rows, last_close_actual = run_prediction(predictor, df, factor)
        all_rows.append(rows)
        final_chg = rows[-1]["cum_chg"]
        directions.append(classify_prediction(final_chg))
        print(f"    第{i+1}次预测: 方向={_dir_label(directions[-1])}, "
              f"5日涨跌={final_chg*100:+.2f}%, 终点={rows[-1]['close']:.2f}", flush=True)

    dir_counts = Counter(directions)
    most_common_dir, most_common_count = dir_counts.most_common(1)[0]
    final_rows = all_rows[-1]

    metrics, n_win, conf = run_backtest(predictor, df, factor)

    return {
        "code": code,
        "factor_ok": factor_ok,
        "forward": {"rows": final_rows, "base": last_close_actual,
                     "factor": factor, "bias_correction": bias_correction,
                     "consensus_count": most_common_count,
                     "consensus_directions": directions},
        "backtest": {"metrics": metrics, "windows": n_win},
        "confidence": {"conf": conf, "base": last_close_actual},
    }


def _dir_label(cls):
    return {"bull": "看涨", "neutral": "看平", "bear": "看跌"}.get(cls, cls)


def _check_ratio(prev, curr, prev_label, curr_label, threshold):
    """单次比值检查，返回 (is_anomaly, detail) 或 (False, "")。"""
    if prev <= 0 or curr <= 0:
        return False, ""
    ratio = curr / prev
    if ratio > threshold or ratio < (1.0 / threshold):
        return True, (
            f"收盘价异常跳变 {ratio:.1f}x: "
            f"{prev:.2f} → {curr:.2f} "
            f"({prev_label} → {curr_label})"
        )
    return False, ""


def _detect_close_anomaly(df, threshold=5.0, ref_close=None, ref_label="", ref_date=None):
    """检测 DataFrame 收盘价是否存在异常跳变（脏数据）。

    检查维度：
    1. df 内部尾部 5 个交易日相邻收盘价比值
    2. 若提供 ref_close（旧数据末笔收盘价），检查跨边界跳变：
       比较 ref_close 与 df 中首条严格晚于 ref_date 的 bar。
       TDengine 回填场景下 df 为全历史（可能起始于 1990 年代），
       此时首条 bar 并非"新增"，不能与旧末笔比较。

    A 股涨跌停 ±10%（科创/创业 ±20%），阈值 5x 足以区分异常与正常波动。

    Returns: (is_anomaly: bool, detail: str)
    """
    if df is None or len(df) == 0:
        return False, ""

    closes = df["close"].values
    # 维度 1: 内部相邻比值
    if len(closes) >= 2:
        check_n = min(5, len(closes) - 1)
        for i in range(len(closes) - check_n, len(closes)):
            is_a, detail = _check_ratio(
                closes[i - 1], closes[i],
                df.index[i - 1].strftime('%Y-%m-%d'),
                df.index[i].strftime('%Y-%m-%d'),
                threshold,
            )
            if is_a:
                return True, detail

    # 维度 2: 跨边界比值（旧数据末笔 → 新增数据首笔）
    # 仅当存在严格晚于 ref_date 的新 bar 时才比较；否则说明导入数据
    # 完全落在旧数据时间范围内，不存在需要校验的"边界"。
    if ref_close is not None and ref_close > 0 and ref_date is not None:
        new_mask = df.index > ref_date  # ndarray of bool
        if new_mask.any():
            first_idx = int(np.argmax(new_mask))
            first_new = closes[first_idx]
            new_label = df.index[first_idx].strftime('%Y-%m-%d') if hasattr(df.index[first_idx], 'strftime') else str(df.index[first_idx])
            is_a, detail = _check_ratio(ref_close, first_new, ref_label, new_label, threshold)
            if is_a:
                return True, detail

    return False, ""


def ensure_fresh_data(codes):
    """检查数据新鲜度，过期则从 TDengine 导入最新数据并合并。

    返回 (fresh_cache, missing_codes)：
        fresh_cache: {code: DataFrame} — 可供预测的数据（含未过期标的，直接使用本地数据）
        missing_codes: set — 过期但 TDengine 中无 1d 数据（或导入失败）的标的，
                               调用方不应再回退到本地 pickle

    未过期（today - latest <= MAX_STALE_DAYS）的标的直接用本地数据填充 fresh_cache，
    避免在 main() 中再次回退到 pickle 读路径。
    """
    stale_codes = []
    existing_data = {}
    for code in codes:
        df = get_data(code)
        if df is None or len(df) == 0:
            stale_codes.append(code)
            continue
        latest = df.index.max().date()
        if (datetime.now().date() - latest).days > MAX_STALE_DAYS:
            stale_codes.append(code)
            existing_data[code] = df

    # 未过期标的直接用本地数据（这些是最新且可直接用的）
    fresh_cache = {}
    for code in codes:
        if code not in stale_codes:
            df = existing_data.get(code) or get_data(code)
            if df is not None and len(df) > 0:
                fresh_cache[code] = df

    if not stale_codes:
        print("数据均为最新，无需导入。")
        return fresh_cache, set()

    print(f"以下 {len(stale_codes)} 只股票数据需要更新: {stale_codes}")

    from scripts.tdx_import import TdxDataImporter
    missing_codes = set()
    anomaly_count = 0

    idx_codes = [c for c in stale_codes if classify_code(c)]
    stk_codes = [c for c in stale_codes if not classify_code(c)]

    for group, dtype in [(stk_codes, "back"), (idx_codes, "none")]:
        if not group:
            continue
        print(f"正在从 TDengine 导入 ({dtype})...")
        try:
            importer = TdxDataImporter(dividend_type=dtype)
            fresh_dataset = importer.build_dataset(
                group, period="1d", check_continuity=False,
            )
        except Exception as e:
            # TDengine 缺表或连接失败：整组标记为缺失，禁止回退本地 pickle
            print(f"⚠ 导入失败: {e}，{len(group)} 只标的将标记为缺失: {group}",
                  file=sys.stderr)
            missing_codes.update(group)
            continue

        for code in group:
            if code not in fresh_dataset:
                # TDengine 中无 1d 数据（表不存在）：标记缺失，不保留旧数据
                print(f"  ⚠ {code}: TDengine 中无 1d 数据（表不存在），标记为缺失",
                      file=sys.stderr)
                missing_codes.add(code)
                continue
            fresh_df = fresh_dataset[code]

            # 检测新导入数据的收盘价异常跳变（内部 + 跨边界）
            old_df = existing_data.get(code)
            ref_close = float(old_df["close"].iloc[-1]) if old_df is not None and len(old_df) > 0 else None
            ref_label = old_df.index[-1].strftime('%Y-%m-%d') if ref_close is not None else ""
            ref_date = old_df.index[-1] if old_df is not None and len(old_df) > 0 else None
            is_anomaly, anomaly_detail = _detect_close_anomaly(
                fresh_df, ref_close=ref_close, ref_label=ref_label, ref_date=ref_date,
            )
            if is_anomaly:
                anomaly_count += 1
                # 新数据异常：丢弃新数据并标记缺失（不再回退旧数据，避免用脏数据预测）
                print(f"  ⚠ {code}: 数据异常({anomaly_detail})，丢弃新数据并标记为缺失",
                      file=sys.stderr)
                missing_codes.add(code)
                continue

            if code in existing_data:
                merged = pd.concat([existing_data[code], fresh_df])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                fresh_cache[code] = merged
            else:
                fresh_cache[code] = fresh_df
            print(f"  {code}: 数据已更新至 {fresh_cache[code].index.max().strftime('%Y-%m-%d')}")

    if anomaly_count:
        print(f"  ⚠ 共 {anomaly_count} 只标的新数据异常，已丢弃并标记为缺失", file=sys.stderr)
    if missing_codes:
        print(f"  ⚠ 共 {len(missing_codes)} 只标的不可用（无 1d 数据或导入异常）: {sorted(missing_codes)}",
              file=sys.stderr)

    return fresh_cache, missing_codes


def apply_price_limits(rows, last_close_actual, limit_rate=LIMIT_RATE):
    """逐日应用涨跌停约束。"""
    lc = last_close_actual
    for r in rows:
        up, dn = lc * (1 + limit_rate), lc * (1 - limit_rate)
        for col in ["open", "high", "low", "close"]:
            r[col] = round(max(min(r[col], up), dn), 2)
        lc = r["close"]


def run_prediction(predictor, df, factor):
    """Forward predict next PRED_LEN days."""
    df = df.rename(columns={"vol": "volume", "amt": "amount"})
    context = df.iloc[-LOOKBACK:]
    last_date = pd.to_datetime(context.index[-1])
    last_close = context["close"].iloc[-1]

    today = pd.Timestamp.today().normalize()
    pred_start = max(last_date, today) + pd.Timedelta(days=1)
    future = pd.bdate_range(
        start=pred_start, periods=PRED_LEN,
        freq="C", weekmask="Mon Tue Wed Thu Fri"
    )
    x_ts = pd.Series(pd.to_datetime(context.index).values, name="timestamps")
    y_ts = pd.Series(future.values)

    pred = predictor.predict(
        df=context, x_timestamp=x_ts, y_timestamp=y_ts,
        pred_len=PRED_LEN, T=T, top_p=TOP_P,
        sample_count=SAMPLE_COUNT, verbose=False
    )

    last_close_actual = float(last_close) / factor
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

    apply_price_limits(rows, last_close_actual)
    return rows, last_close_actual


def run_backtest(predictor, df, factor):
    """Rolling backtest."""
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

    r = pd.DataFrame(results)
    n_win = len(r) // PRED_LEN
    metrics = {}
    for d in range(1, PRED_LEN + 1):
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
    conf = {}
    last_actual = float(df["close"].iloc[-1]) / factor
    for d in range(1, PRED_LEN + 1):
        if d in metrics:
            conf[f"d{d}"] = float(metrics[d]["mdape"]) * last_actual

    return metrics, n_win, conf


def format_table(rows, metrics=None, conf=None):
    """Format prediction rows as markdown table."""
    has_metrics = metrics and len(metrics) > 0
    has_conf = conf and len(conf) > 0

    header = "| 日期 | 开盘 | 最高 | 最低 | 收盘 | 累计涨跌 |"
    sep    = "|---|---|---|---|---|---|"
    if has_metrics:
        header += " MAPE | 准确率 | 高覆盖 | 低覆盖 |"
        sep    += "---|---|---|---|"
    if has_conf:
        header += " 置信± |"
        sep    += "---|"

    lines = [header, sep]
    for i, r in enumerate(rows):
        day = i + 1
        line = (
            f"| {r['date']} | {r['open']:.2f} | {r['high']:.2f} | "
            f"{r['low']:.2f} | {r['close']:.2f} | {r['cum_chg']*100:+.2f}% |"
        )
        if has_metrics and day in metrics:
            m = metrics[day]
            line += f" {m['mape']:.1%} | {m['dir']:.0%} | {m['hi_cov']:.0%} | {m['lo_cov']:.0%} |"
        elif has_metrics:
            line += " — | — | — | — |"
        if has_conf:
            key = f"d{day}"
            if key in conf:
                line += f" {conf[key]:.2f} |"
            else:
                line += " — |"
        lines.append(line)
    return "\n".join(lines)


def console_output(all_forward, all_bt, all_conf, all_codes, errors, all_stability=None):
    """控制台格式输出。"""
    if all_stability is None:
        all_stability = {}
    for code in all_codes:
        if code not in all_forward:
            continue
        info = all_forward[code]
        rows = info["rows"]
        base = info["base"]
        bc = info.get("bias_correction", 0.0)
        pred_len = len(rows)
        bt_metrics = all_bt[code]["metrics"] if code in all_bt else {}

        final_row = rows[-1]
        total_chg = final_row["cum_chg"]
        print(f"\n{'='*70}")
        print(f"  {info['name']} ({code}) 走势预测")
        print(f"{'='*70}")
        print(f"  基准收盘: {base:.2f}  5日涨跌: {total_chg*100:+.2f}% → 终点 {final_row['close']:.2f}")
        consensus_count = info.get("consensus_count")
        consensus_dirs = info.get("consensus_directions")
        if consensus_count is not None:
            dir_str = "/".join(_dir_label(d) for d in consensus_dirs)
            print(f"  三次预测方向: {dir_str} — 一致 {consensus_count}/{len(consensus_dirs)}")
        if abs(bc) > 0.01:
            print(f"  过去一个月模型偏差值: {bc:+.2f} (正值=模型预测偏低，负值=模型预测偏高)")
        if code in all_stability:
            print(f"  ⚠ 预测稳定性: {all_stability[code]}")
        print(f"{'='*70}")

        bt_conf = all_conf[code]["conf"] if code in all_conf else {}
        print(f"  {'日期':<14s} {'开盘':>8s} {'最高':>8s} {'最低':>8s} {'收盘':>8s} {'涨跌幅':>8s} {'MAPE':>7s} {'准确率':>6s} {'置信±':>7s}")
        print(f"  {'-'*83}")

        for i, r in enumerate(rows):
            prev = base if i == 0 else rows[i-1]["close"]
            daily_chg = (r["close"] - prev) / prev * 100
            day = i + 1
            mape_str = f"{bt_metrics[day]['mape']:>6.1%}" if day in bt_metrics else "     —"
            dir_str  = f"{bt_metrics[day]['dir']:>5.0%}" if day in bt_metrics else "    —"
            conf_key = f"d{day}"
            conf_str = f"{bt_conf[conf_key]:>6.2f}" if conf_key in bt_conf else "     —"
            print(f"  {str(r['date']):<14s} "
                  f"{r['open']:>8.2f} {r['high']:>8.2f} "
                  f"{r['low']:>8.2f} {r['close']:>8.2f} {daily_chg:>+7.2f}% "
                  f"{mape_str} {dir_str} {conf_str}")

        final = rows[-1]
        total_chg = (final["close"] - base) / base * 100
        print(f"  {'-'*54}")

        up_days = sum(1 for i in range(1, len(rows)) if rows[i]["close"] > rows[i-1]["close"])
        down_days = pred_len - 1 - up_days
        prices = [r["close"] for r in rows]
        pred_high = max(r["high"] for r in rows)
        pred_low = min(r["low"] for r in rows)
        print(f"\n  涨跌统计: {up_days}涨 / {down_days}跌")
        print(f"  预测区间: {pred_low:.2f} ~ {pred_high:.2f}")
        print(f"  波动幅度: {(pred_high - pred_low) / base * 100:.2f}%")

    if errors:
        print(f"\n错误:")
        for e in errors:
            print(f"  - {e}")


def load_last_predictions():
    """加载上次预测结果。"""
    if not os.path.exists(STABILITY_CACHE):
        return {}
    try:
        with open(STABILITY_CACHE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_predictions(all_forward):
    """保存本次预测结果。"""
    snapshot = {}
    for code, info in all_forward.items():
        if info.get("rows"):
            snapshot[code] = float(round(info["rows"][-1]["cum_chg"], 4))
    try:
        os.makedirs(os.path.dirname(STABILITY_CACHE), exist_ok=True)
        tmp_path = STABILITY_CACHE + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(snapshot, f)
        os.replace(tmp_path, STABILITY_CACHE)
    except Exception as e:
        print(f"[save_predictions] 保存失败: {e}", file=sys.stderr)


def _build_report(sorted_codes, all_forward, all_bt, all_conf, all_stability, errors, title, now):
    """生成 Markdown 报告。"""
    lines = []

    index_count = sum(1 for c in sorted_codes if classify_code(c) and c in all_forward)
    bull_count = sum(1 for c in sorted_codes if not classify_code(c) and c in all_forward
                     and classify_prediction(all_forward[c]["rows"][-1]["cum_chg"]) == "bull")
    neutral_count = sum(1 for c in sorted_codes if not classify_code(c) and c in all_forward
                        and classify_prediction(all_forward[c]["rows"][-1]["cum_chg"]) == "neutral")
    bear_count = sum(1 for c in sorted_codes if not classify_code(c) and c in all_forward
                     and classify_prediction(all_forward[c]["rows"][-1]["cum_chg"]) == "bear")

    lines.append(f"# {title}未来5日收盘价预测报告")
    lines.append("")
    lines.append(f"**时间**: {now} | **模型**: Kronos-base TDX后复权微调版 | **基准日**: 最近交易日")
    lines.append(f"**标的数**: {len([c for c in sorted_codes if c in all_forward])} | "
                 f"指数 {index_count} | 看涨 {bull_count} | 看平 {neutral_count} | 看跌 {bear_count}")
    lines.append("")
    lines.append("> 所有价格为实际市场价（已从后复权换算），已施加涨跌停约束。")
    lines.append("> 分类标准：10日累计涨跌幅 >3% 为看涨，<-3% 为看跌，其余为看平。")
    lines.append("")
    lines.append("---")
    lines.append("")

    section_labels = {0: "指数", 1: "看涨", 2: "看平", 3: "看跌"}
    toc_groups = {0: [], 1: [], 2: [], 3: []}
    for code in sorted_codes:
        if code not in all_forward:
            continue
        sk = _sort_key_for_code(code, all_forward)
        toc_groups[sk[0]].append(code)

    lines.append("## 目录")
    lines.append("")
    for sec_id in (0, 1, 2, 3):
        codes_in_sec = toc_groups.get(sec_id, [])
        if not codes_in_sec:
            continue
        label = section_labels[sec_id]
        items = []
        for code in codes_in_sec:
            info = all_forward[code]
            final = info["rows"][-1]
            chg_str = f"{final['cum_chg']*100:+.2f}%"
            items.append(f"[{info['name']} ({code}) {chg_str}](#{code})")
        lines.append(f"**{label}**: {' | '.join(items)}")
        lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 一、预测结果")
    lines.append("")
    current_section = -1
    for code in sorted_codes:
        if code not in all_forward:
            continue
        sk = _sort_key_for_code(code, all_forward)
        section = sk[0]
        if section != current_section:
            current_section = section
            lines.append(f"### {section_labels.get(section, '其他')}")
            lines.append("")
        info = all_forward[code]
        final = info["rows"][-1]
        cls_tag = ""
        if not classify_code(code):
            cls = classify_prediction(final["cum_chg"])
            cls_labels = {"bull": "看涨", "bear": "看跌"}
            cls_tag = f" [{cls_labels.get(cls, '')}]" if cls in cls_labels else ""
        lines.append(f'<a id="{code}"></a>')
        lines.append("")
        lines.append(f"#### {info['name']} ({code}){cls_tag}")
        lines.append("")
        lines.append(
            f"基准收盘: **{info['base']:.2f}** "
            f"5日涨跌: **{final['cum_chg']*100:+.2f}%** → 终点 **{final['close']:.2f}**"
        )
        consensus_count = info.get("consensus_count")
        consensus_dirs = info.get("consensus_directions")
        if consensus_count is not None:
            dir_str = "/".join(_dir_label(d) for d in consensus_dirs)
            lines.append(f"三次预测方向: {dir_str} — 一致 **{consensus_count}/{len(consensus_dirs)}**")
        bc = info.get("bias_correction", 0.0)
        if abs(bc) > 0.01:
            lines.append(f"过去一个月模型偏差值: **{bc:+.2f}** (正值=模型预测偏低，负值=模型预测偏高)")
        if code in all_stability:
            lines.append("")
            lines.append(f"> ⚠ 预测稳定性告警: {all_stability[code]}")
        lines.append("")
        bt_metrics = all_bt[code]["metrics"] if code in all_bt else None
        bt_conf = all_conf[code]["conf"] if code in all_conf else None
        lines.append(format_table(info["rows"], metrics=bt_metrics, conf=bt_conf))
        lines.append("")

    lines.append('---')
    lines.append('')
    lines.append('## 二、回测指标说明')
    lines.append('')
    lines.append('- **MAPE**: 平均绝对百分比误差（越小越好）')
    lines.append('- **准确率**: 涨跌方向准确率')
    lines.append('- **高覆盖**: 预测最高价 >= 实际最高价的比例')
    lines.append('- **低覆盖**: 预测最低价 <= 实际最低价的比例')
    lines.append('- **置信±**: 历史中位绝对误差（实际价格偏离预测的典型幅度）')
    lines.append('')

    relevant_stability = {c: msg for c, msg in all_stability.items() if c in all_forward}
    if relevant_stability:
        lines.append("---")
        lines.append("")
        lines.append("## 三、预测稳定性告警")
        lines.append("")
        lines.append("以下标的预测结果较上次发生显著变化，请关注：")
        lines.append("")
        for code, msg in relevant_stability.items():
            name = all_forward.get(code, {}).get("name", code)
            lines.append(f"- **{name}** ({code}): {msg}")
        lines.append("")

    relevant_errors = [e for e in errors
                       if any(c in e for c in sorted_codes)]
    if relevant_errors:
        lines.append("---")
        lines.append("")
        lines.append("## 错误")
        lines.append("")
        for e in relevant_errors:
            lines.append(f"- {e}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("> **免责**: 预测结果仅供模型能力验证，不构成投资建议。")

    return "\n".join(lines)


def check_stability(code, cum_chg, last_preds):
    """对比上次预测，返回稳定性标签。"""
    if code not in last_preds:
        return None
    prev = last_preds[code]
    diff = abs(cum_chg - prev)
    if diff > STABILITY_THRESHOLD:
        direction = "更悲观" if cum_chg < prev else "更乐观"
        return f"预测跳变 ({prev*100:+.1f}% → {cum_chg*100:+.1f}%, {direction})"
    prev_cls = classify_prediction(prev)
    curr_cls = classify_prediction(cum_chg)
    if prev_cls != curr_cls:
        labels = {"bull": "看涨", "bear": "看跌", "neutral": "看平"}
        return f"方向翻转 ({labels[prev_cls]} → {labels[curr_cls]})"
    return None


def main():
    parser = argparse.ArgumentParser(description="一键预测个股未来5日收盘价")
    parser.add_argument("codes", nargs="*", help="股票代码（可选，缺失时读取TDX自选股）")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-o", "--output", default=None, help="输出文件路径（默认自动生成）")
    parser.add_argument("--format", choices=["md", "console"], default="md",
                        help="输出格式: md=Markdown报告, console=控制台表格 (默认 md)")
    parser.add_argument("--no-import", action="store_true", help="跳过自动导入，使用现有数据")
    parser.add_argument("-n", type=int, default=None, help="并发线程数（因子获取）")
    args = parser.parse_args()

    global FACTOR_WORKERS
    if args.n is not None:
        FACTOR_WORKERS = args.n

    codes = args.codes
    from_zxg = False
    if not codes:
        zxg_codes = parse_zxg_blk()
        seen = set()
        codes = []
        for c in zxg_codes:
            if c not in seen:
                seen.add(c)
                codes.append(c)
        from_zxg = bool(zxg_codes)
        if not codes:
            print("未指定股票代码且自选股文件为空", file=sys.stderr)
            sys.exit(1)
        print(f"读取: 自选股 {len(zxg_codes)} 只")

    t0 = time.time()
    device = torch.device(args.device)
    print(f"Device: {device}")

    errors = []  # 贯穿 main 收集各类错误/告警，末尾统一输出

    # --no-import: 用户显式要求使用现有本地数据，不做新鲜度检查/DB 导入。
    # 此时仍从本地 pickle 读取（原有行为），缺失标的记入 errors。
    if args.no_import:
        missing_codes = set()
        fresh_cache = {}
        for code in codes:
            d = get_data(code)
            if d is not None and len(d) > 0:
                fresh_cache[code] = d
            else:
                missing_codes.add(code)
                print(f"  ⚠ {code}: 本地无数据，跳过")
    else:
        fresh_cache, missing_codes = ensure_fresh_data(codes)

    # 剔除无 1d 数据（或导入失败）的标的，禁止回退到本地 pickle；同时写入 errors
    if missing_codes:
        for code in sorted(missing_codes):
            errors.append(f"{code}: TDengine 无 1d 数据或导入失败，已禁用本地 pickle 回退，跳过")
        print(f"⚠ 剔除 {len(missing_codes)} 只不可用标的: {sorted(missing_codes)}")
        codes = [c for c in codes if c not in missing_codes]
        if not codes:
            print("所有标的均不可用，退出。", file=sys.stderr)
            sys.exit(1)

    print("获取复权因子...")
    factors = prefetch_factors(codes, fresh_cache)
    ok_count = sum(1 for _, ok in factors.values() if ok)
    fail_codes = [c for c, (_, ok) in factors.items() if not ok]
    if fail_codes:
        print(f"  ⚠ {len(fail_codes)} 只股票复权因子获取失败（将输出后复权价格）: {fail_codes}")
    print(f"  因子获取完成: {ok_count}/{len(codes)} 有效")

    # 仅使用 fresh_cache 中的数据。missing_codes 已在此前剔除并记入 errors，
    # 不再回退到本地 pickle（避免用陈旧数据填充预测）。
    all_data = {code: fresh_cache[code] for code in codes
                if code in fresh_cache and fresh_cache[code] is not None}
    if len(all_data) < len(codes):
        skipped = [c for c in codes if c not in all_data]
        # 防御性兜底：fresh_cache 缺失应已被 missing_codes 捕获，此处仅作保险
        for code in skipped:
            errors.append(f"{code}: fresh_cache 中无数据（未预期路径），跳过")

    from scripts.realtime import append_realtime_bars
    factor_map = {code: f for code, (f, _) in factors.items()}
    append_realtime_bars(list(all_data.keys()), all_data, factor_map)
    for code in all_data:
        fresh_cache[code] = all_data[code]

    print(f"加载微调模型...")
    predictor = load_model(device)

    all_forward = {}
    all_bt = {}
    all_conf = {}
    all_stability = {}

    last_preds = load_last_predictions()

    print("获取股票名称...")
    stock_names = _get_stock_names(codes)

    print(f"开始预测 (股票数={len(codes)})...")
    for code in codes:
        try:
            factor, _ = factors.get(code, (None, False))
            result = process_single(code, predictor, fresh_cache, factor=factor)
        except Exception as e:
            errors.append(f"{code}: {type(e).__name__}: {e}")
            continue

        if "error" in result:
            errors.append(f"{code}: {result['error']}")
            continue

        fwd = result["forward"]
        all_forward[code] = {"name": stock_names.get(code, code), "rows": fwd["rows"], "base": fwd["base"],
                             "factor": fwd["factor"], "bias_correction": fwd["bias_correction"],
                             "consensus_count": fwd.get("consensus_count"),
                             "consensus_directions": fwd.get("consensus_directions")}
        bt = result["backtest"]
        all_bt[code] = {"name": stock_names.get(code, code), "metrics": bt["metrics"], "windows": bt["windows"]}
        conf = result["confidence"]
        all_conf[code] = {"name": stock_names.get(code, code), "conf": conf["conf"], "base": conf["base"]}
        if not result.get("factor_ok", True):
            errors.append(f"{code}: 复权因子获取失败，输出为后复权价格")

        final_chg = fwd["rows"][-1]["cum_chg"]
        stab = check_stability(code, final_chg, last_preds)
        if stab:
            all_stability[code] = stab

    sorted_codes = sorted(codes, key=lambda c: _sort_key_for_code(c, all_forward))
    save_predictions(all_forward)

    elapsed = time.time() - t0

    if args.format == "console":
        console_output(all_forward, all_bt, all_conf, sorted_codes, errors, all_stability)
        print(f"\n总耗时: {elapsed:.1f}s")
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    today = datetime.now().strftime("%Y%m%d")

    if from_zxg:
        title = "自选股"
    else:
        title = "个股"
    report = _build_report(sorted_codes, all_forward, all_bt, all_conf, all_stability, errors, title, now)
    if args.output:
        out_path = args.output
    elif from_zxg:
        out_path = f"outputs/kronos-zxg-{today}.md"
    else:
        codes_str = "_".join(codes[:3])
        out_path = f"outputs/kronos_{codes_str}.md"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\n报告已保存: {out_path}")
    print(f"\n总耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

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

# mootdx.quotes 必须在 tdxdata 之前导入，否则 tdxdata 的依赖链会破坏 mootdx 初始化
from mootdx.quotes import Quotes as _MootdxQuotes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.kronos import KronosTokenizer, Kronos, KronosPredictor
from scripts.calibrate import backtest_calibrate

# ---------------------------------------------------------------------------
FACTOR_DIR = os.path.expanduser("~/.local/share/tdxcfv/drive_c/tc/.factor_cache")
FACTOR_CACHE_1D = os.path.join(FACTOR_DIR, ".factor_1d.json")
FACTOR_CACHE_DAYS = 30        # 复权因子缓存有效期（一个月）
FACTOR_WORKERS = 4            # 复权因子并发获取线程数（避免限速）


def _cache_month_key(date_str):
    """从 'YYYY-MM-DD' 字符串提取 'YYYY-MM' 月份键，无法解析时返回 None。"""
    try:
        key = date_str[:7]
        return key if len(key) == 7 else None
    except Exception:
        return None


def prefetch_factors(codes, fresh_cache):
    """批量获取复权因子（一个月缓存，4线程并发）。

    先检查本地 JSON 缓存（同月内有效），命中则跳过 derive_factor 的网络/TDX 读取。
    未命中的代码用 ThreadPoolExecutor(4) 并发推导因子，避免被数据源限速。
    若最新因子获取失败（返回 1.0）但存在旧缓存，则回退使用旧缓存值。

    返回: {code: (factor: float, ok: bool)}
        ok=True  因子有效（非 1.0 或是指数）
        ok=False 因子获取失败，需用后复权价格输出
    """
    # 读取缓存
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
    need_fetch = []  # 缓存过期/缺失的个股代码

    for code in codes:
        # 指数直接跳过
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

    # 并发推导未命中代码的因子
    if need_fetch:
        print(f"  并发获取复权因子: {len(need_fetch)} 只, {FACTOR_WORKERS} 线程")

        def _fetch_one(code):
            """单只股票因子推导（worker 内独立调用，mootdx 连接每次新建）。"""
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
                # 获取不到最新因子时，回退使用已缓存旧数据
                if factor == 1.0:
                    old = cache.get(code)
                    if old and old.get("factor", 1.0) != 1.0:
                        print(f"  {code}: 最新因子获取失败，使用旧缓存 {old['factor']:.4f}",
                              file=sys.stderr)
                        factor = old["factor"]
                result[code] = (factor, factor != 1.0)
                cache[code] = {"factor": factor, "date": today_str}

    # 持久化缓存
    os.makedirs(FACTOR_DIR, exist_ok=True)
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
TDX_DIR = os.path.expanduser("~/.local/share/tdxcfv/drive_c/tc/")
ZXG_BLK = os.path.expanduser("~/.local/share/tdxcfv/drive_c/tc/T0002/blocknew/zxg.blk")
MAX_STALE_DAYS = 0

# 10日涨跌幅分类阈值
BULL_THRESHOLD = 0.03   # >3% 看涨
BEAR_THRESHOLD = -0.03  # <-3% 看跌

LIMIT_RATE = 0.10                # 涨跌停幅度
CONSENSUS_RUNS = 3               # 连续预测次数（取多数一致）

STABILITY_THRESHOLD = 0.15       # 预测稳定性告警阈值（累计涨跌幅变化）
STABILITY_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "outputs", ".pred_last.json")


def derive_factor(code, df_hfq=None, verbose=True):
    """从数据本身推导复权因子，确保与训练/推理数据一致。

    核心问题：factor cache 可能被重新计算，导致与 pkl 数据中实际使用的
    factor 不一致（如 sh600353 cache=18.60 vs 隐含=10.71，偏差 42%）。

    推导方式：hfq 收盘价 / TDX 原始收盘价 = 因子（保证一致性）。
    回退：本地缓存 → 在线获取 → 1.0。
    """
    # 指数无需复权因子
    if classify_code(code):
        return 1.0

    # 方法1：从 hfq 数据与 TDX 原始数据对比推导（最可靠）
    if df_hfq is not None and not df_hfq.empty:
        try:
            from mootdx.reader import Reader
            reader = Reader.factory(market="std", tdxdir=TDX_DIR)
            raw_df = reader.daily(symbol=code[2:])
            if raw_df is not None and not raw_df.empty:
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
        except Exception:
            pass

    # 方法2：从本地缓存读取
    cache_file = os.path.join(FACTOR_DIR, f"{code}.pkl")
    if os.path.exists(cache_file):
        try:
            f = pd.read_pickle(cache_file)
            f.index = pd.to_datetime(f.index)
            factor = float(f.sort_index()["factor"].iloc[-1])
            if verbose:
                print(f"  复权因子(缓存): {factor:.4f} (可能与数据不一致)", file=sys.stderr)
            return factor
        except Exception:
            pass

    # 方法3：在线获取
    try:
        from mootdx.quotes import Quotes
        from tdxdata.sources.adjust import fetch_factor
        quotes = Quotes.factory(market='std')
        factor_df = fetch_factor(code, "hfq", quotes_client=quotes)
        if factor_df is not None and not factor_df.empty:
            os.makedirs(FACTOR_DIR, exist_ok=True)
            factor_df.to_pickle(cache_file)
            factor = float(factor_df.sort_index()["factor"].iloc[-1])
            if verbose:
                print(f"  复权因子(在线): {factor:.4f}", file=sys.stderr)
            return factor
    except Exception as e:
        if verbose:
            print(f"[derive_factor] 获取 {code} 复权因子失败: {type(e).__name__}: {e}", file=sys.stderr)

    if verbose:
        print(f"[derive_factor] ⚠️ {code} 复权因子获取失败，输出为后复权价格", file=sys.stderr)
    return 1.0


def get_data(code):
    """Return DataFrame for a stock code (6-field, 后复权)."""
    # Try SSE data first (for indices)
    if code.startswith("sh000") or code.startswith("sh999") or code.startswith("sz399"):
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


def parse_zxg_blk(path=ZXG_BLK):
    """解析通达信自选股 blk 文件，返回代码列表 (如 ['sh600353', 'sz002741'])。

    格式：每行一个条目，首位为市场编号 (0=深圳, 1=上海)，后6位为代码，以 \\r\\n 分隔。
    """
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


def fetch_stock_names(codes):
    """通过 mootdx 批量获取股票名称，返回 {code: name}。"""
    try:
        q = _MootdxQuotes.factory(market="std")
        name_map = {}
        for market_id in (0, 1):
            df = q.stocks(market=market_id)
            if df is not None and not df.empty:
                prefix = "sh" if market_id == 1 else "sz"
                for _, row in df.iterrows():
                    name_map[f"{prefix}{row['code']}"] = row["name"]
        return {c: name_map.get(c, c) for c in codes}
    except Exception as e:
        print(f"获取股票名称失败: {e}", file=sys.stderr)
        return {c: c for c in codes}


def classify_code(code):
    """判断是否为指数代码。

    通达信指数代码：sh999xxx（上证指数）、sz399xxx（深证指数）。
    sh000001 是平安银行，不是指数。
    """
    return code.startswith("sh999") or code.startswith("sz399")


def classify_prediction(cum_chg):
    """根据10日累计涨跌幅分类：bull / neutral / bear。"""
    if cum_chg > BULL_THRESHOLD:
        return "bull"
    elif cum_chg < BEAR_THRESHOLD:
        return "bear"
    return "neutral"


def _sort_key_for_code(code, all_forward):
    """排序键：0=指数, 1=看涨, 2=看平, 3=看跌；同组内按涨跌幅降序。"""
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
        chg_val = -info["rows"][-1]["cum_chg"]  # 降序
    return (category, chg_val)


def process_single(code, predictor, fresh_cache, factor=None):
    """处理单只股票：数据获取 → 因子 → 校准 → 三次预测 → 回测。返回结果字典或 None。

    连续预测 CONSENSUS_RUNS 次，若两次以上方向一致则选最后一次作为结果，
    否则仍取最后一次但标记不一致。
    """
    from collections import Counter

    print(f"  处理 {code}...", flush=True)
    df = fresh_cache[code] if code in fresh_cache else get_data(code)
    if df is None:
        return {"code": code, "error": "数据未找到"}

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

    # 连续预测 CONSENSUS_RUNS 次
    all_rows = []
    directions = []
    for i in range(CONSENSUS_RUNS):
        rows, last_close_actual = run_prediction(predictor, df, factor)
        all_rows.append(rows)
        final_chg = rows[-1]["cum_chg"]
        directions.append(classify_prediction(final_chg))
        print(f"    第{i+1}次预测: 方向={_dir_label(directions[-1])}, "
              f"5日涨跌={final_chg*100:+.2f}%, 终点={rows[-1]['close']:.2f}", flush=True)

    # 统计方向一致性
    dir_counts = Counter(directions)
    most_common_dir, most_common_count = dir_counts.most_common(1)[0]

    # 始终取最后一次预测结果
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
    """方向分类 → 中文标签。"""
    return {"bull": "看涨", "neutral": "看平", "bear": "看跌"}.get(cls, cls)


def ensure_fresh_data(codes):
    """检查数据新鲜度，过期则从TDX本地文件导入最新数据并合并。

    返回 {code: merged_df}（仅包含需要刷新的股票）。
    指数用 dividend_type="none"（不复权），个股用 "back"（后复权）。
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

    if not stale_codes:
        print("数据均为最新，无需导入。")
        return {}

    print(f"以下 {len(stale_codes)} 只股票数据需要更新: {stale_codes}")

    from scripts.tdx_import import TdxDataImporter
    result = {}

    # 分组：指数不复权，个股后复权
    idx_codes = [c for c in stale_codes if classify_code(c)]
    stk_codes = [c for c in stale_codes if not classify_code(c)]

    for group, dtype in [(stk_codes, "back"), (idx_codes, "none")]:
        if not group:
            continue
        print(f"正在从本地TDX数据导入 ({dtype})...")
        try:
            importer = TdxDataImporter(tdxdir=TDX_DIR, dividend_type=dtype)
            fresh_dataset = importer.build_dataset(
                group, period="1d", check_continuity=False,
            )
        except Exception as e:
            print(f"导入失败: {e}，将使用现有数据。")
            continue

        for code in group:
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


def apply_price_limits(rows, last_close_actual, limit_rate=LIMIT_RATE):
    """逐日应用涨跌停约束（实际价格空间）。"""
    lc = last_close_actual
    for r in rows:
        up, dn = lc * (1 + limit_rate), lc * (1 - limit_rate)
        for col in ["open", "high", "low", "close"]:
            r[col] = round(max(min(r[col], up), dn), 2)
        lc = r["close"]


def run_prediction(predictor, df, factor):
    """Forward predict next PRED_LEN days. Returns list of dicts."""
    df = df.rename(columns={"vol": "volume", "amt": "amount"})
    context = df.iloc[-LOOKBACK:]
    last_date = pd.to_datetime(context.index[-1])
    last_close = context["close"].iloc[-1]

    # 预测起点：若数据已包含今天（收盘后），则从明天开始；否则从数据末日次日开始
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
    # Confidence: MdAPE × 最新市场价（MdAPE 是尺度不变量，正确反映中位相对误差）
    conf = {}
    last_actual = float(df["close"].iloc[-1]) / factor
    for d in range(1, PRED_LEN + 1):
        if d in metrics:
            conf[f"d{d}"] = float(metrics[d]["mdape"]) * last_actual

    return metrics, n_win, conf


def format_table(rows, metrics=None, conf=None):
    """Format prediction rows as markdown table with merged accuracy."""
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
            # D1→d1, D5→d5, etc. fallback to nearest available
            key = f"d{day}"
            if key in conf:
                line += f" {conf[key]:.2f} |"
            else:
                line += " — |"
        lines.append(line)
    return "\n".join(lines)


def console_output(all_forward, all_bt, all_conf, all_codes, errors, all_stability=None):
    """控制台格式输出：预测表格 + 涨跌统计 + 回测摘要。"""
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
        # 第一行：基准收盘 + 5日涨跌 → 终点
        print(f"  基准收盘: {base:.2f}  5日涨跌: {total_chg*100:+.2f}% → 终点 {final_row['close']:.2f}")
        # 第二行：三次预测方向
        consensus_count = info.get("consensus_count")
        consensus_dirs = info.get("consensus_directions")
        if consensus_count is not None:
            dir_str = "/".join(_dir_label(d) for d in consensus_dirs)
            print(f"  三次预测方向: {dir_str} — 一致 {consensus_count}/{len(consensus_dirs)}")
        # 第三行：模型偏差值
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

        # 涨跌统计
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
    """加载上次预测结果，用于稳定性对比。返回 {code: cum_chg} 或 {}。"""
    if not os.path.exists(STABILITY_CACHE):
        return {}
    try:
        with open(STABILITY_CACHE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_predictions(all_forward):
    """保存本次预测结果，供下次稳定性对比（原子写入）。"""
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
    """根据预测结果生成 Markdown 报告文本。"""
    lines = []

    # 汇总统计
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
    lines.append("> 所有价格为实际市场价（已从后复权换算），已施加涨跌停约束。涨跌幅 = (预测收盘 - 基准收盘) / 基准收盘。")
    lines.append("> 分类标准：10日累计涨跌幅 >3% 为看涨，<-3% 为看跌，其余为看平。")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 目录区 — 按分类分组，链接到各股票锚点
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

    # Forward predictions — 按分类分组输出
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
        # HTML 锚点 + 标题，确保目录链接可跳转
        lines.append(f'<a id="{code}"></a>')
        lines.append("")
        lines.append(f"#### {info['name']} ({code}){cls_tag}")
        lines.append("")
        # 第一行：基准收盘 + 5日涨跌 → 终点
        lines.append(
            f"基准收盘: **{info['base']:.2f}** "
            f"5日涨跌: **{final['cum_chg']*100:+.2f}%** → 终点 **{final['close']:.2f}**"
        )
        # 第二行：三次预测方向
        consensus_count = info.get("consensus_count")
        consensus_dirs = info.get("consensus_directions")
        if consensus_count is not None:
            dir_str = "/".join(_dir_label(d) for d in consensus_dirs)
            lines.append(f"三次预测方向: {dir_str} — 一致 **{consensus_count}/{len(consensus_dirs)}**")
        # 第三行：模型偏差值
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

    # Accuracy note
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

    # Stability alerts
    # 仅筛选当前 sorted_codes 中的稳定性告警
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

    # Errors (仅当前 sorted_codes 相关)
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
    """对比上次预测，返回稳定性标签。None=稳定, str=告警描述。"""
    if code not in last_preds:
        return None
    prev = last_preds[code]
    diff = abs(cum_chg - prev)
    if diff > STABILITY_THRESHOLD:
        direction = "更悲观" if cum_chg < prev else "更乐观"
        return f"预测跳变 ({prev*100:+.1f}% → {cum_chg*100:+.1f}%, {direction})"
    # 方向翻转
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
    args = parser.parse_args()

    # 参数缺失时读取 TDX 自选股
    codes = args.codes
    from_zxg = False
    if not codes:
        zxg_codes = parse_zxg_blk()
        # 去重（保持顺序）
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

    # 获取股票名称
    print("获取股票名称...")
    name_map = fetch_stock_names(codes)

    # 检查并导入最新数据
    fresh_cache = {} if args.no_import else ensure_fresh_data(codes)

    # 批量预取复权因子（1日缓存，因子完整后再进行后续预测）
    print("获取复权因子...")
    factors = prefetch_factors(codes, fresh_cache)
    ok_count = sum(1 for _, ok in factors.values() if ok)
    fail_codes = [c for c, (_, ok) in factors.items() if not ok]
    if fail_codes:
        print(f"  ⚠ {len(fail_codes)} 只股票复权因子获取失败（将输出后复权价格）: {fail_codes}")
    print(f"  因子获取完成: {ok_count}/{len(codes)} 有效")

    # 盘中追加实时行情
    from scripts.realtime import append_realtime_bars
    all_data = {}
    for code in codes:
        if code in fresh_cache and fresh_cache[code] is not None:
            all_data[code] = fresh_cache[code]
        else:
            d = get_data(code)
            if d is not None:
                all_data[code] = d
    factor_map = {code: f for code, (f, _) in factors.items()}
    append_realtime_bars(codes, all_data, factor_map)
    # 将追加后的数据写回 fresh_cache，供 process_single 使用
    for code in codes:
        if code in all_data and all_data[code] is not None:
            fresh_cache[code] = all_data[code]
    print("获取复权因子...")
    factors = prefetch_factors(codes, fresh_cache)
    ok_count = sum(1 for _, ok in factors.values() if ok)
    fail_codes = [c for c, (_, ok) in factors.items() if not ok]
    if fail_codes:
        print(f"  ⚠ {len(fail_codes)} 只股票复权因子获取失败（将输出后复权价格）: {fail_codes}")
    print(f"  因子获取完成: {ok_count}/{len(codes)} 有效")

    print(f"加载微调模型...")
    predictor = load_model(device)

    all_forward = {}
    all_bt = {}
    all_conf = {}
    all_stability = {}
    errors = []

    # 加载上次预测用于稳定性对比
    last_preds = load_last_predictions()

    # 顺序处理
    print(f"开始预测 (股票数={len(codes)})...")
    for code in codes:
        try:
            factor, _ = factors.get(code, (None, False))
            result = process_single(code, predictor, fresh_cache, factor=factor)
        except Exception as e:
            errors.append(f"{name_map.get(code, code)} ({code}): {type(e).__name__}: {e}")
            continue

        if "error" in result:
            errors.append(f"{name_map.get(code, code)} ({code}): {result['error']}")
            continue

        name = name_map.get(code, code)
        fwd = result["forward"]
        all_forward[code] = {"name": name, "rows": fwd["rows"], "base": fwd["base"],
                             "factor": fwd["factor"], "bias_correction": fwd["bias_correction"],
                             "consensus_count": fwd.get("consensus_count"),
                             "consensus_directions": fwd.get("consensus_directions")}
        bt = result["backtest"]
        all_bt[code] = {"name": name, "metrics": bt["metrics"], "windows": bt["windows"]}
        conf = result["confidence"]
        all_conf[code] = {"name": name, "conf": conf["conf"], "base": conf["base"]}
        if not result.get("factor_ok", True):
            errors.append(f"{name} ({code}): 复权因子获取失败，输出为后复权价格")

        # 稳定性检查
        final_chg = fwd["rows"][-1]["cum_chg"]
        stab = check_stability(code, final_chg, last_preds)
        if stab:
            all_stability[code] = stab

    # 按分类排序：指数 → 看涨 → 看平 → 看跌
    sorted_codes = sorted(codes, key=lambda c: _sort_key_for_code(c, all_forward))

    # 保存本次预测供下次稳定性对比
    save_predictions(all_forward)

    elapsed = time.time() - t0

    # --- Output ---
    if args.format == "console":
        console_output(all_forward, all_bt, all_conf, sorted_codes, errors, all_stability)
        print(f"\n总耗时: {elapsed:.1f}s")
        return

    # --- Generate markdown report ---
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

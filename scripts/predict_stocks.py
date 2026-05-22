#!/usr/bin/env python3
"""
一键预测脚本 — 使用微调模型预测个股/指数未来3日走势（3次预测方向一致性校验）。

用法:
    python scripts/predict_stocks.py                                # 读取TDX自选股，输出报告
    python scripts/predict_stocks.py sh600000 sz002741              # 预测个股，输出 MD 报告
    python scripts/predict_stocks.py --format console               # 读取自选股，控制台表格
    python scripts/predict_stocks.py sh600353                       # 预测个股

个股价格已换算为实际市场价（不复权）。指数为实际点位。
报告按 指数→看涨→看平→看跌 排列。

特性:
    - 每次连续预测3次，方向一致时给出平均结果并重点标注
    - 施加10%涨跌停约束，预测价格不会超出日涨跌停范围
    - 模型偏差值仅作参考，不自动修正预测收盘价
    - 与上次预测结果对比，标注稳定性告警（预测跳变/方向翻转）
"""

import argparse, os, sys, pickle, json
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


def prefetch_factors(codes, fresh_cache):
    """批量获取复权因子（1日缓存），完成后检查完整性。

    先检查本地 JSON 缓存（当日有效），命中则跳过网络/TDX 读取。
    因子获取失败不写入缓存，下次重试。

    返回: {code: (factor: float, ok: bool)}
        ok=True  因子有效（含指数和因子确实为 1.0 的个股）
        ok=False 获取失败，需用后复权价格输出
    """
    cache = {}
    if os.path.exists(FACTOR_CACHE_1D):
        try:
            with open(FACTOR_CACHE_1D, "r") as f:
                cache = json.load(f)
        except Exception:
            cache = {}
    today_str = str(datetime.now().date())

    result = {}
    failed = []

    for code in codes:
        # 指数类无需因子
        if is_index_like(code):
            result[code] = (1.0, True)
            continue

        # 命中当日缓存
        cached = cache.get(code)
        if cached and cached.get("date") == today_str:
            factor = cached["factor"]
            ok = cached.get("ok", factor != 1.0)
            result[code] = (factor, ok)
            print(f"  {code}: 复权因子(1日缓存)={factor:.4f}")
            continue

        # 推导因子
        df = fresh_cache.get(code) if fresh_cache else None
        if df is None:
            df = get_data(code)
        if df is not None:
            factor, ok = derive_factor(code, df, verbose=False)
        else:
            factor, ok = 1.0, False

        result[code] = (factor, ok)

        if ok:
            cache[code] = {"factor": factor, "date": today_str, "ok": True}
        else:
            failed.append(code)

    # 持久化缓存（仅成功项）
    os.makedirs(FACTOR_DIR, exist_ok=True)
    try:
        with open(FACTOR_CACHE_1D, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass

    # ── 完整性报告 ──
    total = len(codes)
    idx_count = sum(1 for c in codes if is_index_like(c))
    stk_count = total - idx_count
    ok_count = sum(1 for _, ok in result.values() if ok)
    fail_count = stk_count - (ok_count - idx_count)

    print(f"  因子获取完成: {total} 标的")
    print(f"    指数(无需因子): {idx_count}")
    print(f"    个股成功: {stk_count - fail_count}/{stk_count}"
          + (f"  ⚠ 失败: {failed}" if failed else "  ✓ 全部成功"))
    if failed:
        print(f"    失败标的将使用后复权价格输出")

    return result

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tdx_import", "1d")
SSE_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tdx_import_sse", "1d", "data.pkl")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "tdx_finetune")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _report_version():
    """从 git tag 获取报告版本号，回退到硬编码版本。"""
    try:
        import subprocess
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
        return tag
    except Exception:
        return "1.6.1"

LOOKBACK = 90
PRED_LEN = 3
N_RUNS = 3                       # 连续预测次数，方向一致时取平均
T = 0.6
TOP_P = 0.9
SAMPLE_COUNT = 5
BACKTEST_SAMPLE_COUNT = 3
BACKTEST_WINDOWS = 30
TDX_DIR = os.path.expanduser("~/.local/share/tdxcfv/drive_c/tc/")
ZXG_BLK = os.path.expanduser("~/.local/share/tdxcfv/drive_c/tc/T0002/blocknew/zxg.blk")
MAX_STALE_DAYS = 0

# 3日涨跌幅分类阈值
BULL_THRESHOLD = 0.03   # >3% 看涨
BEAR_THRESHOLD = -0.03  # <-3% 看跌

LIMIT_RATE = 0.10                # 涨跌停幅度

STABILITY_THRESHOLD = 0.15       # 预测稳定性告警阈值（累计涨跌幅变化）
STABILITY_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "outputs", ".pred_last.json")


def derive_factor(code, df_hfq=None, verbose=True):
    """从数据本身推导复权因子，确保与训练/推理数据一致。

    推导方式（优先级）：hfq收盘/TDX原始收盘 → 本地pkl缓存 → 在线获取。

    Returns:
        (factor: float, ok: bool)
        ok=True  因子获取成功（含因子确实为 1.0 的情况）
        ok=False 获取失败，回退为 1.0，需用后复权价格输出
    """
    # 指数类无需复权因子
    if is_index_like(code):
        return 1.0, True

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
                        return factor, True
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
            return factor, True
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
            return factor, True
    except Exception as e:
        if verbose:
            print(f"[derive_factor] 获取 {code} 复权因子失败: {type(e).__name__}: {e}", file=sys.stderr)

    if verbose:
        print(f"[derive_factor] ⚠️ {code} 复权因子获取失败，输出为后复权价格", file=sys.stderr)
    return 1.0, False


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
    """判断是否为指数代码（上证/深证/创业板）。"""
    return code.startswith("sh000") or code.startswith("sh999") or code.startswith("sz399")


def is_index_like(code):
    """判断是否为指数类标的（含板块指数），此类不需要复权因子。"""
    return (code.startswith("sh000") or code.startswith("sh999")
            or code.startswith("sz399") or code.startswith("sh880")
            or code.startswith("sz880") or code.startswith("sh881"))


def classify_prediction(cum_chg):
    """根据3日累计涨跌幅分类：bull / neutral / bear。"""
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
    """处理单只股票：数据获取 → 因子 → 校准 → 预测 → 回测。返回结果字典或 None。

    factor: 预计算的复权因子（由 prefetch_factors 提供），为 None 时内部推导。
    """
    print(f"  处理 {code}...", flush=True)
    df = fresh_cache[code] if code in fresh_cache else get_data(code)
    if df is None:
        return {"code": code, "error": "数据未找到"}

    is_index = is_index_like(code)
    if factor is None:
        factor, factor_ok = derive_factor(code, df)
    else:
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

    avg_rows, all_runs_data, consistent, directions, last_close_actual = \
        run_prediction_multi(predictor, df, factor)

    metrics, n_win, conf = run_backtest(predictor, df, factor)

    return {
        "code": code,
        "factor_ok": factor_ok,
        "forward": {"rows": avg_rows, "base": last_close_actual,
                     "factor": factor, "bias_correction": bias_correction,
                     "consistent": consistent, "directions": directions,
                     "all_runs": all_runs_data},
        "backtest": {"metrics": metrics, "windows": n_win},
        "confidence": {"conf": conf, "base": last_close_actual},
    }


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

    # 分组：指数/板块指数不复权，个股后复权
    idx_codes = [c for c in stale_codes if is_index_like(c)]
    stk_codes = [c for c in stale_codes if not is_index_like(c)]

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


def run_prediction_multi(predictor, df, factor, n_runs=N_RUNS):
    """连续运行 n_runs 次预测，检查方向一致性，返回平均结果。

    Returns:
        avg_rows: 平均预测行列表（方向一致时用于展示）
        all_runs_data: [(rows, last_close_actual), ...] 每次运行的原始结果
        consistent: bool，方向是否一致
        directions: [str, ...] 每次运行的方向分类
        last_close_actual: 基准收盘价（实际价格空间）
    """
    all_runs_data = []
    for run_i in range(n_runs):
        print(f"    第 {run_i + 1}/{n_runs} 次预测...", end="", flush=True)
        rows, last_close = run_prediction(predictor, df, factor)
        all_runs_data.append((rows, last_close))
        direction = classify_prediction(rows[-1]["cum_chg"])
        label = {"bull": "看涨", "bear": "看跌", "neutral": "看平"}[direction]
        print(f" {label} ({rows[-1]['cum_chg'] * 100:+.2f}%)")

    directions = [classify_prediction(r[-1]["cum_chg"]) for r, _ in all_runs_data]
    consistent = len(set(directions)) == 1
    last_close_actual = all_runs_data[0][1]

    # 计算平均价格
    avg_rows = []
    for i in range(PRED_LEN):
        avg_row = {}
        for key in ["open", "high", "low", "close"]:
            vals = [r[i][key] for r, _ in all_runs_data]
            avg_row[key] = round(sum(vals) / n_runs, 2)
        avg_row["date"] = all_runs_data[0][0][i]["date"]
        avg_row["cum_chg"] = (avg_row["close"] - last_close_actual) / last_close_actual
        avg_rows.append(avg_row)

    # 重新施加涨跌停约束（平均后可能略微超出）
    apply_price_limits(avg_rows, last_close_actual)

    if consistent:
        print(f"    ✓ 方向一致 ({label})，使用平均结果")
    else:
        dirs_str = ", ".join({"bull": "看涨", "bear": "看跌", "neutral": "看平"}[d]
                             for d in directions)
        print(f"    ⚠ 方向不一致: {dirs_str}")

    return avg_rows, all_runs_data, consistent, directions, last_close_actual


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
    for d in [1, 2, 3]:
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
    # 中位误差（实际价格空间）
    conf = {}
    for d in [1, 2, 3]:
        day_d = r[r["day"] == d]
        if len(day_d) > 0:
            conf[f"d{d}"] = float(np.median(np.abs(day_d["pc_hfq"] - day_d["ac_hfq"]))) / factor

    return metrics, n_win, conf


def format_table(rows, metrics=None, conf=None):
    """Format prediction rows as markdown table, with optional backtest metrics and confidence."""
    has_m = metrics is not None
    has_c = conf is not None
    header = "| 日期 | 开盘 | 最高 | 最低 | 收盘 | 累计涨跌 |"
    sep = "|---|---|---|---|---|---|"
    if has_m:
        header += " MAPE | MdAPE | <3% | 方向 |"
        sep += "---|---|---|---|"
    if has_c:
        header += " 中位误差 |"
        sep += "---|"
    lines = [header, sep]
    for i, r in enumerate(rows):
        day = i + 1
        row = (f"| {r['date']} | {r['open']:.2f} | {r['high']:.2f} | "
               f"{r['low']:.2f} | {r['close']:.2f} | {r['cum_chg']*100:+.2f}% |")
        if has_m and day in metrics:
            m = metrics[day]
            row += f" {m['mape']:.1%} | {m['mdape']:.1%} | {m['lt3']:.0%} | {m['dir']:.0%} |"
        elif has_m:
            row += " — | — | — | — |"
        if has_c:
            err = conf.get(f"d{day}", None)
            row += f" ±{err:.2f} |" if err is not None else " — |"
        lines.append(row)
    return "\n".join(lines)


def console_output(all_forward, all_bt, all_conf, all_codes, errors, all_stability=None):
    """控制台格式输出：预测表格 + 方向一致性 + 涨跌统计 + 回测摘要。"""
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
        consistent = info.get("consistent", False)
        directions = info.get("directions", [])
        all_runs = info.get("all_runs", [])

        print(f"\n{'='*70}")
        print(f"  {info['name']} ({code}) 走势预测")
        print(f"{'='*70}")
        print(f"  基准收盘价: {base:.2f}")
        if abs(bc) > 0.01:
            print(f"  过去一个月模型偏差值: {bc:+.2f} (正值=模型预测偏低，负值=模型预测偏高)")
        if code in all_stability:
            print(f"  ⚠ 预测稳定性: {all_stability[code]}")

        # 方向一致性
        if not classify_code(code):
            if consistent:
                d = directions[0]
                label = {"bull": "看涨", "bear": "看跌", "neutral": "看平"}[d]
                print(f"  ★ 方向一致 (3/3): {label}")
            else:
                dirs_str = ", ".join(
                    {"bull": "看涨", "bear": "看跌", "neutral": "看平"}[d] for d in directions
                )
                print(f"  ⚠ 方向分歧: {dirs_str}")
                for i, (rows_run, _) in enumerate(all_runs):
                    print(f"     #{i + 1}: {rows_run[-1]['cum_chg'] * 100:+.2f}%")

        print(f"{'='*70}")
        print(f"  {'日期':<14s} {'开盘':>8s} {'最高':>8s} {'最低':>8s} {'收盘':>8s} {'涨跌幅':>8s}")
        print(f"  {'-'*54}")

        for i, r in enumerate(rows):
            prev = base if i == 0 else rows[i-1]["close"]
            daily_chg = (r["close"] - prev) / prev * 100
            print(f"  {str(r['date']):<14s} "
                  f"{r['open']:>8.2f} {r['high']:>8.2f} "
                  f"{r['low']:>8.2f} {r['close']:>8.2f} {daily_chg:>+7.2f}%")

        final = rows[-1]
        total_chg = (final["close"] - base) / base * 100
        print(f"  {'-'*54}")
        print(f"  {pred_len}日预测涨跌幅: {total_chg:+.2f}%")
        print(f"  预测终点: {final['close']:.2f} (起始: {base:.2f})")
        print(f"{'='*70}")

        # 涨跌统计
        up_days = sum(1 for i in range(1, len(rows)) if rows[i]["close"] > rows[i-1]["close"])
        down_days = pred_len - 1 - up_days
        prices = [r["close"] for r in rows]
        pred_high = max(r["high"] for r in rows)
        pred_low = min(r["low"] for r in rows)
        print(f"\n  涨跌统计: {up_days}涨 / {down_days}跌")
        print(f"  预测区间: {pred_low:.2f} ~ {pred_high:.2f}")
        print(f"  波动幅度: {(pred_high - pred_low) / base * 100:.2f}%")

        # 回测指标摘要
        if code in all_bt and all_bt[code]["metrics"]:
            m = all_bt[code]["metrics"]
            if 1 in m:
                print(f"  回测 D1 MAPE: {m[1]['mape']:.1%}  方向准确率: {m[1]['dir']:.0%}")
            if 2 in m:
                print(f"  回测 D2 MAPE: {m[2]['mape']:.1%}  方向准确率: {m[2]['dir']:.0%}")
            if 3 in m:
                print(f"  回测 D3 MAPE: {m[3]['mape']:.1%}  方向准确率: {m[3]['dir']:.0%}")

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
    parser = argparse.ArgumentParser(description="一键预测个股未来3日收盘价（3次预测方向一致性校验）")
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
        codes = parse_zxg_blk()
        if not codes:
            print("未指定股票代码且自选股文件为空或不存在", file=sys.stderr)
            sys.exit(1)
        from_zxg = True
        print(f"从自选股读取 {len(codes)} 只股票")

    device = torch.device(args.device)
    print(f"Device: {device}")

    # 获取股票名称
    print("获取股票名称...")
    name_map = fetch_stock_names(codes)

    # 检查并导入最新数据
    fresh_cache = {} if args.no_import else ensure_fresh_data(codes)

    # 批量预取复权因子（1日缓存，含完整性检查）
    print("获取复权因子...")
    factors = prefetch_factors(codes, fresh_cache)

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
                             "consistent": fwd.get("consistent", False),
                             "directions": fwd.get("directions", []),
                             "all_runs": fwd.get("all_runs", [])}
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

    # --- Output ---
    if args.format == "console":
        console_output(all_forward, all_bt, all_conf, sorted_codes, errors, all_stability)
        return

    # --- Generate markdown report ---
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    today = datetime.now().strftime("%Y%m%d")
    lines = []

    # 汇总统计
    index_count = sum(1 for c in sorted_codes if classify_code(c) and c in all_forward)
    bull_count = sum(1 for c in sorted_codes if not classify_code(c) and c in all_forward
                     and classify_prediction(all_forward[c]["rows"][-1]["cum_chg"]) == "bull")
    neutral_count = sum(1 for c in sorted_codes if not classify_code(c) and c in all_forward
                        and classify_prediction(all_forward[c]["rows"][-1]["cum_chg"]) == "neutral")
    bear_count = sum(1 for c in sorted_codes if not classify_code(c) and c in all_forward
                     and classify_prediction(all_forward[c]["rows"][-1]["cum_chg"]) == "bear")
    consistent_count = sum(1 for c in sorted_codes if c in all_forward
                           and all_forward[c].get("consistent", False))

    title = "自选股" if from_zxg else "个股"
    lines.append(f"# {title}未来3日收盘价预测报告")
    lines.append("")
    lines.append(f"**时间**: {now} | **模型**: Kronos-base TDX后复权微调版 | **基准日**: 最近交易日")
    lines.append(f"**版本**: {_report_version()}")
    lines.append(f"**标的数**: {len(all_forward)} | "
                 f"指数 {index_count} | 看涨 {bull_count} | 看平 {neutral_count} | 看跌 {bear_count}")
    lines.append(f"**方向一致**: {consistent_count}/{max(1, len(all_forward) - index_count)} "
                 f"({consistent_count / max(1, len(all_forward) - index_count) * 100:.0f}%)")
    lines.append("")
    lines.append("> 所有价格为实际市场价（已从后复权换算），已施加涨跌停约束。涨跌幅 = (预测收盘 - 基准收盘) / 基准收盘。")
    lines.append("> 分类标准：3日累计涨跌幅 >3% 为看涨，<-3% 为看跌，其余为看平。")
    lines.append("> 每只标的连续预测3次，方向一致(3/3)时取平均结果并 ★ 标注。")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Forward predictions — 按分类分组输出
    lines.append("## 一、预测结果")
    lines.append("")
    section_labels = {0: "指数", 1: "看涨", 2: "看平", 3: "看跌"}
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
        consistent = info.get("consistent", False)
        directions = info.get("directions", [])
        all_runs = info.get("all_runs", [])

        cls_tag = ""
        if not classify_code(code):
            cls = classify_prediction(final["cum_chg"])
            cls_labels = {"bull": "看涨", "bear": "看跌"}
            cls_tag = f" [{cls_labels.get(cls, '')}]" if cls in cls_labels else ""

        # 方向一致性标注
        consistency_mark = ""
        if not classify_code(code):
            if consistent:
                consistency_mark = " ★ 方向一致"
            else:
                dirs_str = " → ".join(
                    {"bull": "看涨", "bear": "看跌", "neutral": "看平"}[d] for d in directions
                )
                consistency_mark = f" ⚠ 方向分歧({dirs_str})"

        lines.append(f"#### {info['name']} ({code}){cls_tag}{consistency_mark}")
        lines.append("")
        lines.append(f"基准收盘: **{info['base']:.2f}**")
        bc = info.get("bias_correction", 0.0)
        if abs(bc) > 0.01:
            lines.append("")
            lines.append(f"过去一个月模型偏差值: **{bc:+.2f}** (正值=模型预测偏低，负值=模型预测偏高)")
        if code in all_stability:
            lines.append("")
            lines.append(f"> ⚠ 预测稳定性告警: {all_stability[code]}")

        # 方向分歧时显示每次运行的详情
        if not consistent and all_runs and not classify_code(code):
            lines.append("")
            lines.append("| 运行 | 方向 | 3日涨跌 |")
            lines.append("|---|---|---|")
            for i, (rows, _) in enumerate(all_runs):
                d = directions[i]
                label = {"bull": "看涨", "bear": "看跌", "neutral": "看平"}[d]
                chg = rows[-1]["cum_chg"] * 100
                lines.append(f"| #{i + 1} | {label} | {chg:+.2f}% |")

        lines.append("")
        bt_metrics = all_bt.get(code, {}).get("metrics") if code in all_bt else None
        conf_data = all_conf.get(code, {}).get("conf") if code in all_conf else None
        n_win = all_bt.get(code, {}).get("windows", 0)

        lines.append(format_table(info["rows"], metrics=bt_metrics, conf=conf_data))
        if bt_metrics and n_win:
            lines.append(f"> 回测窗口: {n_win}")
        lines.append("")
        # 方向一致时重点标注
        if consistent and not classify_code(code):
            lines.append(f"**3日涨跌: {final['cum_chg'] * 100:+.2f}% → 终点 {final['close']:.2f}**")
        else:
            lines.append(f"3日涨跌: **{final['cum_chg'] * 100:+.2f}%** → 终点 **{final['close']:.2f}**")
        lines.append("")

    # Stability alerts
    if all_stability:
        lines.append("---")
        lines.append("")
        lines.append("## 二、预测稳定性告警")
        lines.append("")
        lines.append("以下标的预测结果较上次发生显著变化，请关注：")
        lines.append("")
        for code, msg in all_stability.items():
            name = all_forward.get(code, {}).get("name", code)
            lines.append(f"- **{name}** ({code}): {msg}")
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
    elif from_zxg:
        out_path = f"outputs/kronos_zxg_{today}.md"
    else:
        codes_str = "_".join(codes[:3])
        out_path = f"outputs/kronos_{codes_str}.md"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)

    print(f"\n报告已保存: {out_path}")


if __name__ == "__main__":
    main()

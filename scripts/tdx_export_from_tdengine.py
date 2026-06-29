#!/usr/bin/env python3
"""
从 TDengine 导出日线数据，计算后复权，生成 train/val/test .pkl。

TDengine kline 为原始未复权数据，adjust 表含分红/送转股/配股事件。
后复权公式：调整历史价格使当前价格为实际市场价，历史价格按累积分红比例上调。

用法:
    python scripts/tdx_export_from_tdengine.py
    python scripts/tdx_export_from_tdengine.py --output-dir data/tdx_import/1d --workers 4
"""

import os
import pickle
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm
from taosws import connect


def get_all_stocks(conn) -> dict[str, str]:
    """获取所有日线股票: {raw_code: full_symbol}。"""
    r = conn.query("select distinct tbname from tdx.kline where cycle='1d'")
    code_map = {}
    for row in r:
        tbname = row[0]  # k_000001_1d
        code = tbname.split('_')[1]
        if code.startswith(('60', '68', '5')):  # 上交所主板/科创/ETF(5xxx)
            code_map[code] = f'sh{code}'
        elif code.startswith(('00', '30', '12', '16', '15', '18', '39')):  # 深交所主板/创业/ETF/LOF/封基/指数(39)
            code_map[code] = f'sz{code}'
        elif code.startswith(('8', '4', '9')):  # 北交所
            code_map[code] = f'bj{code}'
        else:
            code_map[code] = code
    return code_map


def query_stock_with_adjust(conn, code: str) -> tuple[pd.DataFrame | None, list[dict] | None]:
    """查询单只股票的日线 + 分红事件。

    Returns:
        (df, events): df indexed by datetime, events sorted by date ascending
    """
    # OHLCV
    try:
        r = conn.query(
            f"select ts, open, high, low, close, volume, amount "
            f"from tdx.k_{code}_1d order by ts"
        )
    except Exception:
        return None, None

    rows = list(r)
    if len(rows) < 100:
        return None, None

    df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'amt'])
    df['ts'] = pd.to_datetime(df['ts']).dt.tz_localize(None)  # 去时区
    df = df.set_index('ts').sort_index()
    # 保 float64 精度，后复权计算后转 float32
    df = df.astype({c: np.float64 for c in ['open', 'high', 'low', 'close', 'vol', 'amt']})

    # 分红事件
    events = []
    try:
        r2 = conn.query(
            f"select ts, fenhong, peigujia, songzhuangu, peigu "
            f"from tdx.a_{code} order by ts"
        )
        for row in r2:
            ts, fh, pj, sz, pg = row
            fh, pj, sz, pg = float(fh), float(pj), float(sz), float(pg)
            if fh > 0 or sz > 0 or pg > 0:  # 只保留有实际分红的事件
                events.append({
                    'date': pd.Timestamp(ts).tz_localize(None),
                    'fenhong': fh,        # 元/10股
                    'peigujia': pj,       # 元/股
                    'songzhuangu': sz,    # 股/10股
                    'peigu': pg,          # 股/10股
                })
    except Exception:
        pass  # 无分红数据，不需要复权

    return df, events


def compute_back_adjust_factor(df: pd.DataFrame, events: list[dict]) -> np.ndarray:
    """计算后复权因子。

    因子从最新日向历史日累积：
      factor[今天的bar] = 1.0
      遇到分红事件时: factor[事件之前的bar] *= 乘数

    乘数公式:
      D = fenhong/10, S = songzhuangu/10, P = peigu/10, Pp = peigujia
      multiplier = C * (1+S+P) / (C - D + P*Pp)

    Returns:
        np.ndarray of shape [n_bars], dtype float64
    """
    n = len(df)
    factor = np.ones(n, dtype=np.float64)

    if not events:
        return factor

    # 按日期升序排列事件
    events_sorted = sorted(events, key=lambda e: e['date'])
    df_dates = df.index.values  # numpy array of datetime64
    raw_close = df['close'].values

    for evt in events_sorted:
        evt_date = np.datetime64(evt['date'])
        # 找到事件日期在 K 线中的位置 (第一个 >= 事件日期的 bar)
        event_idx = int(np.searchsorted(df_dates, evt_date))
        if event_idx >= n:
            continue
        # C_before = 事件日前一个交易日的原始收盘价
        prev_idx = event_idx - 1
        if prev_idx < 0:
            continue
        C_before = raw_close[prev_idx]
        if C_before <= 0:
            continue

        D = evt['fenhong'] / 10.0        # 元/股
        S = evt['songzhuangu'] / 10.0
        P = evt['peigu'] / 10.0
        Pp = evt['peigujia']

        denominator = C_before - D + P * Pp
        if denominator <= 0:
            continue
        numerator = C_before * (1.0 + S + P)
        multiplier = numerator / denominator

        if abs(multiplier - 1.0) < 1e-12:
            continue

        # 事件日前所有 bar 乘以该乘数 (idx < event_idx)
        factor[:event_idx] *= multiplier

    return factor


def apply_adjustment(df: pd.DataFrame, factor: np.ndarray) -> pd.DataFrame:
    """应用后复权因子到 OHLC。"""
    df_adj = df.copy()
    for col in ['open', 'high', 'low', 'close']:
        df_adj[col] = (df[col].values * factor).astype(np.float32)
    # vol/amt 不变
    df_adj['vol'] = df_adj['vol'].astype(np.float32)
    df_adj['amt'] = df_adj['amt'].astype(np.float32)
    return df_adj


def split_by_date(df: pd.DataFrame, train_end: str, val_end: str):
    """按日期切分。"""
    te = pd.Timestamp(train_end)
    ve = pd.Timestamp(val_end)
    train_df = df[df.index <= te].copy() if (df.index <= te).any() else None
    val_df = df[(df.index > te) & (df.index <= ve)].copy() if ((df.index > te) & (df.index <= ve)).any() else None
    test_df = df[df.index > ve].copy() if (df.index > ve).any() else None
    return train_df, val_df, test_df


def process_stock(args: tuple) -> tuple[str, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, str | None]:
    """处理单只股票（供线程池调用）。每线程独立连接。"""
    code, full_symbol, train_end, val_end = args
    conn = connect()
    try:
        df, events = query_stock_with_adjust(conn, code)
        if df is None:
            return full_symbol, None, None, None, "no data or <100 bars"

        factor = compute_back_adjust_factor(df, events)
        df_adj = apply_adjustment(df, factor)
        train_df, val_df, test_df = split_by_date(df_adj, train_end, val_end)

        return full_symbol, train_df, val_df, test_df, None
    except Exception as e:
        return full_symbol, None, None, None, str(e)
    finally:
        conn.close()


def export_tdengine(output_dir: str, workers: int = 4):
    """主函数：TDengine → .pkl。"""
    os.makedirs(output_dir, exist_ok=True)

    conn = connect()

    print("获取股票列表...")
    code_map = get_all_stocks(conn)
    print(f"共 {len(code_map)} 只日线股票")

    train_end = '2025-12-25'
    val_end = '2026-03-25'

    train_data, val_data, test_data = {}, {}, {}
    errors = []

    print(f"导出 + 后复权计算 ({workers} 线程)...")

    tasks = [(c, code_map[c], train_end, val_end) for c in code_map]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_stock, t) for t in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="导出"):
            sym, tdf, vdf, tedf, err = future.result()
            if err:
                errors.append((sym, err))
                continue
            if tdf is not None and len(tdf) >= 10:
                train_data[sym] = tdf
            if vdf is not None and len(vdf) >= 1:
                val_data[sym] = vdf
            if tedf is not None and len(tedf) >= 1:
                test_data[sym] = tedf

    # 保存
    for name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        path = os.path.join(output_dir, f'{name}_data.pkl')
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"{name}_data.pkl: {len(data)} stocks, {size_mb:.1f} MB")

    if errors:
        print(f"\n错误 ({len(errors)} 只):")
        for s, e in errors[:10]:
            print(f"  {s}: {e}")
        if len(errors) > 10:
            print(f"  ... 还有 {len(errors) - 10} 个错误")

    print("\n完成!")


def main():
    parser = argparse.ArgumentParser(description='从 TDengine 导出后复权日线数据')
    parser.add_argument('--output-dir', default='./data/tdx_import/1d',
                        help='输出目录')
    parser.add_argument('--workers', '-n', type=int, default=4,
                        help='线程数 (默认: 4)')
    args = parser.parse_args()
    export_tdengine(args.output_dir, args.workers)


if __name__ == '__main__':
    main()

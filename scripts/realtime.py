#!/usr/bin/env python3
"""
盘中实时行情补充模块。

在交易时段内，通过 tdxdata 网络接口获取当日实时快照，
构建为一根"当日 K 线"拼接到历史 DataFrame 末尾。

用法:
    from scripts.realtime import append_realtime_bars
    append_realtime_bars(codes, data_map, factor_map)
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# A 股交易时段（含余量）
_TRADING_START = (9, 25)    # 09:25
_TRADING_END = (15, 5)      # 15:05


def is_trading_hours() -> bool:
    """判断当前时间是否可能在 A 股交易时段（含午休）。

    工作日 9:25 ~ 15:05 之间返回 True。
    不做节假日判断——节假日无成交数据，快照中 volume=0 会自然跳过。
    """
    now = datetime.now()
    # 周一=0 ... 周日=6
    if now.weekday() >= 5:
        return False
    t = (now.hour, now.minute)
    return _TRADING_START <= t <= _TRADING_END


def _strip_prefix(code):
    """将 sh600000 / sz002741 转为纯数字代码。"""
    if code.startswith("sh") or code.startswith("sz"):
        return code[2:]
    return code


def _is_index(code):
    """判断是否为指数代码。

    通达信指数代码规则：
      - 上证指数：sh999xxx（如 sh999999=上证大盘，sh999998=上证50）
      - 深证指数：sz399xxx（如 sz399001=深证成指，sz399006=创业板指）
      - 注意：sh000001 是平安银行（深圳市场代码 000001 在上海无对应指数）
    """
    return code.startswith("sh999") or code.startswith("sz399")


def append_realtime_bars(codes, data_map, factor_map):
    """为每只股票追加当日实时 K 线（仅盘中）。

    通过 tdxdata 网络接口获取实时快照，构建当日 K 线，
    拼接到 data_map 中对应 DataFrame 末尾。

    Args:
        codes: 股票代码列表 ['sh600000', 'sz002741', ...]
        data_map: {code: DataFrame} — 现有历史数据（会被原地修改）
        factor_map: {code: float} — 复权因子

    Returns:
        int — 追加了实时行情的股票数量
    """
    if not is_trading_hours():
        return 0

    # 过滤出有数据的股票
    valid_codes = [c for c in codes if c in data_map and data_map[c] is not None]
    if not valid_codes:
        return 0

    # 批量获取实时行情（使用 tdxdata 高层接口，字段名标准化）
    try:
        from tdxdata import TdxData
        pure_codes = [_strip_prefix(c) for c in valid_codes]
        # 构建 sh/sz 前缀 → 纯代码 映射
        pure_to_orig = {}
        for code in valid_codes:
            pure = _strip_prefix(code)
            pure_to_orig[pure] = code
    except ImportError:
        print("  tdxdata 未安装，跳过实时行情", file=sys.stderr)
        return 0

    try:
        with TdxData() as api:
            quotes_df = api.fetch_realtime(stock_list=pure_codes)
    except Exception as e:
        print(f"  实时行情获取失败: {e}", file=sys.stderr)
        return 0

    if quotes_df is None or quotes_df.empty:
        return 0

    today = pd.Timestamp(datetime.now().date())
    appended = 0

    for _, row in quotes_df.iterrows():
        # tdxdata 返回 stock_code（纯数字）和标准字段名
        pure_code = str(row['stock_code'])
        orig_code = pure_to_orig.get(pure_code)
        if orig_code is None or orig_code not in data_map:
            continue

        # 检查是否有成交（停牌/节假日 volume=0）
        volume = row.get('volume', 0)
        if pd.isna(volume) or volume <= 0:
            continue

        # tdxdata 字段名已标准化：open/high/low/close/volume/amount
        close_price = float(row.get('close', 0))
        open_price = float(row.get('open', 0))
        high_price = float(row.get('high', 0))
        low_price = float(row.get('low', 0))
        vol = float(row.get('volume', 0))
        amt = float(row.get('amount', 0))

        # 价格有效性检查
        if close_price <= 0 or open_price <= 0:
            continue

        # 个股：转为后复权价格（指数不需要复权）
        factor = factor_map.get(orig_code, 1.0)
        is_idx = _is_index(orig_code)
        if not is_idx and factor > 0 and factor != 1.0:
            open_price *= factor
            high_price *= factor
            low_price *= factor
            close_price *= factor

        # 构建当日 K 线（Kronos 6 字段格式）
        df = data_map[orig_code]
        bar = pd.DataFrame({
            'open': [open_price],
            'high': [high_price],
            'low': [low_price],
            'close': [close_price],
            'vol': [vol],
            'amt': [amt],
        }, index=pd.DatetimeIndex([today]))

        # 如果当天已存在（收盘后重新运行），更新最后一行
        if len(df) > 0 and df.index[-1].date() == today.date():
            df.iloc[-1] = bar.iloc[0]
        else:
            data_map[orig_code] = pd.concat([df, bar])
        appended += 1

    if appended > 0:
        print(f"  已追加 {appended}/{len(valid_codes)} 只股票的当日实时行情")
    return appended

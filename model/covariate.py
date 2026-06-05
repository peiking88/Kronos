"""
Kronos + CZSC 协变量注入模块。

包含:
    - InputInjectionBlock: IIB 残差注入块，借鉴 ChronosX 设计
    - CZSCFeatureExtractor: 从 K 线数据提取 CZSC 7 维特征
    - 辅助函数: 7 个维度特征提取 + 中枢计算 + 笔映射
"""

import numpy as np
import torch
import torch.nn as nn

from czsc._native import CZSC, RawBar, Freq, Mark, Direction, ZS


# ======================================================================
# IIB 模块
# ======================================================================

class InputInjectionBlock(nn.Module):
    """Input Injection Block — 残差注入协变量信息到 token embeddings。

    借鉴 ChronosX 的 InputInjectionBlock 设计，适配 Kronos 的 d_model=832。
    当 covariates=None 时，forward 返回零张量，对主模型无任何影响。

    参数量 ~560K，占 Kronos 总参数量 (102.3M) 的 0.55%。
    """

    def __init__(self, d_model=832, cov_dim=7, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.cov_dim = cov_dim
        self.hidden_dim = hidden_dim

        self.emb_proj = nn.Linear(d_model, hidden_dim, bias=True)
        self.cov_proj = nn.Linear(cov_dim, hidden_dim, bias=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model, bias=True),
        )

    def forward(self, token_embeddings, covariates=None):
        """
        Args:
            token_embeddings: [B, T, d_model]
            covariates:       [B, T, cov_dim] or None
        Returns:
            残差修正量 [B, T, d_model]，若 covariates=None 则返回零张量
        """
        if covariates is None:
            return torch.zeros_like(token_embeddings)
        emb_h = self.emb_proj(token_embeddings)   # [B, T, hidden]
        cov_h = self.cov_proj(covariates)          # [B, T, hidden]
        combined = torch.cat([emb_h, cov_h], dim=-1)  # [B, T, hidden*2]
        return self.ffn(combined)                   # [B, T, d_model]


# ======================================================================
# CZSC 特征提取辅助函数
# ======================================================================

def build_raw_bars(df, symbol="unknown", freq=Freq.D):
    """将 DataFrame 转换为 czsc RawBar 列表。

    DataFrame 必须包含列: open, high, low, close, vol, amt
    且 index 为 DatetimeIndex。

    注意: RawBar 构造参数顺序为 (symbol, dt, freq, open, close, high, low, vol, amount, id)
    """
    bars = []
    for i, (dt, row) in enumerate(df.iterrows()):
        bar = RawBar(
            symbol=symbol,
            dt=dt,
            freq=freq,
            open=float(row['open']),
            close=float(row['close']),
            high=float(row['high']),
            low=float(row['low']),
            vol=float(row['vol']),
            amount=float(row['amt']),
            id=i,
        )
        bars.append(bar)
    return bars


def _build_bar_to_bi_map(bi_list, bar_dts):
    """构建 K 线到笔的映射。

    遍历已完成的笔列表，根据时间范围将 K 线映射到对应笔。

    Args:
        bi_list: 已完成笔列表 (c.bi_list)
        bar_dts: K 线时间戳列表 (pd.DatetimeIndex)

    Returns:
        dict[int, BI]: K 线索引 → 所属笔对象
    """
    bi_map = {}
    for bi in bi_list:
        for j, dt in enumerate(bar_dts):
            if j not in bi_map and bi.sdt <= dt <= bi.edt:
                bi_map[j] = bi
    return bi_map


def _find_fractal_for_bar(fx_list, bar_dt):
    """找到 K 线附近的分型。

    分型由 3 根相邻 K 线构成，中间那根为分型点。
    如果 K 线时间等于分型时间，返回该分型。
    """
    for fx in fx_list:
        if fx.dt == bar_dt:
            return fx
    return None


# ---- D1: 强分型 ----

def get_strong_fractal(fx_list, bar_dt):
    """D1: 强分型编码（含影线+量能增强）。

    基础编码 (power_str):
        弱=0, 中=1, 强=2

    影线因子 (0~0.3):
        顶分型看上影线占比，底分型看下影线占比

    量能因子 (0~0.2):
        分型中间K线量 / 前后K线平均量，超出1.0部分线性映射

    最终: sign(mark) * (base + shadow_score + volume_score)
    值域: [-2.5, +2.5]
    """
    fx = _find_fractal_for_bar(fx_list, bar_dt)
    if fx is None:
        return 0.0

    # 基础分型强度
    base = {'弱': 0, '中': 1, '强': 2}.get(fx.power_str, 0)

    # 影线因子 (0~0.3)
    mid = fx.elements[1]
    body_range = mid.high - mid.low
    shadow_score = 0.0
    if body_range > 0:
        if fx.mark == Mark.G:  # 顶分型看上影线
            shadow = (mid.high - max(mid.close, mid.open)) / body_range
        else:                   # 底分型看下影线
            shadow = (min(mid.close, mid.open) - mid.low) / body_range
        shadow_score = min(shadow, 1.0) * 0.3

    # 量能因子 (0~0.2)
    prev, nxt = fx.elements[0], fx.elements[2]
    avg_vol = (prev.vol + nxt.vol) / 2
    volume_score = 0.0
    if avg_vol > 0:
        vol_ratio = mid.vol / avg_vol
        volume_score = min(max(vol_ratio - 1.0, 0.0) / 4.0, 1.0) * 0.2

    # 复合编码
    raw = base + shadow_score + volume_score
    if fx.mark == Mark.G:
        return raw
    elif fx.mark == Mark.D:
        return -raw
    return 0.0


# ---- D2: 笔方向 ----

def get_bi_direction(bi):
    """D2: 笔方向编码。

    返回: +1 向上, -1 向下, 0 未归属
    """
    if bi is None:
        return 0
    if bi.direction == Direction.Up:
        return 1
    elif bi.direction == Direction.Down:
        return -1
    return 0


# ---- D3: 中枢相对位置 ----

def compute_zhongshu_list(bi_list):
    """从笔列表计算中枢，使用 czsc 原生 ZS 类。

    连续 3 笔的价格重叠区间构成中枢。
    """
    zs_list = []
    for i in range(2, len(bi_list)):
        three_bis = [bi_list[i - 2], bi_list[i - 1], bi_list[i]]
        zs = ZS(three_bis)
        if zs.is_valid():
            zs_list.append(zs)
    return zs_list


def get_zhongshu_position(close, zs_list, bar_dt):
    """D3: 中枢相对位置。

    归一化到 [0, 1] 区间:
        < 0  在中枢下方（超卖或向下突破）
         0   在中枢下沿（强支撑位）
        0.5  在中枢正中间（典型震荡）
         1   在中枢上沿（强阻力位）
        > 1  在中枢上方（超买或向上突破）
    """
    for zs in reversed(zs_list):
        if zs.sdt <= bar_dt <= zs.edt:
            span = zs.zg - zs.zd
            if span > 0:
                pos = (close - zs.zd) / span
                return max(-2.0, min(3.0, pos))
    return 0.5  # 不在中枢内，返回中性值


# ---- D4: 笔力度 ----

def get_bi_power(bi, mean_power, std_power):
    """D4: 笔力度 z-score 归一化。"""
    if bi is None:
        return 0.0
    return (bi.power - mean_power) / std_power


# ---- D5: 背驰信号 ----

def get_beichi(bi_list, current_bi):
    """D5: 背驰信号（买卖点）。

    返回值:
        正值 → 底背驰（向下笔力度减弱）→ 买方信号
        负值 → 顶背驰（向上笔力度减弱）→ 卖方信号
    """
    if current_bi is None:
        return 0.0
    # 找同方向的前一笔
    prev = None
    for b in reversed(bi_list):
        if b.direction == current_bi.direction and b is not current_bi:
            prev = b
            break
    if prev is None:
        return 0.0
    diff = (prev.power - current_bi.power) / (prev.power + 1e-8)
    # 向下笔力度减弱 = 底背驰 = 正值（买）
    # 向上笔力度减弱 = 顶背驰 = 负值（卖）
    if current_bi.direction == Direction.Down:
        return diff
    else:
        return -diff


# ---- D6: 笔拟合度 ----

def get_bi_rsq(bi):
    """D6: 笔拟合度 R²。"""
    if bi is None:
        return 0.0
    return bi.rsq  # 原值，范围 [0, 1]


# ---- D7: 嵌套笔数 ----

def get_fake_bi_count(bi, mean_fk, std_fk):
    """D7: 嵌套笔数 z-score 归一化。"""
    if bi is None:
        return 0.0
    return (len(bi.fake_bis) - mean_fk) / std_fk


# ======================================================================
# CZSC 特征提取器
# ======================================================================

class CZSCFeatureExtractor:
    """从 K 线数据提取 CZSC 7 维特征。

    7 个维度:
        D1: 强分型 (离散, [-2, +2])
        D2: 笔方向 (离散, {-1, 0, +1})
        D3: 中枢相对位置 (连续, 约 [-2, 3])
        D4: 笔力度 z-score (连续)
        D5: 背驰信号 (连续, 约 [-0.5, +0.5])
        D6: 笔拟合度 R² (连续, [0, 1])
        D7: 嵌套笔数 z-score (连续)
    """

    def extract(self, df, symbol="unknown"):
        """从 DataFrame 提取 7 维特征。

        Args:
            df: DataFrame, 必须包含 open/high/low/close/vol/amt 列，
                index 为 DatetimeIndex
            symbol: 股票代码

        Returns:
            np.ndarray, shape [T, 7], dtype float32
        """
        n = len(df)
        if n < 10:
            # K 线太少无法进行缠论分析
            return np.zeros((n, 7), dtype=np.float32)

        # 构建 RawBar 列表并创建 CZSC 对象
        bars = build_raw_bars(df, symbol=symbol)
        c = CZSC(bars)

        bar_dts = df.index
        closes = df['close'].values

        # 预计算中枢列表
        zs_list = compute_zhongshu_list(c.bi_list)

        # 预计算笔统计量 (D4, D7 归一化用)
        if len(c.bi_list) > 0:
            all_powers = [b.power for b in c.bi_list]
            mean_power = np.mean(all_powers)
            std_power = np.std(all_powers) + 1e-8

            all_fakes = [len(b.fake_bis) for b in c.bi_list]
            mean_fk = np.mean(all_fakes)
            std_fk = np.std(all_fakes) + 1e-8
        else:
            mean_power, std_power = 0.0, 1.0
            mean_fk, std_fk = 0.0, 1.0

        # 构建 K 线 → 笔映射
        bi_map = _build_bar_to_bi_map(c.bi_list, bar_dts)

        # 遍历每根 K 线提取特征
        features = np.zeros((n, 7), dtype=np.float32)
        for i in range(n):
            dt = bar_dts[i]
            bi = bi_map.get(i)

            features[i, 0] = get_strong_fractal(c.fx_list, dt)       # D1
            features[i, 1] = get_bi_direction(bi)                     # D2
            features[i, 2] = get_zhongshu_position(closes[i], zs_list, dt)  # D3
            features[i, 3] = get_bi_power(bi, mean_power, std_power)  # D4
            features[i, 4] = get_beichi(c.bi_list, bi)                # D5
            features[i, 5] = get_bi_rsq(bi)                           # D6
            features[i, 6] = get_fake_bi_count(bi, mean_fk, std_fk)   # D7

        return features

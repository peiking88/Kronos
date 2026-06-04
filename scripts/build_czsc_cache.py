#!/usr/bin/env python3
"""
预计算 CZSC 7 维特征缓存。

为 data/tdx_import/1d/ 下的 train_data.pkl / val_data.pkl / test_data.pkl
中的每只股票、每根 K 线计算 CZSC 特征，保存为对应的 czsc_features_*.pkl。

缓存格式:
    {symbol: np.ndarray, shape [N, 7], dtype=float32}

    7 个维度: [D1强分型, D2笔方向, D3中枢位置, D4笔力度, D5背驰, D6拟合度, D7嵌套笔数]

用法:
    python scripts/build_czsc_cache.py --data-dir data/tdx_import/1d
    python scripts/build_czsc_cache.py --data-dir data/tdx_import/1d -n 4
    python scripts/build_czsc_cache.py --data-dir data/tdx_import/1d --split train
"""

import os
import sys
import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

# 项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.covariate import CZSCFeatureExtractor


def process_symbol(args):
    """处理单只股票，提取 CZSC 特征。"""
    symbol, df = args
    try:
        extractor = CZSCFeatureExtractor()
        features = extractor.extract(df, symbol=symbol)
        return symbol, features, None
    except Exception as e:
        return symbol, None, str(e)


def build_cache(data_path, output_dir, n_workers=1):
    """为单个数据文件构建 CZSC 特征缓存。"""
    split_name = os.path.basename(data_path).replace('_data.pkl', '')
    output_path = os.path.join(output_dir, f"czsc_features_{split_name}.pkl")

    # 断点续传：加载已有缓存
    existing_cache = {}
    if os.path.exists(output_path):
        with open(output_path, 'rb') as f:
            existing_cache = pickle.load(f)
        print(f"已加载 {len(existing_cache)} 只股票的缓存 ({output_path})")

    # 加载数据
    print(f"加载数据: {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    print(f"共 {len(data)} 只股票")

    # 过滤已处理的
    to_process = {k: v for k, v in data.items() if k not in existing_cache}
    print(f"待处理: {len(to_process)}, 已缓存: {len(existing_cache)}")

    if not to_process:
        print("所有股票已缓存，跳过")
        return output_path

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 提取特征
    results = {}
    errors = []
    items = list(to_process.items())

    if n_workers <= 1:
        # 单进程模式
        extractor = CZSCFeatureExtractor()
        for symbol, df in tqdm(items, desc=f"提取 {split_name}"):
            try:
                features = extractor.extract(df, symbol=symbol)
                results[symbol] = features
            except Exception as e:
                errors.append((symbol, str(e)))
    else:
        # 多进程模式
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(process_symbol, (s, df)): s
                for s, df in items
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"提取 {split_name}"):
                symbol, features, error = future.result()
                if features is not None:
                    results[symbol] = features
                else:
                    errors.append((symbol, error))

    # 合并缓存
    cache = {**existing_cache, **results}

    # 保存
    with open(output_path, 'wb') as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"已保存 {len(cache)} 只股票的特征到 {output_path}")

    # 错误报告
    if errors:
        print(f"\n错误 ({len(errors)} 只):")
        for symbol, error in errors[:10]:
            print(f"  {symbol}: {error}")
        if len(errors) > 10:
            print(f"  ... 还有 {len(errors) - 10} 个错误")

    # 分布统计
    print_feature_stats(cache)
    return output_path


def print_feature_stats(cache):
    """打印特征分布统计。"""
    all_features = np.concatenate(list(cache.values()), axis=0)
    n_bars, n_dim = all_features.shape
    dim_names = ['D1强分型', 'D2笔方向', 'D3中枢位置', 'D4笔力度', 'D5背驰', 'D6拟合度', 'D7嵌套笔数']

    print(f"\n特征分布统计 (共 {n_bars} 根 K 线, {len(cache)} 只股票):")
    print(f"{'维度':<10} {'均值':>8} {'标准差':>8} {'NaN%':>6} {'Inf%':>6} {'范围':>20}")
    print("-" * 65)
    for i, name in enumerate(dim_names):
        col = all_features[:, i]
        nan_pct = np.isnan(col).mean() * 100
        inf_pct = np.isinf(col).mean() * 100
        valid = col[~np.isnan(col) & ~np.isinf(col)]
        mean = np.mean(valid) if len(valid) > 0 else 0
        std = np.std(valid) if len(valid) > 0 else 0
        vmin = np.min(valid) if len(valid) > 0 else 0
        vmax = np.max(valid) if len(valid) > 0 else 0
        print(f"{name:<10} {mean:>8.3f} {std:>8.3f} {nan_pct:>5.1f}% {inf_pct:>5.1f}% [{vmin:.2f}, {vmax:.2f}]")


def main():
    parser = argparse.ArgumentParser(description='预计算 CZSC 7 维特征缓存')
    parser.add_argument('--data-dir', default='./data/tdx_import/1d',
                        help='数据目录，包含 train_data.pkl / val_data.pkl / test_data.pkl')
    parser.add_argument('--output-dir', default=None,
                        help='输出目录 (默认: <data-dir>/czsc_features/)')
    parser.add_argument('--split', default=None, choices=['train', 'val', 'test'],
                        help='只处理指定分片 (默认: 全部)')
    parser.add_argument('-n', '--workers', type=int, default=1,
                        help='并行进程数 (默认: 1)')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.data_dir, 'czsc_features')

    splits = ['train', 'val', 'test'] if args.split is None else [args.split]

    for split in splits:
        data_path = os.path.join(args.data_dir, f"{split}_data.pkl")
        if not os.path.exists(data_path):
            print(f"跳过 {split}: {data_path} 不存在")
            continue
        print(f"\n{'='*60}")
        print(f"处理: {split}")
        print(f"{'='*60}")
        build_cache(data_path, output_dir, n_workers=args.workers)

    print("\n完成!")


if __name__ == '__main__':
    main()

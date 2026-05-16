#!/usr/bin/env python3
"""
枚举 TDX 本地日线目录中的全部股票代码，写入到文本文件，
作为 ``tdx_import.py`` 的 ``--symbol-file`` 输入。

过滤规则：

- 接受 ``s[2:]`` 以 ``6/0/3/4/8/9`` 开头的代码
- 默认保留 ``00`` 开头代码（包含深市主板 sz000xxx 和沪市指数 sh000xxx）
- 排除 ``bj`` 前缀（北交所）

注意：

* 可通过 ``--exclude-00`` 排除 ``00`` 开头代码（排除沪市指数 sh000xxx，
  但同时也会排除深市主板 sz000xxx）
* 沪市 B 股 ``9xxxxx``、深市 B 股 ``2xxxxx`` 不在排除规则里，
  会被一并写入；如不需要可后处理过滤。

用法
----
.. code-block:: bash

    .venv/bin/python scripts/discover_stocks.py --output /tmp/tdx_all_stocks.txt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.tdx_import import TdxDataImporter


def filter_stocks(stocks, exclude_00=False, exclude_bj=True):
    out = []
    for s in stocks:
        tail = s[2:]
        if not tail.startswith(("6", "0", "3", "4", "8", "9")):
            continue
        if exclude_00 and tail.startswith("00"):
            continue
        if exclude_bj and s.startswith("bj"):
            continue
        out.append(s)
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description="筛选 TDX 本地日线目录中的可训练股票代码",
    )
    p.add_argument(
        "--output",
        default="/tmp/tdx_all_stocks.txt",
        help="输出文件路径（每行一个代码），默认 /tmp/tdx_all_stocks.txt",
    )
    p.add_argument(
        "--tdxdir",
        default=None,
        help="TDX 数据根目录，默认走 TdxDataImporter 内置探测",
    )
    p.add_argument(
        "--exclude-00",
        action="store_true",
        help="排除 00 开头代码（排除沪市指数 sh000xxx，但同时也会排除深市主板 sz000xxx）",
    )
    p.add_argument(
        "--no-exclude-bj",
        action="store_true",
        help="不排除北交所（bj 前缀）",
    )
    return p.parse_args()


def main():
    args = parse_args()
    importer = TdxDataImporter(tdxdir=args.tdxdir)
    all_stocks = importer.discover_stocks()
    real = filter_stocks(
        all_stocks,
        exclude_00=args.exclude_00,
        exclude_bj=not args.no_exclude_bj,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for s in real:
            f.write(s + "\n")

    print(f"Discovered: {len(all_stocks)}")
    print(f"After filter: {len(real)}")
    print(f"Written to: {args.output}")


if __name__ == "__main__":
    main()

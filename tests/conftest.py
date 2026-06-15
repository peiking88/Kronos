"""
测试环境 import 顺序隔离（conftest，pytest 启动时最先加载）。

背景
----
mootdx 对 import 顺序高度敏感（见 scripts/predict_stocks.py 顶部注释：
"mootdx.quotes 必须在 tdxdata 之前导入"）。此外 czsc 在 import 时会向
sys.path 注入脏路径（../opentdx），把 opentdx 解析为 PEP 420 namespace
package（无 __init__.py），丢失 EX_MARKET 等属性。

故障现象
--------
pytest 按字母序 collection：先 test_covariate（import czsc → 污染 sys.path），
后 test_prefetch_factors（import scripts.predict_stocks → mootdx →
`from opentdx import EX_MARKET`）。此刻 opentdx 已是被污染的 namespace，
于是 ImportError: cannot import name 'EX_MARKET' from 'opentdx'。

生产环境（python scripts/predict_stocks.py）不受影响：predict_stocks.py
顶部刻意将 mootdx 的 import 排在 covariate（czsc）之前，opentdx 在干净
环境下先行缓存为真包。

修复
----
在此（pytest 启动、任何 collection 之前）预导入 mootdx.quotes，使 opentdx
等以真包形式缓存进 sys.modules。之后无论 czsc 如何污染 sys.path，
`from opentdx import EX_MARKET` 都从缓存命中真包，规避顺序污染。
"""

import mootdx.quotes  # noqa: F401  预导入，副作用：缓存 opentdx 真包

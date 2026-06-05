"""
IIB 模块和 CZSC 特征提取器单元测试。

测试覆盖:
    - InputInjectionBlock: 输出形状、None 时零输出、参数量、梯度流通
    - KronosWithIIB: 无协变量时一致性、有协变量时形状正确
    - CZSCFeatureExtractor: 提取形状、无 NaN/Inf、值域范围
    - CovFill: 协变量填充模式（last / decay / zero）
"""

import pytest
import numpy as np
import pandas as pd
import torch

from model.covariate import InputInjectionBlock, CZSCFeatureExtractor


# ======================================================================
# TestInputInjectionBlock
# ======================================================================

class TestInputInjectionBlock:
    """IIB 模块单元测试。"""

    def test_output_shape_with_covariates(self):
        """有协变量时输出形状 = 输入形状。"""
        iib = InputInjectionBlock(d_model=832, cov_dim=7, hidden_dim=256)
        x = torch.randn(2, 10, 832)
        cov = torch.randn(2, 10, 7)
        out = iib(x, cov)
        assert out.shape == (2, 10, 832)

    def test_none_covariates_returns_zeros(self):
        """covariates=None 时输出零张量。"""
        iib = InputInjectionBlock(d_model=832, cov_dim=7, hidden_dim=256)
        x = torch.randn(2, 10, 832)
        out = iib(x, None)
        assert torch.all(out == 0).item()

    def test_parameter_count(self):
        """参数量约 560K。"""
        iib = InputInjectionBlock(d_model=832, cov_dim=7, hidden_dim=256)
        n = sum(p.numel() for p in iib.parameters())
        # emb_proj: 832*256+256=213248
        # cov_proj: 7*256+256=2048
        # ffn[0]: 512*256+256=131328
        # ffn[3]: 256*832+832=213664
        # total: 560288
        assert 550_000 < n < 570_000, f"Expected ~560K, got {n}"

    def test_gradient_flows_through(self):
        """梯度能正常回传。"""
        iib = InputInjectionBlock(d_model=832, cov_dim=7, hidden_dim=256)
        x = torch.randn(2, 10, 832, requires_grad=True)
        cov = torch.randn(2, 10, 7, requires_grad=True)
        out = iib(x, cov)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert cov.grad is not None

    def test_batch_independence(self):
        """不同 batch 的输出相互独立（eval 模式下无 dropout 干扰）。"""
        iib = InputInjectionBlock(d_model=832, cov_dim=7, hidden_dim=256)
        iib.eval()  # eval 模式关闭 dropout
        x = torch.randn(4, 10, 832)
        cov = torch.randn(4, 10, 7)
        with torch.no_grad():
            out_full = iib(x, cov)
            out_single = iib(x[:1], cov[:1])
        # eval 模式下单独计算和批量计算的第一行应一致
        assert torch.allclose(out_full[0], out_single[0], atol=1e-5)

    def test_different_hidden_dims(self):
        """支持不同隐藏维度。"""
        iib = InputInjectionBlock(d_model=256, cov_dim=7, hidden_dim=64)
        x = torch.randn(1, 5, 256)
        cov = torch.randn(1, 5, 7)
        out = iib(x, cov)
        assert out.shape == (1, 5, 256)


# ======================================================================
# TestKronosWithIIB
# ======================================================================

class TestKronosWithIIB:
    """Kronos 模型集成 IIB 测试。"""

    @pytest.fixture
    def small_model(self):
        """创建小型 Kronos 模型（减少层数和维度以加速测试）。"""
        from model.kronos import Kronos
        model = Kronos(
            s1_bits=9, s2_bits=9, n_layers=2, d_model=256,
            n_heads=4, ff_dim=512,
            ffn_dropout_p=0.1, attn_dropout_p=0.1,
            resid_dropout_p=0.1, token_dropout_p=0.1, learn_te=False
        )
        model.eval()
        return model

    def test_forward_without_covariates_shape(self, small_model):
        """无协变量时 forward 输出形状正确。"""
        s1 = torch.randint(0, 512, (2, 10))
        s2 = torch.randint(0, 512, (2, 10))
        with torch.no_grad():
            s1_logits, s2_logits = small_model(s1, s2)
        assert s1_logits.shape == (2, 10, 512)
        assert s2_logits.shape == (2, 10, 512)

    def test_forward_with_covariates_shape(self, small_model):
        """有协变量时输出形状不变。"""
        s1 = torch.randint(0, 512, (2, 10))
        s2 = torch.randint(0, 512, (2, 10))
        cov = torch.randn(2, 10, 7)
        with torch.no_grad():
            s1_logits, s2_logits = small_model(s1, s2, past_covariates=cov)
        assert s1_logits.shape == (2, 10, 512)
        assert s2_logits.shape == (2, 10, 512)

    def test_decode_s1_with_covariates(self, small_model):
        """decode_s1 支持协变量。"""
        s1 = torch.randint(0, 512, (1, 10))
        s2 = torch.randint(0, 512, (1, 10))
        cov = torch.randn(1, 10, 7)
        with torch.no_grad():
            s1_logits, context = small_model.decode_s1(s1, s2, past_covariates=cov)
        assert s1_logits.shape[0] == 1
        assert context.shape == (1, 10, 256)

    def test_decode_s2_unaffected(self, small_model):
        """decode_s2 不接受协变量，正常工作。"""
        s1 = torch.randint(0, 512, (1, 10))
        s2 = torch.randint(0, 512, (1, 10))
        cov = torch.randn(1, 10, 7)
        with torch.no_grad():
            s1_logits, context = small_model.decode_s1(s1, s2, past_covariates=cov)
            s2_logits = small_model.decode_s2(context, s1[:, -1:])
        assert s2_logits.shape[0] == 1

    def test_iib_parameters_exist(self, small_model):
        """模型中包含 IIB 参数。"""
        iib_params = [n for n, p in small_model.named_parameters() if 'iib' in n]
        assert len(iib_params) > 0, "模型中未找到 IIB 参数"

    def test_iib_parameter_count(self, small_model):
        """IIB 参数量在预期范围。"""
        iib_params = sum(p.numel() for n, p in small_model.named_parameters() if 'iib' in n)
        total_params = sum(p.numel() for p in small_model.parameters())
        # 小模型 d_model=256 时 IIB 约 265K / 总 2.5M ≈ 10.5%
        # 真实 d_model=832 时 IIB 约 560K / 总 102M ≈ 0.55%
        assert iib_params > 0
        assert iib_params < total_params * 0.2  # 小模型中 IIB 占比略高

    def test_covariates_change_output(self, small_model):
        """有协变量和无协变量的输出应不同。"""
        s1 = torch.randint(0, 512, (1, 10))
        s2 = torch.randint(0, 512, (1, 10))
        cov = torch.randn(1, 10, 7)
        with torch.no_grad():
            out_no_cov = small_model(s1, s2)
            out_with_cov = small_model(s1, s2, past_covariates=cov)
        # 输出应该不同（协变量有实际影响）
        assert not torch.allclose(out_no_cov[0], out_with_cov[0])


# ======================================================================
# TestCZSCFeatureExtractor
# ======================================================================

def _make_test_df(n=120, seed=42):
    """生成测试用 K 线 DataFrame。"""
    np.random.seed(seed)
    dates = pd.date_range('2025-01-01', periods=n, freq='B')
    price = 10 + np.cumsum(np.random.randn(n) * 0.2)

    df = pd.DataFrame({
        'open': price + np.random.rand(n) * 0.1,
        'high': price + np.abs(np.random.randn(n)) * 0.3,
        'low': price - np.abs(np.random.randn(n)) * 0.3,
        'close': price + np.random.randn(n) * 0.1,
        'vol': np.random.rand(n) * 1000 + 100,
        'amt': np.random.rand(n) * 10000 + 1000,
    }, index=dates)

    # 确保 OHLC 关系正确
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
    return df


class TestCZSCFeatureExtractor:
    """CZSC 特征提取器测试。"""

    def test_extract_shape(self):
        """提取特征形状正确 [T, 7]。"""
        extractor = CZSCFeatureExtractor()
        df = _make_test_df(n=120)
        features = extractor.extract(df, symbol='test')
        assert features.shape == (120, 7)

    def test_extract_dtype(self):
        """输出为 float32。"""
        extractor = CZSCFeatureExtractor()
        df = _make_test_df()
        features = extractor.extract(df)
        assert features.dtype == np.float32

    def test_no_nan(self):
        """特征无 NaN。"""
        extractor = CZSCFeatureExtractor()
        df = _make_test_df()
        features = extractor.extract(df)
        assert not np.any(np.isnan(features))

    def test_no_inf(self):
        """特征无 Inf。"""
        extractor = CZSCFeatureExtractor()
        df = _make_test_df()
        features = extractor.extract(df)
        assert not np.any(np.isinf(features))

    def test_d1_range(self):
        """D1 (强分型) 值域 [-2.5, +2.5]。"""
        extractor = CZSCFeatureExtractor()
        df = _make_test_df()
        features = extractor.extract(df)
        assert features[:, 0].min() >= -2.5
        assert features[:, 0].max() <= 2.5

    def test_d2_values(self):
        """D2 (笔方向) 只含 {-1, 0, +1}。"""
        extractor = CZSCFeatureExtractor()
        df = _make_test_df()
        features = extractor.extract(df)
        unique = np.unique(features[:, 1])
        for v in unique:
            assert v in [-1.0, 0.0, 1.0], f"D2 含非法值: {v}"

    def test_d6_range(self):
        """D6 (拟合度) 值域 [0, 1]。"""
        extractor = CZSCFeatureExtractor()
        df = _make_test_df()
        features = extractor.extract(df)
        assert features[:, 5].min() >= 0.0
        assert features[:, 5].max() <= 1.0

    def test_short_input(self):
        """K 线数不足时返回全零。"""
        extractor = CZSCFeatureExtractor()
        df = _make_test_df(n=5)
        features = extractor.extract(df)
        assert features.shape == (5, 7)
        assert np.all(features == 0)

    def test_different_lengths(self):
        """不同长度输入均能处理。"""
        extractor = CZSCFeatureExtractor()
        for n in [50, 100, 200]:
            df = _make_test_df(n=n, seed=n)
            features = extractor.extract(df)
            assert features.shape == (n, 7)
            assert not np.any(np.isnan(features))

    def test_reproducibility(self):
        """相同输入产生相同输出。"""
        extractor = CZSCFeatureExtractor()
        df = _make_test_df()
        f1 = extractor.extract(df, symbol='test')
        f2 = extractor.extract(df, symbol='test')
        np.testing.assert_array_equal(f1, f2)

    def test_d1_no_fractal_is_zero(self):
        """D1: 绝大多数非分型 bar 的 D1 值为 0。"""
        extractor = CZSCFeatureExtractor()
        df = _make_test_df(n=200)
        features = extractor.extract(df)
        d1 = features[:, 0]
        # 分型 bar 稀疏，大多数应为 0
        zero_ratio = (d1 == 0.0).sum() / len(d1)
        assert zero_ratio > 0.5, f"非零 bar 占比 {1 - zero_ratio:.1%}，预期 < 50%"

    def test_d1_continuous_values(self):
        """D1: 增强后允许连续值（非纯整数）。"""
        extractor = CZSCFeatureExtractor()
        # 用多组随机数据增加出现影线/量能增强的概率
        has_continuous = False
        for seed in range(20):
            df = _make_test_df(n=200, seed=seed)
            features = extractor.extract(df)
            d1 = features[:, 0]
            nonzero = d1[d1 != 0.0]
            if len(nonzero) > 0:
                # 如果有任何非整数值，说明影线/量能增强生效
                non_integer = nonzero[nonzero != np.round(nonzero)]
                if len(non_integer) > 0:
                    has_continuous = True
                    break
        # 至少有一组数据产生了连续值
        assert has_continuous, "D1 未产生任何连续值，影线/量能增强可能未生效"

    def test_d1_large_volume_boosts_value(self):
        """D1: 放量分型应产生更高的绝对值。"""
        extractor = CZSCFeatureExtractor()
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2025-01-01', periods=n, freq='B')
        price = 10 + np.cumsum(np.random.randn(n) * 0.2)

        # 构造放量数据：在所有 bar 的基础上，随机让某些 bar 成交量放大 5 倍
        base_vol = np.random.rand(n) * 1000 + 100
        boosted_vol = base_vol.copy()
        boosted_vol[::10] *= 5  # 每 10 根 bar 放量一次

        df = pd.DataFrame({
            'open': price + np.random.rand(n) * 0.1,
            'high': price + np.abs(np.random.randn(n)) * 0.5,
            'low': price - np.abs(np.random.randn(n)) * 0.5,
            'close': price + np.random.randn(n) * 0.1,
            'vol': boosted_vol,
            'amt': boosted_vol * price,
        }, index=dates)
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)

        features = extractor.extract(df)
        d1 = features[:, 0]
        nonzero = np.abs(d1[d1 != 0.0])
        # 有非零值且最大值 > 2.0 说明量能增强生效
        if len(nonzero) > 0:
            assert nonzero.max() > 2.0, f"D1 最大绝对值 {nonzero.max():.3f}，量能增强未生效"


# ======================================================================
# TestRegressionCompatibility
# ======================================================================

class TestRegressionCompatibility:
    """确保 IIB 修改不破坏原有功能。"""

    def test_kronos_forward_no_cov_backward_compat(self):
        """不传 past_covariates 时，forward 仍正常工作（向后兼容）。"""
        from model.kronos import Kronos
        model = Kronos(
            s1_bits=9, s2_bits=9, n_layers=2, d_model=256,
            n_heads=4, ff_dim=512,
            ffn_dropout_p=0.1, attn_dropout_p=0.1,
            resid_dropout_p=0.1, token_dropout_p=0.1, learn_te=False
        )
        model.eval()
        s1 = torch.randint(0, 512, (1, 10))
        s2 = torch.randint(0, 512, (1, 10))
        with torch.no_grad():
            s1_logits, s2_logits = model(s1, s2)
        assert s1_logits.shape[0] == 1
        assert s2_logits.shape[0] == 1

    def test_decode_s1_backward_compat(self):
        """decode_s1 不传协变量时向后兼容。"""
        from model.kronos import Kronos
        model = Kronos(
            s1_bits=9, s2_bits=9, n_layers=2, d_model=256,
            n_heads=4, ff_dim=512,
            ffn_dropout_p=0.1, attn_dropout_p=0.1,
            resid_dropout_p=0.1, token_dropout_p=0.1, learn_te=False
        )
        model.eval()
        s1 = torch.randint(0, 512, (1, 10))
        s2 = torch.randint(0, 512, (1, 10))
        with torch.no_grad():
            logits, ctx = model.decode_s1(s1, s2)
        assert logits.shape[0] == 1

    def test_iib_none_is_noop(self):
        """IIB(covariates=None) 对模型无影响。"""
        iib = InputInjectionBlock(d_model=256, cov_dim=7, hidden_dim=64)
        x = torch.randn(1, 5, 256)
        out = iib(x, None)
        assert torch.all(out == 0).item()
        # x + out 应等于 x
        result = x + out
        assert torch.allclose(x, result)


# ======================================================================
# TestCovFill — 协变量填充模式测试
# ======================================================================

class TestCovFill:
    """auto_regressive_inference 协变量填充模式测试。

    通过 mock 捕获 decode_s1 收到的协变量，验证三种填充模式行为:
        - 'zero': 不滚动，向后兼容
        - 'last':  滚动 + 末值延续
        - 'decay': 滚动 + 指数衰减
    """

    @pytest.fixture
    def inference_setup(self):
        """创建推理测试所需的 mock 组件。"""
        from model.kronos import auto_regressive_inference  # noqa: F401

        vocab_size = 512
        tokenizer = _MockTokenizer(vocab_size)
        model = _MockModel(vocab_size)
        return tokenizer, model, model.captured_covs

    def test_zero_mode_no_rolling(self, inference_setup):
        """cov_fill='zero' 时 cov_buffer 不滚动，保持向后兼容。"""
        from model.kronos import auto_regressive_inference
        tokenizer, model, captured = inference_setup

        B, T, C = 1, 10, 6
        x = torch.randn(B, T, C)
        x_stamp = torch.randn(B, T, 4)
        y_stamp = torch.randn(B, 5, 4)
        cov = torch.randn(B, T, 7)

        auto_regressive_inference(
            tokenizer, model, x, x_stamp, y_stamp,
            max_context=10, pred_len=5, sample_count=1,
            past_covariates=cov, cov_fill='zero',
        )

        non_none = [c for c in captured if c is not None]
        assert len(non_none) >= 2, "至少应有 2 步捕获"
        # 所有步的 cov_buffer 应完全相同（不滚动）
        for i in range(1, len(non_none)):
            assert torch.allclose(non_none[i], non_none[0], atol=1e-6), \
                f"cov_fill='zero' 时第 {i} 步 cov_buffer 不应变化"

    def test_last_mode_rolling(self, inference_setup):
        """cov_fill='last' 时 cov_buffer 随滑动窗口滚动。"""
        from model.kronos import auto_regressive_inference
        tokenizer, model, captured = inference_setup

        B, T, C = 1, 10, 6
        x = torch.randn(B, T, C)
        x_stamp = torch.randn(B, T, 4)
        y_stamp = torch.randn(B, 5, 4)
        cov = torch.randn(B, T, 7)

        auto_regressive_inference(
            tokenizer, model, x, x_stamp, y_stamp,
            max_context=10, pred_len=5, sample_count=1,
            past_covariates=cov, cov_fill='last',
        )

        non_none = [c for c in captured if c is not None]
        assert len(non_none) >= 2
        # 滚动后每步 cov_buffer 应不同
        assert not torch.allclose(non_none[0], non_none[1], atol=1e-6), \
            "cov_fill='last' 时 cov_buffer 应随滑动变化"

        # 滚动后末位应等于最后已知协变量
        last_known = cov[0, -1, :]  # [7]
        for cov_tensor in non_none[1:]:
            last_pos = cov_tensor[0, -1, :]
            assert torch.allclose(last_pos, last_known, atol=1e-5), \
                "滚动后末位应等于最后已知协变量"

    def test_decay_mode_rolling(self, inference_setup):
        """cov_fill='decay' 时 cov_buffer 也随滑动窗口滚动。"""
        from model.kronos import auto_regressive_inference
        tokenizer, model, captured = inference_setup

        B, T, C = 1, 10, 6
        x = torch.randn(B, T, C)
        x_stamp = torch.randn(B, T, 4)
        y_stamp = torch.randn(B, 5, 4)
        cov = torch.randn(B, T, 7)

        auto_regressive_inference(
            tokenizer, model, x, x_stamp, y_stamp,
            max_context=10, pred_len=5, sample_count=1,
            past_covariates=cov, cov_fill='decay',
        )

        non_none = [c for c in captured if c is not None]
        assert len(non_none) >= 2
        assert not torch.allclose(non_none[0], non_none[1], atol=1e-6), \
            "cov_fill='decay' 时 cov_buffer 应随滑动变化"

    def test_no_covariates_all_modes(self, inference_setup):
        """past_covariates=None 时三种模式均正常工作。"""
        from model.kronos import auto_regressive_inference
        tokenizer, model, captured = inference_setup

        B, T, C = 1, 10, 6
        x = torch.randn(B, T, C)
        x_stamp = torch.randn(B, T, 4)
        y_stamp = torch.randn(B, 5, 4)

        for fill in ['zero', 'last', 'decay']:
            captured.clear()
            auto_regressive_inference(
                tokenizer, model, x, x_stamp, y_stamp,
                max_context=10, pred_len=5, sample_count=1,
                past_covariates=None, cov_fill=fill,
            )
            assert all(c is None for c in captured), \
                f"past_covariates=None 时 decode_s1 不应收到协变量 (fill={fill})"

    def test_default_fill_is_last(self, inference_setup):
        """默认 cov_fill='last'。"""
        from model.kronos import auto_regressive_inference
        tokenizer, model, captured = inference_setup

        B, T, C = 1, 10, 6
        x = torch.randn(B, T, C)
        x_stamp = torch.randn(B, T, 4)
        y_stamp = torch.randn(B, 5, 4)
        cov = torch.randn(B, T, 7)

        # 不传 cov_fill，应等同于 'last'
        auto_regressive_inference(
            tokenizer, model, x, x_stamp, y_stamp,
            max_context=10, pred_len=5, sample_count=1,
            past_covariates=cov,
        )

        non_none = [c for c in captured if c is not None]
        assert len(non_none) >= 2
        # 应该有滚动（非 zero 行为）
        assert not torch.allclose(non_none[0], non_none[1], atol=1e-6), \
            "默认 cov_fill 应为 'last'，cov_buffer 应滚动"


class _MockTokenizer:
    """用于测试的 mock tokenizer。"""

    def __init__(self, vocab_size=512):
        self.vocab_size = vocab_size

    def encode(self, x, half=True):
        B, T = x.shape[0], x.shape[1]
        return (
            torch.randint(0, self.vocab_size, (B, T)),
            torch.randint(0, self.vocab_size, (B, T)),
        )

    def decode(self, tokens, half=True):
        B, T = tokens[0].shape[0], tokens[0].shape[1]
        return torch.randn(B, T, 6)


class _MockModel:
    """用于测试的 mock model，捕获传入 decode_s1 的协变量。"""

    def __init__(self, vocab_size=512):
        self.vocab_size = vocab_size
        self.captured_covs = []

    def decode_s1(self, s1, s2, stamp=None, padding_mask=None, past_covariates=None):
        if past_covariates is not None:
            self.captured_covs.append(past_covariates.detach().clone())
        else:
            self.captured_covs.append(None)
        B, T = s1.shape[0], s1.shape[1]
        s1_logits = torch.randn(B, T, self.vocab_size)
        context = torch.randn(B, T, 256)
        return s1_logits, context

    def decode_s2(self, context, s1_ids, padding_mask=None):
        B, T = context.shape[0], context.shape[1]
        return torch.randn(B, T, self.vocab_size)

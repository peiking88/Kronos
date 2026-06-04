"""
IIB 模块和 CZSC 特征提取器单元测试。

测试覆盖:
    - InputInjectionBlock: 输出形状、None 时零输出、参数量、梯度流通
    - KronosWithIIB: 无协变量时一致性、有协变量时形状正确
    - CZSCFeatureExtractor: 提取形状、无 NaN/Inf、值域范围
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
        """D1 (强分型) 值域 [-2, +2]。"""
        extractor = CZSCFeatureExtractor()
        df = _make_test_df()
        features = extractor.extract(df)
        assert features[:, 0].min() >= -2.0
        assert features[:, 0].max() <= 2.0

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

import os


class TdxFineTuneConfig:
    """
    Single-GPU fine-tuning configuration for Kronos using TDX local data.

    TDX local data covers ~1996 to present with ~6500 A-share stocks (ex-BJ).
    Uses 后复权 (hfq/back adjustment) to match original Kronos Qlib training data.
    """

    def __init__(self):
        # =================================================================
        # Data & Feature Parameters
        # =================================================================
        self.qlib_data_path = "~/.qlib/qlib_data/cn_data"  # Not used; kept for compat
        self.instrument = 'csi300'

        # TDX data date range (full history to present)
        self.dataset_begin_time = "2011-01-01"
        self.dataset_end_time = "2026-05-16"

        # Sliding window parameters
        self.lookback_window = 90
        self.predict_window = 10
        self.max_context = 512

        # 6-field Kronos format
        self.feature_list = ['open', 'high', 'low', 'close', 'vol', 'amt']
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']

        # 后复权 (hfq) — matches original Kronos Qlib training data convention
        self.dividend_type = "back"

        # =================================================================
        # Dataset Splitting & Paths
        # =================================================================
        # Non-overlapping splits: train(14yr) | val(3.5mo) | test(3mo)
        self.train_time_range = ["2011-01-01", "2025-10-31"]
        self.val_time_range = ["2025-11-01", "2026-02-14"]
        self.test_time_range = ["2026-02-15", "2026-05-16"]
        self.backtest_time_range = ["2026-02-15", "2026-05-16"]

        # Directory for processed pickle datasets (produced by tdx_import.py)
        self.dataset_path = "./data/tdx_import"

        # =================================================================
        # Training Hyperparameters (Single GPU)
        # =================================================================
        self.clip = 5.0

        self.epochs = 30
        self.log_interval = 50

        # Batch sizes tuned for GPU VRAM
        self.batch_size = 128        # Tokenizer: bf16 AMP (RTX 5080 16GB)
        self.predictor_batch_size = 128  # Predictor: bf16 AMP (RTX 5080 16GB)
        self.predictor_accumulation = 1  # Effective bs = 128 * 1 = 128

        # Number of samples per epoch
        self.n_train_iter = 1000 * self.batch_size
        self.n_val_iter = 200 * self.batch_size

        # Learning rates
        self.tokenizer_learning_rate = 2e-4
        self.predictor_learning_rate = 4e-5

        # Early stopping
        self.early_stop_patience = 5

        # Gradient accumulation (tokenizer)
        self.accumulation_steps = 1

        # AdamW
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.1

        # Data loading
        self.num_workers = 2

        # Reproducibility
        self.seed = 100

        # =================================================================
        # Mixed Precision
        # =================================================================
        self.use_amp = True  # bf16 AMP for training (RTX 5080 原生 bf16)

        # =================================================================
        # Phase Control — 两阶段训练
        # =================================================================
        self.phase = 'full'                # 'full'=Phase1 全参数微调, 'iib'=Phase2 IIB训练
        self.phase1_epochs = 10            # Phase 1 epoch 数（热身，不宜过多）

        # =================================================================
        # Model Paths — downloaded from HuggingFace via hf-mirror.com
        # =================================================================
        self.pretrained_tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
        self.pretrained_predictor_path = "NeoQuasar/Kronos-base"

        # =================================================================
        # Experiment Logging & Saving
        # =================================================================
        self.use_comet = False  # Disabled by default for local training

        self.save_path = "./outputs/tdx_finetune"
        self.tokenizer_save_folder_name = 'tdx_tokenizer'
        self.predictor_save_folder_name = 'tdx_predictor'
        self.backtest_save_folder_name = 'tdx_backtest'

        # Finetuned model paths (set after tokenizer training)
        self.finetuned_tokenizer_path = (
            f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        )
        self.finetuned_predictor_path = (
            f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"
        )

        # =================================================================
        # IIB (Input Injection Block) 配置
        # =================================================================
        self.use_iib = True                        # 是否启用 IIB 协变量注入
        self.cov_dim = 7                           # 协变量维度（CZSC 7 维特征）
        self.iib_hidden_dim = 256                  # IIB 内部隐藏维度
        self.iib_dropout = 0.3                     # IIB dropout（从 0.1 提高）
        self.iib_n_layers = 2                      # IIB 残差 MLP 层数（从 1 升级）
        self.iib_learning_rate = 3e-4              # IIB 学习率（从 1e-3 降低）
        self.iib_weight_decay = 0.2                # IIB 权重衰减
        self.freeze_predictor = True               # 冻结 Kronos 主体，仅训练 IIB
        self.czsc_cache_path = os.path.join(self.dataset_path, "czsc_features")

        # 渐进式解冻（Phase 2）
        self.iib_only_epochs = 5                   # Stage A: 仅训练 IIB
        self.iib_plus_top_epochs = 5               # Stage B: IIB + 后 4 层
        self.transformer_top_lr = 1e-5             # Stage B/C 顶层学习率
        self.transformer_base_lr = 5e-6            # Stage C 全参数学习率（极低）

        # =================================================================
        # Backtesting
        # =================================================================
        self.backtest_result_path = "./outputs/tdx_backtest_results"
        self.backtest_n_symbol_hold = 50
        self.backtest_n_symbol_drop = 5
        self.backtest_hold_thresh = 5
        self.inference_T = 0.6
        self.inference_top_p = 0.9
        self.inference_top_k = 0
        self.inference_sample_count = 5
        self.backtest_batch_size = 1000
        self.backtest_benchmark = "SH000300"

    def to_dict(self):
        """Return config as dict for training scripts."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

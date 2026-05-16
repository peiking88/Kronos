import os


class TdxFineTuneConfig:
    """
    Single-GPU fine-tuning configuration for Kronos using TDX local data.

    TDX local data covers ~2024-06-03 to ~2026-04-30 with ~5250 A-share stocks.
    Uses 后复权 (hfq/back adjustment) to match original Kronos Qlib training data.
    The short history (~2 years) makes this a domain-adaptation fine-tune.
    """

    def __init__(self):
        # =================================================================
        # Data & Feature Parameters
        # =================================================================
        self.qlib_data_path = "~/.qlib/qlib_data/cn_data"  # Not used; kept for compat
        self.instrument = 'csi300'

        # TDX data adjusted to available local data range (2024-06 ~ 2026-04)
        self.dataset_begin_time = "2024-06-01"
        self.dataset_end_time = "2026-04-30"

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
        # Narrower splits to fit ~2 years of TDX data
        self.train_time_range = ["2024-06-01", "2025-12-31"]
        self.val_time_range = ["2025-10-01", "2026-03-31"]
        self.test_time_range = ["2026-01-01", "2026-04-30"]
        self.backtest_time_range = ["2026-01-01", "2026-04-30"]

        # Directory for processed pickle datasets (produced by tdx_import.py)
        self.dataset_path = "./data/tdx_import"

        # =================================================================
        # Training Hyperparameters (Single GPU)
        # =================================================================
        self.clip = 5.0

        self.epochs = 30
        self.log_interval = 50

        # Batch sizes tuned for RTX 4060 Laptop 8GB
        self.batch_size = 50         # Tokenizer: OK at bs=50 (fp32, ~5GB)
        self.predictor_batch_size = 12  # Predictor: bs=12 with AMP fp16
        self.predictor_accumulation = 4  # Effective bs = 12 * 4 = 48

        # Number of samples per epoch
        self.n_train_iter = 2000 * self.batch_size
        self.n_val_iter = 400 * self.batch_size

        # Learning rates
        self.tokenizer_learning_rate = 2e-4
        self.predictor_learning_rate = 4e-5

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
        self.use_amp = True  # Automatic Mixed Precision for predictor training

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

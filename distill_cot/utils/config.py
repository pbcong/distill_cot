import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration for training and model parameters."""

    def __init__(self):
        # Model settings
        self.model_name: str = os.getenv("MODEL_NAME", "Qwen/Qwen1.5-0.5B-Chat")

        # Dataset settings
        self.dataset_name: str = os.getenv("DATASET_NAME", "pbcong/gsm8k_step_by_step")
        self.dataset_config: str = os.getenv("DATASET_CONFIG", "main")

        # Output settings
        self.output_dir: str = os.getenv("OUTPUT_DIR", "./results")
        self.logs_dir: str = os.getenv("LOGS_DIR", "./logs")

        # LoRA settings
        self.lora_rank: int = int(os.getenv("LORA_RANK", "16"))
        self.lora_alpha: int = int(os.getenv("LORA_ALPHA", "32"))
        self.lora_dropout: float = float(os.getenv("LORA_DROPOUT", "0.05"))

        # Training settings
        self.learning_rate: float = float(os.getenv("LEARNING_RATE", "2e-5"))
        self.per_device_train_batch_size: int = int(os.getenv("TRAIN_BATCH_SIZE", "2"))
        self.per_device_eval_batch_size: int = int(os.getenv("EVAL_BATCH_SIZE", "2"))
        self.num_train_epochs: float = float(os.getenv("NUM_EPOCHS", "0.5"))
        self.warmup_steps: int = int(os.getenv("WARMUP_STEPS", "500"))
        self.weight_decay: float = float(os.getenv("WEIGHT_DECAY", "0.01"))

        # Logging and evaluation settings
        self.logging_steps: int = int(os.getenv("LOGGING_STEPS", "100"))
        self.save_steps: int = int(os.getenv("SAVE_STEPS", "1000"))
        self.eval_steps: int = int(os.getenv("EVAL_STEPS", "100000"))
        self.evaluation_strategy: str = os.getenv("EVAL_STRATEGY", "steps")
        self.load_best_model_at_end: bool = (
            os.getenv("LOAD_BEST_MODEL", "true").lower() == "true"
        )
        self.save_total_limit: int = int(os.getenv("SAVE_TOTAL_LIMIT", "3"))

        # Generation settings
        self.predict_with_generate: bool = True
        self.max_length_questions: int = int(os.getenv("MAX_LENGTH_QUESTIONS", "512"))
        self.max_length_answers: int = int(os.getenv("MAX_LENGTH_ANSWERS", "256"))
        self.max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "256"))

        # Computed settings
        self.max_length: int = self.max_length_questions + self.max_length_answers

        self._validate_config()
        self._create_directories()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.max_length_questions <= 0 or self.max_length_answers <= 0:
            raise ValueError("Max lengths must be positive integers")

        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        if self.num_train_epochs <= 0:
            raise ValueError("Number of training epochs must be positive")

        if not isinstance(self.load_best_model_at_end, bool):
            raise ValueError("load_best_model_at_end must be a boolean")

        if self.evaluation_strategy not in ["steps", "epoch", "no"]:
            raise ValueError("Invalid evaluation strategy")

    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        try:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            Path(self.logs_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directories: {self.output_dir}, {self.logs_dir}")
        except Exception as e:
            logger.error(f"Error creating directories: {str(e)}")
            raise

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }

    @classmethod
    def from_env_file(cls, env_path: str) -> "Config":
        """Create config from environment file."""
        if not os.path.exists(env_path):
            logger.warning(f"Environment file {env_path} not found, using defaults")
            return cls()

        # Load environment variables from file
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value

        return cls()


def test_config():
    """Test configuration functionality."""
    try:
        config = Config()
        print("Configuration loaded successfully:")
        for key, value in config.to_dict().items():
            print(f"{key}: {value}")
        return config
    except Exception as e:
        logger.error(f"Error in test_config: {str(e)}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_config()

import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from distill_cot.data.dataset import DataProcessor
from distill_cot.training.trainer import ModelTrainer
from distill_cot.utils.config import Config

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("training.log"),
        ],
    )


def load_model_and_tokenizer(config: Config):
    """Load the model and tokenizer."""
    try:
        logger.info(f"Loading model and tokenizer from {config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name, device_map="auto" if torch.cuda.is_available() else None
        )

        # Ensure the tokenizer has necessary tokens
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        return model, tokenizer

    except Exception as e:
        logger.error(f"Error loading model and tokenizer: {str(e)}")
        raise


def prepare_dataset(config: Config, tokenizer: AutoTokenizer):
    """Load and prepare the dataset."""
    try:
        logger.info(f"Loading dataset {config.dataset_name}")
        dataset = load_dataset(config.dataset_name, config.dataset_config)

        logger.info("Processing dataset")
        processor = DataProcessor(tokenizer, config)
        processed_dataset = dataset.map(
            processor.get_preprocess_function(),
            batched=True,
            remove_columns=dataset["train"].column_names,
        )

        return processed_dataset

    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}")
        raise


def train(config_path: Optional[str] = None, log_level: str = "INFO") -> None:
    """
    Main training function.

    Args:
        config_path: Optional path to environment file for configuration
        log_level: Logging level to use
    """
    try:
        # Setup
        setup_logging(log_level)
        logger.info("Starting training process")

        # Load configuration
        config = Config.from_env_file(config_path) if config_path else Config()
        logger.info("Configuration loaded successfully")

        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(config)

        # Prepare dataset
        processed_dataset = prepare_dataset(config, tokenizer)

        # Initialize trainer
        trainer = ModelTrainer(model, tokenizer, config)

        # Start training
        logger.info("Starting training")
        results = trainer.train(
            train_dataset=processed_dataset["train"],
            eval_dataset=processed_dataset.get("validation"),
        )

        logger.info("Training completed successfully")
        logger.info(f"Training results: {results}")

        # Save final model
        final_model_path = Path(config.output_dir) / "final_model"
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        logger.info(f"Model and tokenizer saved to {final_model_path}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


def main():
    """Entry point for the training script."""
    import argparse

    parser = argparse.ArgumentParser(description="Train a language model")
    parser.add_argument(
        "--config", type=str, help="Path to environment file for configuration"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    args = parser.parse_args()
    train(config_path=args.config, log_level=args.log_level)


if __name__ == "__main__":
    main()

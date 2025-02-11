import logging
from typing import Any, Dict, Optional

from datasets import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from distill_cot.utils.config import Config

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training and evaluation."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Config,
        data_collator: Optional[DataCollatorForSeq2Seq] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: The model to train
            tokenizer: Tokenizer for processing text
            config: Training configuration
            data_collator: Optional custom data collator
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Initialize data collator if not provided
        self.data_collator = data_collator or DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )

        # Create training arguments
        self.training_args = self._create_training_args()

    def _create_training_args(self) -> Seq2SeqTrainingArguments:
        """Create training arguments from config."""
        generation_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        return Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir=self.config.logs_dir,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            save_total_limit=self.config.save_total_limit,
            predict_with_generate=self.config.predict_with_generate,
            learning_rate=self.config.learning_rate,
            generation_config=generation_config,
        )

    def create_trainer(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[callable] = None,
    ) -> Seq2SeqTrainer:
        """
        Create a trainer instance.

        Args:
            train_dataset: Dataset for training
            eval_dataset: Optional dataset for evaluation
            compute_metrics: Optional function for computing metrics

        Returns:
            Configured Seq2SeqTrainer instance
        """
        return Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
        )

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            train_dataset: Dataset for training
            eval_dataset: Optional dataset for evaluation
            compute_metrics: Optional function for computing metrics

        Returns:
            Training results
        """
        try:
            trainer = self.create_trainer(train_dataset, eval_dataset, compute_metrics)
            results = trainer.train()
            logger.info("Training completed successfully")
            return results

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise


def test_trainer():
    """Test the trainer functionality."""
    try:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from distill_cot.data.dataset import DataProcessor

        # Initialize components
        config = Config()
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        model = AutoModelForCausalLM.from_pretrained(config.model_name)

        # Prepare dataset
        data = load_dataset(config.dataset_name)
        processor = DataProcessor(tokenizer, config)
        processed_dataset = data.map(
            processor.get_preprocess_function(),
            batched=True,
            remove_columns=data["train"].column_names,
        )

        # Create and test trainer
        trainer = ModelTrainer(model, tokenizer, config)
        test_trainer = trainer.create_trainer(processed_dataset["train"])
        print("Trainer created successfully:", test_trainer)

        return trainer

    except Exception as e:
        logger.error(f"Error in test_trainer: {str(e)}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_trainer()

from functools import partial
from typing import Dict, List, Any
from datasets import Dataset
from transformers import PreTrainedTokenizer
import logging

from distill_cot.utils.config import Config

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data preprocessing for training language models."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, config: Config):
        """
        Initialize the data processor.
        
        Args:
            tokenizer: The tokenizer to use for processing text
            config: Configuration object containing processing parameters
        """
        self.tokenizer = tokenizer
        self.config = config
        
    def preprocess_function(self, examples: Dict[str, List[Any]], 
                          is_training: bool = True) -> Dict[str, Any]:
        """
        Preprocess the examples for model training or inference.
        
        Args:
            examples: Dictionary containing 'question' and 'answer' lists
            is_training: Whether preprocessing is for training (affects label creation)
            
        Returns:
            Dictionary containing model inputs (input_ids, attention_mask, labels)
        """
        try:
            sources = []
            targets = []
            
            # Process each example
            for question, answer in zip(examples['question'], examples['answer']):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}
                ]
                source = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                target = answer + self.tokenizer.eos_token
                sources.append(source)
                targets.append(target)

            # Combine sources and targets for tokenization
            combined_texts = [s + t for s, t in zip(sources, targets)]
            
            # Tokenize combined texts
            tokenized_examples = self.tokenizer(
                combined_texts,
                max_length=self.config.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            if is_training:
                # Create labels by masking source tokens
                labels = tokenized_examples['input_ids'].clone()
                
                # Get source lengths for masking
                tokenized_sources = self.tokenizer(
                    sources,
                    truncation=True,
                    padding='max_length',
                    max_length=self.config.max_length_questions,
                    return_tensors='pt'
                )
                source_lengths = [len(ids) for ids in tokenized_sources['input_ids']]
                
                # Mask source tokens in labels
                for i in range(labels.shape[0]):
                    labels[i, :source_lengths[i]] = -100
                
                tokenized_examples['labels'] = labels
            
            return tokenized_examples
            
        except Exception as e:
            logger.error(f"Error preprocessing examples: {str(e)}")
            raise

    def get_preprocess_function(self, is_training: bool = True):
        """
        Returns a partial function for preprocessing with fixed parameters.
        
        Args:
            is_training: Whether the preprocessing is for training data
            
        Returns:
            Partial function that takes only examples as argument
        """
        return partial(self.preprocess_function, is_training=is_training)

def test_processor():
    """Test the data processor functionality."""
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        from distill_cot.utils.config import Config
        
        # Load test data and tokenizer
        config = Config()
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        data = load_dataset(config.dataset_name)
        
        # Create processor and process data
        processor = DataProcessor(tokenizer, config)
        tokenized_dataset = data.map(
            processor.get_preprocess_function(),
            batched=True,
            remove_columns=data['train'].column_names
        )
        
        print("Processed dataset:", tokenized_dataset)
        return tokenized_dataset
        
    except Exception as e:
        logger.error(f"Error in test_processor: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_processor()

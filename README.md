# Model Distillation Framework

A flexible and extensible framework for knowledge distillation of large language models into smaller, more efficient models. This framework supports both logit-based and language output-based distillation approaches.

## Features

- Two distillation modes:
  - **Logit-based**: Uses the teacher model's logits to guide the student model's training
  - **Language output-based**: Compares generated text outputs between teacher and student models
- Extensible architecture using OOP principles
- Factory pattern for easy distiller creation
- Configurable parameters for fine-tuning the distillation process
- Built-in support for model evaluation and validation
- Example implementation with BERT models

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/distill-cot.git
cd distill-cot

# Install dependencies
pip install torch transformers datasets
```

## Quick Start

Here's a simple example of how to use the framework:

```python
from transformers import BertForSequenceClassification, BertTokenizerFast
from src.distiller.factory import DistillerFactory

# Initialize models
teacher = BertForSequenceClassification.from_pretrained('bert-base-uncased')
student_config = teacher.config
student_config.num_hidden_layers = 6  # Smaller model
student = BertForSequenceClassification.from_config(student_config)

# Create distiller
distiller = DistillerFactory.create_distiller(
    mode="logit",
    teacher_model=teacher,
    student_model=student,
    config={
        'temperature': 2.0,
        'alpha': 0.7
    }
)

# Train the student model
distiller.train(train_dataloader, val_dataloader)
```

## Framework Structure

```
distill-cot/
├── src/
│   └── distiller/
│       ├── base.py         # Abstract base class for distillers
│       ├── logit.py        # Logit-based distillation implementation
│       ├── language.py     # Language output-based distillation
│       └── factory.py      # Factory class for creating distillers
├── examples/
│   └── distill_bert.py     # Example using BERT models
└── README.md
```

## Distillation Modes

### Logit-based Distillation

Uses the teacher model's logits (pre-softmax outputs) to guide the training of the student model. This approach is based on the paper "Distilling the Knowledge in a Neural Network" by Hinton et al.

```python
distiller = DistillerFactory.create_distiller(
    mode="logit",
    teacher_model=teacher,
    student_model=student,
    config={
        'temperature': 2.0,  # Controls softening of probability distributions
        'alpha': 0.7        # Balances distillation and task-specific losses
    }
)
```

### Language Output-based Distillation

Compares the generated text sequences between teacher and student models using various text similarity metrics.

```python
distiller = DistillerFactory.create_distiller(
    mode="language",
    teacher_model=teacher,
    student_model=student,
    tokenizer=tokenizer,    # Required for text processing
    config={
        'temperature': 1.0,
        'max_length': 128   # Maximum sequence length for generation
    }
)
```

## Configuration Options

### Common Parameters

- `temperature`: Controls the softening of probability distributions
- `config`: Additional configuration dictionary for custom parameters

### Logit Mode Parameters

- `alpha`: Weight for balancing distillation and task-specific losses

### Language Mode Parameters

- `max_length`: Maximum sequence length for text generation
- `tokenizer`: Tokenizer instance for processing text inputs/outputs

## Example Usage

Check out `examples/distill_bert.py` for a complete example of distilling a BERT-base model into a smaller 6-layer version using the SST-2 sentiment analysis dataset.

## Extending the Framework

The framework is designed to be easily extensible. To add a new distillation approach:

1. Create a new class that inherits from `BaseDistiller`
2. Implement the required abstract methods:
   - `compute_loss()`
   - `train_step()`
   - `validate()`
3. Add the new distiller to the factory class

Example:

```python
class CustomDistiller(BaseDistiller):
    def compute_loss(self, *args, **kwargs):
        # Implement your custom loss computation
        pass

    def train_step(self, *args, **kwargs):
        # Implement your training step
        pass

    def validate(self, *args, **kwargs):
        # Implement your validation logic
        pass
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

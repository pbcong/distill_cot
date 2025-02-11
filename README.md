# Distill CoT

A Python project for training and fine-tuning language models with Chain-of-Thought reasoning capabilities.

**WIP**
Need to add:

- Allow training on multiple gpu
- Fix some training bug
- create pipeline for cot data creation

## Project Structure

```
distill_cot/
├── distill_cot/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py      # Dataset processing and preparation
│   ├── models/
│   │   └── __init__.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py      # Training logic and configuration
│   ├── utils/
│   │   ├── __init__.py
│   │   └── config.py       # Configuration management
│   └── __init__.py
├── main.py                 # Main entry point
├── pyproject.toml          # Project metadata and dependencies
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Features

- Flexible configuration management with environment variable support
- Structured dataset processing for Chain-of-Thought training
- Robust training pipeline with logging and error handling
- Support for model fine-tuning with LoRA
- Comprehensive logging and model checkpointing

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/distill_cot.git
cd distill_cot
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

The project can be configured through environment variables or a .env file. Key configuration options include:

- `MODEL_NAME`: Base model to use (default: "Qwen/Qwen1.5-0.5B-Chat")
- `DATASET_NAME`: Dataset to use for training (default: "pbcong/gsm8k_step_by_step")
- `OUTPUT_DIR`: Directory for saving model outputs (default: "./results")
- `LOGS_DIR`: Directory for saving training logs (default: "./logs")

See `distill_cot/utils/config.py` for all available configuration options.

## Usage

1. Basic training:

```bash
python main.py
```

2. Training with custom configuration:

```bash
python main.py --config path/to/config.env --log-level DEBUG
```

## Project Components

### Data Processing (`dataset.py`)

The `DataProcessor` class handles:

- Dataset loading and preprocessing
- Text tokenization and formatting
- Label creation for training

### Training (`trainer.py`)

The `ModelTrainer` class provides:

- Training configuration and setup
- Model training and evaluation
- Checkpoint management
- Training metrics logging

### Configuration (`config.py`)

The `Config` class manages:

- Environment variable loading
- Configuration validation
- Directory creation
- Default value handling

## Development

1. Install development dependencies:

```bash
pip install -e ".[dev]"
```

2. Set up pre-commit hooks:

```bash
pre-commit install
```

This will set up the following checks to run automatically on git commit:

- black: Code formatting
- isort: Import sorting
- ruff: Fast Python linting
- Additional checks for YAML, TOML, and file formatting

3. Run tests:

```bash
pytest
```

You can also run the pre-commit hooks manually on all files:

```bash
pre-commit run --all-files
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

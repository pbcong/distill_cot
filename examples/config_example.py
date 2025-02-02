import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import Config

def main():
    # Load the configuration
    config = Config.from_yaml("examples/llm_config.yaml")
    
    # Print configuration sections
    print("\nTeacher Model Configuration:")
    print(f"Name: {config.model.teacher.name}")
    print(f"Path: {config.model.teacher.path}")
    print(f"Device: {config.model.teacher.device}")
    
    print("\nStudent Model Configuration:")
    print(f"Path: {config.model.student}")
    
    print("\nDistillation Configuration:")
    print(f"Temperature: {config.distillation.temperature}")
    print(f"Alpha: {config.distillation.alpha}")
    
    print("\nTraining Configuration:")
    print(f"Batch Size: {config.training.batch_size}")
    print(f"Number of Epochs: {config.training.num_epochs}")
    print(f"Learning Rate: {config.training.learning_rate}")
    print(f"Max Length: {config.training.max_length}")

if __name__ == "__main__":
    main()

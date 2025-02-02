from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class StudentModel:
    def __init__(self, model_name: str, device: str = 'cpu'):
        """Initialize the student model.
        
        Args:
            model_name: Name of the pre-trained model to load.
            device: Device to load the model on ('cpu' or 'cuda').
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """Forward pass through the model.
        
        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask for the input IDs.
            
        Returns:
            Model outputs.
        """
        return self.model(input_ids=input_ids, attention_mask=attention_mask)
    
    def save(self, save_path: str):
        """Save the model and tokenizer.
        
        Args:
            save_path: Path to save the model and tokenizer.
        """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

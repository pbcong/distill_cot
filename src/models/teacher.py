"""Teacher model for distillation using DeepSeek API."""

import torch
import requests
import json
import os

class TeacherModel:
    def __init__(self, model_name: str, device: str = 'cpu', api_key: str = ""):
        """Initialize the teacher model.
        
        Args:
            model_name: Name/ID of the DeepSeek model to use
            device: Device for tensor operations
            api_key: DeepSeek API key (optional, can also use DEEPSEEK_API_KEY env var)
        """
        self.model_name = model_name
        self.device = device
        
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("DeepSeek API key must be provided or set in DEEPSEEK_API_KEY environment variable")
            
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        self.api_url = "https://api.openai.com/v1/chat/completions"
    
    def forward(self, text: str = None):
        if text is None:
            raise ValueError("text parameter is required for DeepSeek API calls")
            
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": text}],
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": False
        }
        
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"DeepSeek API error: {response.text}")
            
        result = response.json()
        
        # Extract response text
        response_text = result['choices'][0]['message']['content']
        cot = result['choices'][0]['message']['reasoning_content']
        # # Convert to logits if available, otherwise return None
        # logits = torch.tensor(result.get('logits', [])).to(self.device) if 'logits' in result else None
        
        return {
            # 'logits': logits,
            'text': response_text,
            'cot': cot
        }

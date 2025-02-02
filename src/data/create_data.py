from typing import List, Dict, Any
import json
from pathlib import Path
from tqdm import tqdm
from utils.config import config
from models.teacher import TeacherModel
from datasets import load_dataset
import os
import dotenv
import datasets
dotenv.load_dotenv()

def create_prompt(question: str) -> str:
    return f"{question}\n"

def generate_output(
    model: TeacherModel,
    questions: List[str],
    temperature: float = 0.7
) -> List[str]:
    chain_of_thoughts = []
    for question in tqdm(questions, desc="Generating chain-of-thought"):
        prompt = create_prompt(question)
        output = model.forward(text=prompt)
        chain_of_thoughts.append(output)
    return chain_of_thoughts

def save_augmented_data(
    data: List[Dict[str, Any]],
    chain_of_thoughts: List[str],
    output_path: str = "augmented_data.json",
    upload: str = None
):
    augmented_data = []
    
    for item, cot in zip(data, chain_of_thoughts):
        augmented_item = item.copy()
        augmented_item['chain_of_thought'] = cot
        augmented_data.append(augmented_item)
    
    if upload:
        dataset = datasets.from_dict(augmented_data)
        dataset.push_to_hub(upload, use_temp_dir=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(augmented_data, f, indent=2)

def main():
    Teacher = TeacherModel(model_name=config.model.teacher.name, device="cuda", api_key=os.environ.get('DEEPSEEK_API_KEY'))
    data = load_dataset("pbcong/gsm8k_step_by_step", split=config.data.split)
    questions = data['question']
    chain_of_thoughts = generate_output(Teacher, questions)
    save_augmented_data(data, chain_of_thoughts, config.data.output_path, config.data.upload if hasattr(config.data, 'upload') else None)
if __name__ == "__main__":
    main()

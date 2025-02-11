import evaluate
import numpy as np
from datasets import load_dataset
from peft import LoftQConfig, LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

import distill_cot.utils.config as config

# --- Model and Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(
    config.model_name, padding_side="left", trust_remote_code=True
)  # CRUCIAL: padding_side="left"
# Qwen requires trust_remote_code=True

model = AutoModelForCausalLM.from_pretrained(
    config.model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True
)

# Add pad token if not already present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

prompt = "A family of 12 monkeys collected 10 piles of bananas. 6 piles had 9 hands, with each hand having 14 bananas, while the remaining piles had 12 hands, with each hand having 9 bananas. How many bananas would each monkey get if they divide the bananas equally amongst themselves?"
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that can solve math problems.",
    },
    {"role": "user", "content": prompt},
]

text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(
    model.device
)  # No padding during inference, typically
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=config.max_new_tokens,  # Use max_new_tokens during generation
    do_sample=False,  # for evaluation, best to not sample
)
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
# --- LoRA and LoftQ ---
loftq_config = LoftQConfig(loftq_bits=4)
lora_config = LoraConfig(
    init_lora_weights="loftq",  # Use LoftQ initialization
    loftq_config=loftq_config,
    r=config.lora_rank,
    lora_alpha=config.lora_alpha,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "w1",
        "w2",
    ],  # include w1 and w2 for Qwen2
    lora_dropout=config.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)
peft_model = get_peft_model(model, lora_config)

# --- Dataset ---
data = load_dataset(config.dataset_name)


def preprocess_function(examples):
    # Format with chat template.
    sources = []
    targets = []
    for question, answer in zip(examples["question"], examples["answer"]):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ]
        # Use apply_chat_template here
        source = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        target = answer + tokenizer.eos_token  # ensure end of seq token
        sources.append(source)
        targets.append(target)

    # Tokenize sources and targets TOGETHER to form model inputs
    examples = [s + t for s, t in zip(sources, targets)]  # combine for tokenizer

    tokenized_examples = tokenizer(
        examples,
        max_length=config.max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Create labels, and mask out the source part.  Clone input_ids for labels.
    labels = tokenized_examples["input_ids"].clone()

    # Tokenize ONLY sources again to find the lengths of the source sequences.  Key for masking.
    tokenized_sources = tokenizer(
        sources,
        truncation=True,
        padding="max_length",
        max_length=config.max_length_questions,
        return_tensors="pt",
    )  # padding and truncation
    source_lengths = [len(ids) for ids in tokenized_sources["input_ids"]]

    # Replace source tokens in labels with -100. Iterate through each example in the batch.
    for i in range(labels.shape[0]):
        labels[i, : source_lengths[i]] = -100  # correct indexing

    # Prepare model inputs
    model_inputs = {
        "input_ids": tokenized_examples["input_ids"],
        "attention_mask": tokenized_examples["attention_mask"],
        "labels": labels,
    }

    return model_inputs


tokenized_dataset = data.map(
    preprocess_function, batched=True, remove_columns=data["train"].column_names
)

# --- Evaluation Metric (Exact Match) ---
import evaluate
import numpy as np

rouge = evaluate.load("rouge")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Compute ROUGE scores
    result = rouge.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )  # use_stemmer helps a bit
    # Extract a few ROUGE scores for logging
    result = {
        key: value * 100 for key, value in result.items()
    }  # Convert to percentages

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)  # Add generation length
    return {k: round(v, 4) for k, v in result.items()}


# --- Training Arguments ---
generation_config = GenerationConfig(
    max_new_tokens=4096, pad_token_id=tokenizer.pad_token_id
)
train_args = Seq2SeqTrainingArguments(
    output_dir=config.output_dir,
    num_train_epochs=config.num_train_epochs,
    max_steps=100,  # Use max_steps for shorter training
    per_device_train_batch_size=config.per_device_train_batch_size,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    warmup_steps=config.warmup_steps,
    weight_decay=config.weight_decay,
    logging_dir=config.logs_dir,
    logging_steps=config.logging_steps,
    save_steps=config.save_steps,
    eval_steps=config.eval_steps,
    evaluation_strategy=config.evaluation_strategy,
    load_best_model_at_end=config.load_best_model_at_end,
    save_total_limit=config.save_total_limit,
    predict_with_generate=True,
    learning_rate=config.learning_rate,
    # generation_max_length is only a fallback, max_new_tokens is preferred
    generation_config=generation_config,
)

# --- Data Collator ---
# Use DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
)  # pad_to_multiple_of can help with efficiency


# --- Trainer ---
trainer = Seq2SeqTrainer(
    model=peft_model,
    args=train_args,
    train_dataset=tokenized_dataset["train"],
    # eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    # compute_metrics=compute_metrics,
)

# --- Train ---
trainer.train()

# --- Save Model ---
peft_model.save_pretrained(config.output_dir)  # Save the LoRA adapter
tokenizer.save_pretrained(config.output_dir)

# --- Inference (Example) ---
prompt = "A family of 12 monkeys collected 10 piles of bananas. 6 piles had 9 hands, with each hand having 14 bananas, while the remaining piles had 12 hands, with each hand having 9 bananas. How many bananas would each monkey get if they divide the bananas equally amongst themselves?"
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that can solve math problems.",
    },
    {"role": "user", "content": prompt},
]

text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(
    model.device
)  # No padding during inference, typically
generated_ids = peft_model.generate(
    **model_inputs,
    max_new_tokens=config.max_new_tokens,  # Use max_new_tokens during generation
    do_sample=False,  # for evaluation, best to not sample
)
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

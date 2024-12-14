# Import necessary libraries
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import pandas as pd
import ast

# Configuration
MAX_SEQ_LENGTH = 2048
DTYPE = None  # None for auto detection
LOAD_IN_4BIT = True
TOKEN = "YOUR_HF_TOKEN_HERE"  # Replace with your actual token

# Model selection
MODEL_NAME = "unsloth/mistral-7b-bnb-4bit"

# Initialize model and tokenizer
def initialize_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        token=TOKEN,
    )
    return model, tokenizer

# Configure PEFT model
def configure_peft_model(model):
    return FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

# Alpaca prompt template
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Formatting function for prompts
def formatting_prompts_func(examples):
    texts = []
    for instruction, input, output in zip(examples["instructions"], examples["input"], examples["output"]):
        text = ALPACA_PROMPT.format(instruction, input, output) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

# Data loading and preprocessing
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['evidence_top5'] = data['evidence_top5'].apply(convert_to_list)
    data['Evidence_List'] = data['Evidence_List'].apply(convert_to_list)
    return data

def convert_to_list(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value

# Transform data for model input
def transform_data(table):
    table["input"] = table.apply(lambda row: f"###Context: {'. '.join(row['evidence_top5'])} ###Sentence: {row['Statement']}", axis=1)
    table["output"] = table["labels"].astype(str)
    table["instructions"] = """You will be presented with a long context, followed by a statement. Your task is to fact-check the statement based on the provided context. You must categorize the statement into one of three categories: Supported(0): Choose this if the statement is true and fully supported by the context. Refuted(1): Choose this if the statement is false and contradicted by the context. Not Enough Information(2): Choose this if the statement contains content that is not covered by the context, making it impossible to determine its accuracy. Only output the code of the label, do not generate irrelevant content."""
    return table[['instructions', 'input', 'output']]

if __name__ == "__main__":
    # Initialize model and tokenizer
    model, tokenizer = initialize_model()
    model = configure_peft_model(model)

    # Load and preprocess data
    train_data = load_and_preprocess_data("./train_sbert.csv")
    dev_data = load_and_preprocess_data("./dev_sbert.csv")
    test_data = load_and_preprocess_data("./test_sbert.csv")

    # Transform data
    train = transform_data(train_data)
    dev = transform_data(dev_data)
    test = transform_data(test_data)

    # Convert to Dataset objects
    train_dataset = Dataset.from_pandas(train, split='train')
    dev_dataset = Dataset.from_pandas(dev, split='dev')
    test_dataset = Dataset.from_pandas(test, split='test')

    # Prepare dataset for training
    dataset = train_dataset.map(formatting_prompts_func, batched=True)

    # Configure trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=8,
            gradient_accumulation_steps=8,
            warmup_steps=5,
            num_train_epochs=5,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )

    trainer_stats = trainer.train()
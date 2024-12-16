# Script to load and tokenize dataset

from datasets import load_dataset
from utils import DATASET_SEED, DATASET_NAME


def get_train_dataset():
    dataset = load_dataset(DATASET_NAME, split="train").shuffle(seed=DATASET_SEED)
    return dataset


def get_val_dataset():
    dataset = load_dataset(DATASET_NAME, split="validation").shuffle(seed=DATASET_SEED)
    return dataset


def get_test_dataset():
    dataset = load_dataset(DATASET_NAME, split="test").shuffle(seed=DATASET_SEED)
    return dataset


def tokenize_function(example, tokenizer):
    prompt = [
        f"Summarize the following conversation.\n\n{dialogue}\n\nSummary: "
        for dialogue in example["dialogue"]
    ]

    example["input_ids"] = tokenizer(
        prompt, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids
    example["labels"] = tokenizer(
        example["summary"], padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids

    return example


def tokenize_dataset(dataset, tokenizer):
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    # Remove useless columns
    tokenized_dataset = tokenized_dataset.remove_columns(
        [
            "id",
            "topic",
            "dialogue",
            "summary",
        ]
    )

    return tokenized_dataset

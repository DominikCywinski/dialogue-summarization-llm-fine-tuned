import torch
import os

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import PeftModel
from utils import BASE_MODEL_NAME, BEST_PEFT_MODEL_NAME


def load_base_model(model_name: str = BASE_MODEL_NAME):
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    )

    return base_model


def load_peft_model(
    base_model_name: str = BASE_MODEL_NAME, peft_model_path: str = BEST_PEFT_MODEL_NAME
):
    base_model = load_base_model(base_model_name)
    peft_model = PeftModel.from_pretrained(base_model, peft_model_path)

    return peft_model


def load_quantized_model(model_name: str = BASE_MODEL_NAME):
    # Set bitsandbytes config
    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    # Load base model
    device_map = {"": 0}
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    return model


def load_tokenizer(model_name: str = BASE_MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return tokenizer


def load_model_with_tokenizer(model_name: str = BASE_MODEL_NAME):
    model = load_base_model(model_name)
    tokenizer = load_tokenizer(model_name)

    return model, tokenizer


def load_peft_with_tokenizer(
    base_model_name: str = BASE_MODEL_NAME, peft_model_path: str = BEST_PEFT_MODEL_NAME
):
    model = load_peft_model(base_model_name, peft_model_path)
    tokenizer = load_tokenizer(base_model_name)

    return model, tokenizer


# Load models from models directory
def get_models(path: str = "models"):
    models = []
    for model in os.listdir(path):
        models.append(os.path.join(path, model))

    return models


def create_prompt(dialogue):
    prompt = f"""
    Summarize the following conversation.

    {dialogue}

    Summary: """

    return prompt


def generate_model_output(dialogue, model, tokenizer):
    prompt = create_prompt(dialogue)

    input_ids = tokenizer(
        prompt, truncation=True, padding=True, return_tensors="pt"
    ).input_ids

    model_output = model.generate(
        input_ids=input_ids,
        generation_config=GenerationConfig(max_new_tokens=200, num_beams=1),
    )
    model_output = tokenizer.decode(model_output[0], skip_special_tokens=True)

    return model_output

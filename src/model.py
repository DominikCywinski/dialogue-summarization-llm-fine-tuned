import torch
import os
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from utils import MODEL_NAME


def load_base_model(model_name: str = MODEL_NAME):
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    )

    return base_model


def load_quantized_model(model_name: str = MODEL_NAME):
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


def load_tokenizer(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return tokenizer


# Load models from models directory
def get_models(path: str = "models"):
    models = []
    for model in os.listdir(path):
        models.append(os.path.join(path, model))

    return models

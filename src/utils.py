DATASET_NAME = "knkarthick/dialogsum"
DATASET_SEED = 42
BASE_MODEL_NAME = "google/flan-t5-base"
BEST_PEFT_MODEL_NAME = (
    "models/peft-dialogue-summarizer-epochs-1-lr-1e-04-lora-r-32-lora-alpha-32"
)
dash_line = "-".join("" for x in range(100))

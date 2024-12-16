from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from itertools import product

from dataset import get_train_dataset, tokenize_dataset
from model import load_base_model, load_tokenizer


# Train model with provided hyperparameters
def train_with_params(
    base_model,
    tokenizer,
    tokenized_dataset,
    lora_r=32,
    lora_alpha=32,
    learning_rate=1e-3,
    num_train_epochs=1,
):
    new_model_name = f"peft-dialogue-summarizer-epochs-{num_train_epochs}-lr-{learning_rate}-lora-r-{lora_r}-lora-alpha-{lora_alpha}"
    print(f"Start Training with {new_model_name}....")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    peft_model = get_peft_model(base_model, lora_config)

    peft_training_args = TrainingArguments(
        output_dir=f"./peft_training/{new_model_name}",
        auto_find_batch_size=True,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_steps=10,
        max_steps=-1,
        report_to="tensorboard",
    )

    peft_trainer = Trainer(
        model=peft_model,
        args=peft_training_args,
        train_dataset=tokenized_dataset,
    )

    peft_model_path = f"./models/peft-dialogue-summarizer-epochs-{num_train_epochs}-lr-{learning_rate}-lora-r-{lora_r}-lora-alpha-{lora_alpha}"

    peft_trainer.train()

    peft_trainer.model.save_pretrained(peft_model_path)
    tokenizer.save_pretrained(peft_model_path)


# Train view models with provided sets of hyperparameters
def train_with_params_lists(
    lora_r_vals, lora_alpha_vals, learning_rate_vals, num_train_epochs_vals
):
    base_model = load_base_model()
    tokenizer = load_tokenizer()
    train_dataset = get_train_dataset()
    tokenized_dataset = tokenize_dataset(train_dataset, tokenizer)
    hyperparams_combinations = list(
        product(lora_r_vals, lora_alpha_vals, learning_rate_vals, num_train_epochs_vals)
    )

    for lora_r, lora_alpha, learning_rate, num_train_epochs in hyperparams_combinations:
        train_with_params(
            base_model=base_model,
            tokenizer=tokenizer,
            tokenized_dataset=tokenized_dataset,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
        )


def main():
    lora_r_list = [16, 32]
    lora_alpha_list = [16, 32]
    learning_rate_list = [1e-4, 2e-5, 1e-5]
    num_train_epochs_list = [1, 2]

    train_with_params_lists(
        lora_r_list, lora_alpha_list, learning_rate_list, num_train_epochs_list
    )


if __name__ == "__main__":
    main()

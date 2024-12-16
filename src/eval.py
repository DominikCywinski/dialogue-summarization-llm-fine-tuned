import evaluate
from peft import PeftModel, PeftConfig
from dataset import get_test_dataset, get_val_dataset
from model import (
    load_model_with_tokenizer,
    generate_model_output,
    get_models,
    get_peft_model,
)
from utils import dash_line, BASE_MODEL_NAME


# Generate model outputs for n val dataset samples
def generate_model_outputs(model, tokenizer, dataset):
    model_outputs = []

    for index in range(len(dataset)):
        dialogue = dataset[index]["dialogue"]

        model_output = generate_model_output(dialogue, model, tokenizer)
        model_outputs.append(model_output)

    return model_outputs


def evaluate_rouge(model_outputs, val_outputs):
    rouge = evaluate.load("rouge")

    rouge_score = rouge.compute(
        predictions=model_outputs,
        references=val_outputs,
        use_aggregator=True,
        use_stemmer=True,
    )

    return rouge_score


def evaluate_model(model, tokenizer, dataset):
    val_outputs = generate_model_outputs(model, tokenizer, dataset)
    rouge_score = evaluate_rouge(val_outputs, dataset["summary"])

    print("Original Model")
    print(rouge_score)
    print(dash_line)


def main():
    num_samples = 10
    val_dataset = get_test_dataset().select(range(num_samples))
    base_model, tokenizer = load_model_with_tokenizer()

    # Eval base model
    evaluate_model(base_model, tokenizer, val_dataset)

    models = get_models()

    for model_path in models:
        print(model_path)

        peft_model = get_peft_model(BASE_MODEL_NAME, model_path)
        val_outputs = generate_model_outputs(peft_model, tokenizer, val_dataset)
        rouge_score = evaluate_rouge(val_outputs, val_dataset["summary"])

        print(rouge_score)
        print(dash_line)


if __name__ == "__main__":
    main()

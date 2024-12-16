# Generative AI App: Dialogue Summarization

## Application Description

This application leverages the **fine-tuned Flan-T5 Base model** for generating concise and accurate summaries of dialogue-based texts. 
By integrating **LoRA (Low-Rank Adaptation)** and **PEFT (Parameter-Efficient Fine-Tuning)**, the app offers state-of-the-art performance 
while remaining computationally efficient.

## Key features of the application:

- Accurate Summarization: Tailored specifically for dialogue-heavy texts, such as transcripts, chat logs, or interviews.
- Efficiency with LoRA and PEFT: Uses parameter-efficient fine-tuning methods to achieve high performance with minimal computational resources.
- Scalability: Ideal for deployment on cloud platforms or local environments, ensuring cost-effectiveness for large-scale summarization tasks.
- Customizable Fine-Tuning: Fine-tuned on domain-specific datasets to enhance relevance and context understanding.
- Interactive User Interface with Streamlit

## Technologies

- **Flan-T5 Base model** fine-tuned on "knkarthick/dialogsum" dataset.
- **Streamlit** for creating a user-friendly web interface.
- **TensorBoard** for monitoring training metrics and performance.

## Use Cases:
- Summarizing chat logs for customer support insights.
- Creating concise meeting notes from long conversation transcripts.
- Generating quick summaries for interviews or legal transcripts.

---

## Setup and Installation

### 1. Clone the repository (skip if you already have it locally):

```bash
git clone https://github.com/DominikCywinski/dialogue-summarization-llm-lora-peft.git
cd dialogue-summarization-llm-lora-peft
```

### 2. If you don’t have conda installed, download and install Miniconda or Anaconda.

### 3. Set Up a Virtual Environment

```bash
conda env create -f environment.yml
```

## Run the Application locally

### 1. Activate conda env:

```bash
conda activate genai-env
```

### 2. Start a local web server:

```bash
streamlit run app.py
```

### 3. The application will be available at the local address provided by Streamlit (usually http://localhost:8501).

### 4. Provide a dialogue i.e.:

```bash
#Person1#: I am confused by what he said. 
#Person2#: Why do you say that? 
#Person1#: I don't know what he wants to do. Does he want help me or just scold me? 
#Person2#: Think a little. I think he means well at the bottom of his heart.
```

## License

This project is licensed under the MIT License.
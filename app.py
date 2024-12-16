### script responsible for running the web application ###

import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from model import load_peft_with_tokenizer, generate_model_output
from web_layout import create_layout


def show_results(response):
    st.subheader("Summary:")
    st.header(response)
    print("Summary: ", response)


# Load model in cache
@st.cache_resource
def load_model():
    model, tokenizer = load_peft_with_tokenizer()
    print("Model Loaded")

    return model, tokenizer


def process_user_input(_user_input, _peft_model, _tokenizer):
    return generate_model_output(_user_input, _peft_model, _tokenizer)


# Create page layout
user_input, submit_clicked = create_layout()

peft_model, tokenizer = load_model()

if submit_clicked:
    if user_input != "":
        try:
            # Process user input or get from cache
            print(user_input)
            summarization = process_user_input(user_input, peft_model, tokenizer)
            # print result
            show_results(summarization)

        except Exception as e:
            st.header("Sorry, something went wrong :(. Please try again.")
            print(e)
    else:
        st.header("Please enter a dialogue.")

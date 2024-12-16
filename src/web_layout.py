### Layout of web application ###

import streamlit as st


def create_layout():
    st.set_page_config(page_title="Summarizer App", page_icon="🤖")

    st.markdown(
        '<h1 style="text-align: center;">Dialogue Summarizer App</h1>',
        unsafe_allow_html=True,
    )

    user_input = st.text_input(
        "Input: ",
        key="input",
        placeholder="Enter full dialogue here",
    )

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        submit_clicked = st.button("Summarize")

    return user_input, submit_clicked

import streamlit as st
from inference import translate  # your function

st.title("English → French Translator")

text = st.text_area("Enter English text:")
if st.button("Translate"):
    translation = translate(text)
    st.write("French translation:", translation)
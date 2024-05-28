import streamlit as st
from summary import recursive_summarize
import re

def text_preprocessing(txt):
    txt = re.sub(r'[^\x00-\x7F]+', ' ', txt)
    # txt = re.sub(r'[^\w\d\s]+', '', txt)
    txt = re.sub(' +', ' ', txt)
    return txt

def take_first_4000_words(input_string):
    # Split the string into words
    words = input_string.split()

    # Take the first 4000 words
    selected_words = words[:3000]

    # Join the selected words back into a string
    result_string = ' '.join(selected_words)

    return result_string

def generate_summary(text):
    input_txt = text_preprocessing(text)
    final_txt = take_first_4000_words(input_txt)
    summary = recursive_summarize(final_txt)  # You can adjust the number of sentences in the summary
    return summary

def main():
    st.title("Privacy Policy Summarization App")

    uploaded_file = st.file_uploader("Choose a text file", type="txt")

    if uploaded_file is not None:
        text = uploaded_file.read().decode('utf-8')

        if st.button("Summarize(Click once and wait)"):
            summary = generate_summary(text)

            st.subheader("Original Text:")
            st.write(text)

            st.subheader("Summary:")
            st.write(summary)

            st.markdown("""
            #### Download Summary as Text File
            Click the button below to download the summary as a text file.
            """)

            st.download_button(
                label="Download Summary",
                data=summary,
                file_name="summary.txt",
                mime="text/plain"
            )


if __name__ == "__main__":
    main()

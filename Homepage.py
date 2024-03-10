import streamlit as st

st.set_page_config(
    page_title="Multipage App",
    page_icon="",
)

st.title("Multipage App")
st.markdown("""
Here is a guide to the page setup: adjacent to the page setup is the multi-page feature.

On the second page, students are granted the flexibility to upload any number of PDFs. However, please note that the model will only respond to the content that has already been uploaded.

The third page facilitates interaction with the pre-imported knowledge base. The contents of the imported knowledge base include HTML, CSS, and JavaScript.
""")

st.header('knowledge base')
st.markdown('[knowledge base PDF]https://drive.google.com/drive/folders/1S2Va_4vTLB1uZ9NFZs_QhlSd9fGk1Mig?usp=sharing')

# https://drive.google.com/drive/folders/1S2Va_4vTLB1uZ9NFZs_QhlSd9fGk1Mig?usp=sharing
st.sidebar.success("Select a page above.")
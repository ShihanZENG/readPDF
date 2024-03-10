import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Qdrant
import qdrant_client
from htmlTemplates import css, bot_template, user_template
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)  # 更新为使用PdfReader
        for page in pdf_reader.pages:
            text += page.extract_text() if page.extract_text() else ''
    return text
def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)  # 假设正确的方法是 split_text 而不是 splitter.split_text
    return chunks

def get_vectorstore(text_chunks, use_qdrant=True):
    embeddings = OpenAIEmbeddings()
    if use_qdrant:
        # 如果使用Qdrant，确保已经设置了环境变量
        client = qdrant_client.QdrantClient(
            os.getenv("QDRANT_HOST"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        vectorstore = Qdrant(
            client=client,
            collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
            embeddings=embeddings,
        )
    else:
        # 如果不使用Qdrant，则使用FAISS
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vectorstore
# def get_vectorstore(text_chunks):
#     embeddings = OpenAIEmbeddings()
#     # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore
#
# # model = INSTRUCTOR('hkunlp/instructor-large')

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i,message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    # st.write(response)
def main():
    load_dotenv()
    st.set_page_config(
        page_title="Chat with multiple PDFs",
        page_icon=":books"
    )
    st.write(css, unsafe_allow_html=True)

    if"conversation" not in st.session_state:
        st.session_state.conversation = None
    if"chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :book:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your document")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:  # 检查是否有文件被上传
                with st.spinner("Processing"):
                    # get pdf text
                    raw_text = get_pdf_text(pdf_docs)
                    #get the text chunks
                    text_chunks = get_text_chunks(raw_text)
                    # st.write(raw_text)  # 显示处理后的文本
                    st.write(text_chunks)
                    #create vector store
                    vectorstore = get_vectorstore(text_chunks)
                    #create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)
            else:
                st.warning("Please upload at least one PDF file.")  # 如果没有上传文件则显示警告

if __name__ == '__main__':
    main()
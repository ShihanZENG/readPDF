import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Qdrant
import qdrant_client
from htmlTemplates import css, bot_template, user_template
from langchain.prompts import PromptTemplate

# 定义用于生成问题回答的模板
template_v1 = PromptTemplate(
    template="""Answer the questions based on the following document context. 
    Please quote as much of the document as possible and make sure your answer is concise. 
    Question: {question}""",
    input_variables=["context", "question"]
)

def get_pdf_text(pdf_docs):
    # 从多个PDF文件中提取文本
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + ' '
    return text.strip()

def get_text_chunks(raw_text):
    # 将长文本分割成适合处理的大小
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(raw_text)

def get_vectorstore(text_chunks, use_qdrant=True):
    # 根据文本块创建或更新向量存储
    embeddings = OpenAIEmbeddings()
    if use_qdrant:
        client = qdrant_client.QdrantClient(
            os.getenv("QDRANT_HOST"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        vectorstore = Qdrant(
            client=client,
            collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
            embeddings=embeddings,
        )
    return vectorstore

def get_conversation_chain(vectorstore):
    # 创建一个会话链，结合LLM和检索器
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question, pdf_text):
    # 处理用户输入，并通过Streamlit显示回答
    full_prompt = template_v1.format(context=pdf_text, question=user_question)  # 使用文档内容和用户问题生成完整提示
    response = st.session_state.conversation({'question': full_prompt})
    st.session_state.chat_history = response['chat_history']

    # 输出聊天历史
    for i, message in enumerate(st.session_state.chat_history):
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    st.sidebar.header("Upload your PDFs")
    pdf_docs = st.sidebar.file_uploader("Upload your PDFs here", accept_multiple_files=True)
    if pdf_docs:
        pdf_text = get_pdf_text(pdf_docs)  # Get all the text content of the PDF, but do not display the
        st.header("Chat with multiple PDFs :book:")
        user_question = st.text_input("Ask a question about your documents:", key="user_question")
        if user_question:
            handle_userinput(user_question, pdf_text)
        if st.sidebar.button("Process", key="process_button"):
            with st.spinner("Processing..."):
                text_chunks = get_text_chunks(pdf_text)
                st.sidebar.write(text_chunks)
                vectorstore = get_vectorstore(text_chunks)
                st.success("You can ask questions now.")
                st.session_state.conversation = get_conversation_chain(vectorstore)
    else:
        st.warning("Please upload at least one PDF file.")

if __name__ == '__main__':
    main()

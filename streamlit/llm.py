from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

if "database" not in st.session_state:
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    upstage_api_key = os.getenv("UPSTAGE_API_KEY")
    embedding = UpstageEmbeddings(model="solar-embedding-1-large")
    index_name = 'table-markdown-index'
    # Split documents into smaller chunks
    loader = Docx2txtLoader("../tax_with_markdown.docx")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    document_list = loader.load_and_split(text_splitter=text_splitter)
    chunked_documents = text_splitter.split_documents(document_list)
    # Initialize the PineconeVectorStore
    database = PineconeVectorStore.from_documents(
        documents=[],  # Start with an empty list
        embedding=embedding,
        index_name=index_name
    )
    # Upload documents in batches
    batch_size = 100
    for i in range(0, len(chunked_documents), batch_size):
        print(f'index: {i}, batch size: {batch_size}')
        batch = chunked_documents[i:i + batch_size]
        database.add_documents(batch)  # Add documents to the existing database

    st.session_state.database = database
    st.session_state.llm = ChatUpstage()

database = st.session_state.database
llm = st.session_state.llm

def get_ai_message(user_message):
    prompt = hub.pull("rlm/rag-prompt")
    retriever = database.as_retriever(search_kwargs={"k": 4})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    prompt = ChatPromptTemplate.from_messages(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 마세요.
        그런 경우에는 질문만 리턴해주세요.

        사전: {dictionary}
        사용자의 질문: {user_message}
    """)
    dictionary_chain = prompt | llm | StrOutputParser()
    tax_chain = {"query": dictionary_chain} | qa_chain
    ai_message = tax_chain.invoke({"question": user_message})
    return ai_message["result"]
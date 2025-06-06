from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains import RetrievalQA, create_history_aware_retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
import os
from dotenv import load_dotenv

store = {}
database = None  # Global variable to store the database instance
llm = None

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

@st.cache_resource
def initialize_database():
    embedding = UpstageEmbeddings(model="solar-embedding-1-large")
    index_name = 'table-markdown-index'
    loader = Docx2txtLoader("../tax_with_markdown.docx")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    document_list = loader.load_and_split(text_splitter=text_splitter)
    chunked_documents = text_splitter.split_documents(document_list)

    database = PineconeVectorStore.from_documents(
        documents=[],  # Start with an empty list
        embedding=embedding,
        index_name=index_name
    )

    batch_size = 100
    for i in range(0, len(chunked_documents), batch_size):
        batch = chunked_documents[i:i + batch_size]
        database.add_documents(batch)
    return database

@st.cache_resource
def initialize_llm():
    return ChatUpstage()

# Ensure initialize_database() runs only once
if database is None:
    database = initialize_database()

if llm is None:
    llm = initialize_llm()

def get_retriever():
    retriever = database.as_retriever(search_kwargs={"k": 4})
    return retriever

def get_dictionary_chain(user_message):
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    prompt = ChatPromptTemplate.from_messages(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 마세요.
        그런 경우에는 질문만 리턴해주세요.

        사전: {dictionary}
        사용자의 질문: {user_message}
    """)
    dictionary_chain = prompt | llm | StrOutputParser()
    return dictionary_chain

def get_rag_chain():
    retriever = get_retriever()
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    system_prompt = (
        "You are a helpful assistant that answers questions about income tax "
        "based on the provided context. If the question is not answerable "
        "with the given context, respond with 'I don't know'."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), MessagesPlaceholder("context"), ("human", "{question}")]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_message_key="input",
        history_message_key="chat_history",
        output_message_key="answer",
    )
    return conversational_rag_chain

def get_ai_response(user_message):
    dictionary_chain = get_dictionary_chain(user_message)
    rag_chain = get_rag_chain()
    tax_chain = {"input": dictionary_chain} | rag_chain
    try:
        # Invoke the chain with the correct input format
        ai_response = tax_chain.invoke(
            {"question": user_message}, config={"configurable": {"session_id": "abc123"}}
        )
        return ai_response # Extract the answer from the response
    except ValueError as e:
        print(f"Error invoking tax_chain: {e}")
        return "An error occurred while processing your request."
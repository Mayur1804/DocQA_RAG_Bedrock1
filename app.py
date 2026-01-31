import json
import os
import sys
import boto3
import streamlit as st


# Titan embedding is used for creating generate embeddings

from langchain_aws import BedrockEmbeddings
from langchain_aws import ChatBedrock

# Data ingestion

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader


# Vector Embeddings and vector store
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

#Bedrock clients
bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
bedrock_embeddings = BedrockEmbeddings(client=bedrock, model_id="amazon.titan-embed-text-v1")


#Data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader('data')
    documents = loader.load()
    
    # Text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

#Vector Embedding and vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

def get_llama_llm():
    llm = ChatBedrock(client=bedrock, model_id="meta.llama3-8b-instruct-v1:0", model_kwargs={"max_tokens": 512})
    return llm

def get_nova_llm():
    llm = ChatBedrock(client=bedrock, model_id="amazon.nova-lite-v1:0", model_kwargs={"max_tokens": 512})
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question at the end but use atleast summariza with
250 words with detailed explanations. If you don't know the answer, just say you don't know.
Context: {context}
Question: {question}

Assistant:
"""

PROMPT = PromptTemplate( input_variables=["context", "question"], template=prompt_template )

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    answer = qa.invoke(query)
    return answer



def main():
    st.set_page_config("RAG with AWS Bedrock")

    st.header("RAG with AWS Bedrock")
    user_question = st.text_input("Ask a question about your documents")

    if not os.path.exists("data"):
        os.makedirs("data")

    with st.sidebar:
        st.title("Update PDF ")

        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

        if st.button("Upload"):
            if uploaded_file is not None:
                file_path = os.path.join("data", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success("File uploaded successfully")

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Vector store updated successfully")

    if st.button("Llama output"):
        with st.spinner("Processing"):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llama_llm()
            
            response = get_response_llm(llm, faiss_index, user_question)
            st.write(response["result"])
            st.success("Llama output")

    if st.button("Nova output"):
        with st.spinner("Processing"):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_nova_llm()
            
            response = get_response_llm(llm, faiss_index, user_question)
            st.write(response["result"])
            st.success("Nova output")

if __name__ == "__main__":
    main()
# DocQA_RAG_Bedrock

This project is a production-ready Retrieval-Augmented Generation (RAG) application that allows users to chat with PDF documents. Built with Streamlit, LangChain, and AWS Bedrock, it features a multi-model approach allowing users to compare responses from different Large Language Models (LLMs) like Meta Llama 3 and Amazon Nova Lite based on the same document context.

🚀 Overview
The application enables users to upload PDF documents, index them into a local vector store, and perform semantic queries. It leverages Amazon Titan for generating high-quality text embeddings and uses FAISS for efficient similarity searching.

🛠️ Tech Stack

Frontend: Streamlit 


LLM Orchestration: LangChain (LangChain-AWS, LangChain-Community) 


Vector Database: FAISS (Local CPU version) 

AI Models (via AWS Bedrock):


Embeddings: Amazon Titan Text Embeddings V1 


LLM 1: Meta Llama 3-8b-instruct 


LLM 2: Amazon Nova Lite 


Infrastructure: Boto3 (AWS SDK) 

🏗️ Project Architecture

Data Ingestion: Loads PDFs from a local directory using PyPDFDirectoryLoader.


Text Splitting: Breaks documents into 10,000-character chunks with a 1,000-character overlap for better context retention.


Vectorization: Converts text chunks into embeddings using amazon.titan-embed-text-v1.


Retrieval: When a user asks a question, the top 3 most relevant document chunks are retrieved from the FAISS index.


Generation: The context and query are passed to the selected LLM (Llama or Nova) using a custom prompt template designed for detailed, 250-word explanations.

⚙️ Setup & Installation
Configure AWS Credentials: Ensure your local machine is configured with AWS credentials that have access to Amazon Bedrock runtime.

Bash
aws configure
Install Dependencies:

Bash
pip install -r requirements.txt
Run the Application:

Bash
streamlit run app.py
📝 Usage

Upload: Use the sidebar to upload a PDF file.


Process: Click "Vectors Update" to ingest the document and build the FAISS index.


Query: Type your question in the main text input field.


Compare: Select either "Llama output" or "Nova output" to generate a response from the respective model.

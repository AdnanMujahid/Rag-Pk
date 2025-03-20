from langchain.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def load_and_process_document(pdf_path):
    # Load document
    loader = PyPDFium2Loader(pdf_path)
    data = loader.load()
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len
    )
    chunks = text_splitter.split_documents(data)
    
    return chunks

def create_vectorstore(chunks, persist_directory=None):
    # Create embeddings
    model_name = "sentence-transformers/all-mpnet-base-V2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Create vector store
    ids = [str(i) for i in range(len(chunks))]
    
    if persist_directory:
        os.makedirs(persist_directory, exist_ok=True)
        vectordb = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings,
            ids=ids,
            persist_directory=persist_directory
        )
        vectordb.persist()
    else:
        vectordb = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings,
            ids=ids
        )
    
    return vectordb
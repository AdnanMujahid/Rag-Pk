import gradio as gr
import pypdfium2
import requests
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from huggingface_hub import login
from langchain.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Authenticate with Hugging Face
login("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")  # Replace with your API Key

# Download PDF from URL
pdf_url = "https://na.gov.pk/uploads/documents/1333523681_951.pdf"
pdf_path = "document.pdf"
response = requests.get(pdf_url)
with open(pdf_path, "wb") as f:
    f.write(response.content)

# Load and Process PDF
def load_and_process_pdf(file_path):
    loader = PyPDFium2Loader(file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(data)
    return chunks

chunks = load_and_process_pdf(pdf_path)

# Load Mistral-7B Model
def llm():
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True, device_map="auto")
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)
    return HuggingFacePipeline(pipeline=pipe)

# Load Vector Database
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectordb = Chroma.from_documents(chunks, embeddings_model)
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

# Set Up RAG
rag_chain = RetrievalQA.from_chain_type(
    llm=llm(),
    chain_type="map_reduce",
    retriever=retriever
)

# Gradio Interface
def query_system(query):
    if not query:
        return "Error: Query is required."
    
    greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    if query.lower() in greetings:
        return "Hello! How can I assist you today?"
    
    response = rag_chain.invoke(query)
    return response.strip()[:256]

gui = gr.Interface(
    fn=query_system,
    inputs=gr.Textbox(label="Enter your query"),
    outputs=gr.Textbox(label="Response"),
    title="Legal Case Assistance Chatbot",
    description="Ask questions related to legal cases, and get AI-generated responses using RAG."
)

gui.launch(server_name="0.0.0.0", server_port=5000, share=True)

import gradio as gr
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from huggingface_hub import login
from langchain.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ Step 1: Authenticate with Hugging Face
login("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")  # Replace with your API Key

# ✅ Step 2: Load and Process PDF
def load_and_process_pdf(file_path):
    loader = PyPDFium2Loader(file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Smaller chunks for better relevance
        chunk_overlap=50  # More overlap for continuity
    )
    chunks = text_splitter.split_documents(data)
    return chunks

chunks = load_and_process_pdf("adn.pdf")

# ✅ Step 3: Load Mistral-7B Model
def llm():
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True, device_map="auto")
    
    # ✅ Changed to "text2text-generation" for structured responses
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)  # ✅ Reduced max_length for concise answers
    
    return HuggingFacePipeline(pipeline=pipe)

# ✅ Step 4: Load Vector Database
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")  # Better embeddings
vectordb = ChroAma.from_documents(chunks, embeddings_model)
retriever = vectordb.as_retriever(search_kwargs={"k": 2})  # Fetch fewer chunks for precise answers

# ✅ Step 5: Set Up RAG
rag_chain = RetrievalQA.from_chain_type(
    llm=llm(),
    chain_type="map_reduce",  # ✅ Changed to "map_reduce" for better response filtering
    retriever=retriever
)

# ✅ Gradio Interface
def query_system(query):
    if not query:
        return "Error: Query is required."
    
    # Handle greetings
    greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    if query.lower() in greetings:
        return "Hello! How can I assist you today?"
    
    response = rag_chain.invoke(query)
    return response.strip()[:256]  # ✅ Limit response length to 256 characters

gui = gr.Interface(
    fn=query_system,
    inputs=gr.Textbox(label="Enter your query"),
    outputs=gr.Textbox(label="Response"),
    title="Legal Case Assistance Chatbot",
    description="Ask questions related to legal cases, and get AI-generated responses using RAG."
)

gui.launch(server_name="0.0.0.0", server_port=5000, share=True)

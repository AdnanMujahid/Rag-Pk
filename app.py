import os
import gradio as gr
import requests
from huggingface_hub import login as hf_login
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ✅ Load API Keys securely
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

# ✅ Authenticate with Hugging Face
hf_login(HF_API_KEY)

# ✅ Load and chunk text file
def load_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents([Document(page_content=text)])
    print(f"✅ Loaded {len(docs)} chunks.")
    return docs

chunks = load_text_file("constitution.txt")

# ✅ Embed and store chunks
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
retriever = vectordb.as_retriever(search_kwargs={"k": 1})

# ✅ Gemini API call
def call_gemini_flash(prompt: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return f"❌ Gemini Error: {str(e)}"

# ✅ Query logic
def query_system(query):
    if not query:
        return "Please enter a question."
    try:
        docs = retriever.get_relevant_documents(query)
        context = docs[0].page_content if docs else ""
        full_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        return call_gemini_flash(full_prompt)
    except Exception as e:
        return f"❌ Retrieval Error: {str(e)}"

# ✅ Gradio UI
gui = gr.Interface(
    fn=query_system,
    inputs=gr.Textbox(label="Ask something from the text file"),
    outputs=gr.Textbox(label="Gemini 2.0 Flash Response"),
    title="Fast Text QA with Gemini 2.0 Flash",
    description="Ask questions from the uploaded constitution text using Gemini Flash and Chroma."
)

gui.launch(server_name="0.0.0.0", server_port=7860)

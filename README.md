# Pakistan Constitution RAG System

This application uses Retrieval-Augmented Generation (RAG) to answer questions about Pakistan's constitution. It combines a Mistral-7B language model with a document retrieval system.

## Features

- Question answering about Pakistan's constitution
- Multilingual support (try asking in Urdu, English, etc.)
- Based on the official constitution document

## How It Works

1. The application loads the Pakistan constitution document
2. It splits the document into manageable chunks
3. These chunks are converted into embeddings using a sentence transformer
4. When you ask a question, the system:
   - Finds relevant sections in the document
   - Uses these sections to generate an answer with Mistral-7B

## Example Questions

- "What are the fundamental rights in the constitution?"
- "How is the Prime Minister elected?"
- "پاکستان کی تاریخ کے بارے میں بتائیں۔"
- "What is the role of the Supreme Court?"

## Technical Details

Built using:

- LangChain for the RAG pipeline
- Hugging Face models (Mistral-7B, sentence-transformers)
- Chroma vector database
- Gradio for the user interface

# LlamaParse Chatbot with Langchain

This project uses a combination of `LlamaParse` for document parsing, `Langchain` for conversational retrieval, and `Groq` for a conversational AI. The goal is to parse a PDF document, build a vector store from the extracted text, and integrate the system into a chatbot that can retrieve information based on the parsed content.

## Project Structure

- **llamaParse.py**: Initializes the LlamaParse API, extracts text from documents, and saves the text for further processing.
- **vector_stores.py**: Builds a vector store using `OllamaEmbeddings` and `Chroma` from the extracted text.
- **extractTextVector.py**: Checks if the vector store exists; if not, it extracts text and builds the vector store.
- **groqChat.py**: Handles chatbot interaction using `Chainlit`, initializes vector stores, processes messages, and uses `Langchain` to retrieve and respond based on the vector store data.

## Requirements

- Python 3.8+
- Install dependencies via `pip`:

  ```bash
  pip install -r requirements.txt

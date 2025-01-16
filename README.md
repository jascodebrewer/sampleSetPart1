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

# Setup

## Step 1: Install the Required Packages

Create a virtual environment and install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt

## Step 2: Setup the `.env` File

Create a `.env` file in the project root directory with the following content:

```makefile
LLAMA_API_KEY=your_llama_api_key
GROQ_API_KEY=your_groq_api_key
```
Replace your_llama_api_key and your_groq_api_key with your actual API keys.

## Step 3: Extract Text and Build Vector Store

Before running the chatbot, you need to extract text from a PDF and build a vector store:

1. Place your PDF (e.g., `statement.pdf`) in the project directory.
2. Run the `extractTextVector.py` script:

```bash
   python extractTextVector.py
```
This will extract text from the PDF and build a vector store.

## Step 4: Start the Chatbot

Run the chatbot with Chainlit:

```bash
chainlit run groqChat.py
```

This will start the chatbot that you can interact with. You can query it with questions based on the extracted document.
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

VECTOR_STORE_FOLDER = 'vector_stores'
file_name='statement_extracted.pdf'

def build_vector_store():
    embeddings = OllamaEmbeddings(model='mxbai-embed-large')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)

    # File path of the text file to load
    input_file_name = 'statement_extracted.txt'
    input_file_path = os.path.join(input_file_name)

    # Load the content of the text file
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        pdf_text = input_file.read()

    file_texts = text_splitter.split_text(pdf_text)
    file_metadatas = [{"source": f"{i}-{file_name}"} for i in range(len(file_texts))]
    persist_directory=os.path.join(VECTOR_STORE_FOLDER, f"{file_name}.vector_store")
    
    # Create a vector store and save it
    vector_store = Chroma.from_texts(file_texts, embeddings, metadatas=file_metadatas,  persist_directory=persist_directory)
            
    print(f"Vector store for {file_name} built and saved. {vector_store}")
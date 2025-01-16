from vector_stores import build_vector_store
from llamaParse import initialize_parser
import os

# Constants
FILENAME = 'statement_extracted.txt'
VECTOR_STORE_FOLDER = 'vector_stores'

file_path = os.path.join(FILENAME)
vector_store_path = os.path.join(VECTOR_STORE_FOLDER)

def is_vector_store_built():
    """Check if the vector store directory exists and is not empty."""
    return os.path.exists(vector_store_path) and any(os.scandir(vector_store_path))

if is_vector_store_built():
    print(f"Vector store already built in '{vector_store_path}'.")
else:
    if os.path.exists(file_path):
        print(f"File '{file_path}' found. Building vector store...")
        build_vector_store()
    else:
        print(f"File '{file_path}' not found. Extracting text from the source...")
        parser = initialize_parser()
        try:
            # Load and process the source file
            documents = parser.load_data([file_path])
            pdf_text = "".join(doc.text for doc in documents)

            # Save the extracted text to a new file
            with open(file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(pdf_text)

            print(f"Extracted text saved to '{file_path}'. Building vector store...")
            build_vector_store()
        except Exception as e:
            print(f"An error occurred: {e}")


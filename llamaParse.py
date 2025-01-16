from dotenv import load_dotenv
import os
from llama_parse import LlamaParse

# Load environment variables
load_dotenv()

# Retrieve the API key for LlamaParse
llama_api_key = os.getenv('LLAMA_API_KEY')

# Function to initialize LlamaParse
def initialize_parser():
    return LlamaParse(
        api_key=llama_api_key,
        result_type="markdown",
        verbose=True,
        language="en",
    )


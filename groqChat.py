import chainlit as cl
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
import uuid
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory


VECTOR_STORE_FOLDER = 'vector_stores'
# Initialize the LLM with the API key
groq_api_key = os.getenv('GROQ_API_KEY')
llm_groq = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.2
)

@cl.on_chat_start
async def on_chat_start():
    session_key = f"chain_{uuid.uuid4().hex}"
    cl.user_session.set("current_chain_key", session_key)

    vector_stores = {}
    document_names = []
    try:
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    except Exception as e:
        await cl.Message(content=f"Error initializing embeddings: {str(e)}").send()
        return

    # Load pre-built vector stores
    for file_name in os.listdir(VECTOR_STORE_FOLDER):
        if file_name.endswith('.vector_store'):
            vector_store_path = os.path.join(VECTOR_STORE_FOLDER, file_name)
            try:
                vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
                vector_stores[file_name.replace('.vector_store', '')] = vector_store
                document_names.append(file_name.replace('.vector_store', ''))
            except Exception as e:
                await cl.Message(content=f"Error loading vector store {file_name}: {str(e)}").send()
                return

    if not vector_stores:
        await cl.Message(content="No vector stores found.").send()
        return

    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Save session data
    cl.user_session.set(session_key, {
        "vector_stores": vector_stores,
        "document_names": document_names,
        "memory": memory
    })


@cl.on_message
async def main(message: cl.Message):
    session_key = cl.user_session.get("current_chain_key")
    if session_key is None:
        await cl.Message(content="No active session found. Please start a new session.").send()
        return

    session_data = cl.user_session.get(session_key)
    if not session_data:
        await cl.Message(content="No data found for this session.").send()
        return

    vector_stores = session_data.get('vector_stores', {})
    memory = session_data.get('memory')
    document_names = session_data.get('document_names', [])

    if not vector_stores:
        await cl.Message(content="No vector stores available.").send()
        return

    retrievers = [vector_stores[doc_name].as_retriever() for doc_name in document_names if doc_name in vector_stores]
    if not retrievers:
        await cl.Message(content="No valid retrievers found.").send()
        return

    docsearch = retrievers[0]  

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_groq,
        chain_type="stuff",
        retriever=docsearch,
        memory=memory,
        return_source_documents=True,
    )

    cb = cl.AsyncLangchainCallbackHandler()

    try:
        res = await chain.ainvoke(message.content, callbacks=[cb])
    except Exception as e:
        await cl.Message(content=f"Error during chain invocation: {str(e)}").send()
        return

    answer = res.get("answer", "No answer found.")
    source_documents = res.get("source_documents", [])
    text_elements = []

    # Process source documents if available
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            if hasattr(source_doc, 'page_content'):
                source_name = f"source_{source_idx}"
                text_elements.append(
                    cl.Text(content=source_doc.page_content, name=source_name, display="side")
                )

    if text_elements:
        source_names = [text_el.name for text_el in text_elements]
        answer += f"\nSources: {', '.join(source_names)}"
    else:
        answer += "\nNo sources found."

    await cl.Message(content=answer, elements=text_elements).send()

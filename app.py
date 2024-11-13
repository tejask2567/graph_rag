import streamlit as st
import time
import datetime
import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field
from typing import Tuple, List
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import TokenTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain.graphs.graph_document import GraphDocument, Node, Relationship
import uuid
import easyocr  # Import EasyOCR for image text extraction

st.set_page_config(page_title="Cybersecurity RAG App", layout="wide")

# Create docs directory if it doesn't exist
DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)

# Environment variables setup through Streamlit
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
def add():
    pass
# Sidebar for API keys and credentials
with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("GROQ API Key", type="password")
    neo4j_uri = st.text_input("Neo4j URI", value="bolt://localhost:7687")
    neo4j_username = st.text_input("Neo4j Username", value="neo4j")
    neo4j_password = st.text_input("Neo4j Password", type="password")

    if all([groq_api_key, neo4j_uri, neo4j_username, neo4j_password]):
        os.environ["GROQ_API_KEY"] = groq_api_key
        os.environ["NEO4J_URI"] = neo4j_uri
        os.environ["NEO4J_USERNAME"] = neo4j_username
        os.environ["NEO4J_PASSWORD"] = neo4j_password
        st.session_state.initialized = True

# Initialize models and databases
@st.cache_resource
def init_models():
    if not st.session_state.initialized:
        return None, None
    
    llm = ChatGroq(
        temperature=0,
        model_name="llama-3.1-8b-instant",
    )
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    return llm, embeddings

@st.cache_resource
def init_neo4j():
    if not st.session_state.initialized:
        return None, None
    
    try:
        graph = Neo4jGraph(
            url=os.environ["NEO4J_URI"],
            username=os.environ["NEO4J_USERNAME"],
            password=os.environ["NEO4J_PASSWORD"]
        )
        vector_index = Neo4jVector.from_existing_graph(
            embeddings,
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        return graph, vector_index
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {str(e)}")
        return None, None

if st.session_state.initialized:
    llm, embeddings = init_models()
    graph, vector_index = init_neo4j()
else:
    st.warning("Please provide all required credentials in the sidebar to initialize the application.")
    st.stop()

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Rest of your functions remain the same
def generate_full_text_query(input: str) -> str:
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    if not words:
        return ""
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

def structured_retriever(question: str, entity_chain) -> str:
    """Modified retriever to use the new index."""
    result = ""
    try:
        entities = entity_chain.invoke({"question": question})
        for entity in entities.names:
            query = generate_full_text_query(entity)
            if query:
                response = graph.query(
                    """
                    CALL db.index.fulltext.queryNodes('entitySearch', $query)
                    YIELD node, score
                    WITH node
                    MATCH (node)<-[r:MENTIONS]-(doc:Document)
                    RETURN node.name + ' found in document' AS output
                    LIMIT 5
                    """,
                    {"query": query},
                )
                result += "\n".join([el['output'] for el in response])
    except Exception as e:
        st.error(f"Error in structured retriever: {str(e)}")
    return result

def retriever(question: str, entity_chain):
    structured_data = structured_retriever(question, entity_chain)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    return f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ". join(unstructured_data)}"""

def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

def process_chat(question: str, chat_history: List[Tuple[str, str]], entity_chain):
    
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
    in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""  # noqa: E501
    
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    Use natural language and be concise.
    Answer:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    prompt = ChatPromptTemplate.from_template(template)
    
    _search_query = RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))),
            RunnablePassthrough.assign(
                chat_history=lambda x: _format_chat_history(x["chat_history"])
            )
            | CONDENSE_QUESTION_PROMPT
            | llm
            | StrOutputParser(),
        ),
        RunnableLambda(lambda x: x["question"]),
    )
    
    chain = (
        RunnableParallel(
            {
                "context": _search_query | (lambda x: retriever(x, entity_chain)),
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke(
        {
            "question": question,
            "chat_history": chat_history,
        }
    )

# Main Streamlit interface
st.title("Graph-based RAG for Cybersecurity")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'entity_chain' not in st.session_state:
    entity_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are extracting organization and person entities from the text."),
        ("human", "Use the given format to extract information from the following input: {question}"),
    ])
    st.session_state.entity_chain = entity_prompt 

# File upload in sidebar
uploaded_file = st.sidebar.file_uploader("Upload a text file", type=["txt"])
uploaded_image = st.sidebar.file_uploader("Upload an image for OCR", type=["jpg", "jpeg", "png"])
if uploaded_file or uploaded_image:
    content=""
    if uploaded_file:
        content += uploaded_file.getvalue().decode("utf-8")
    if uploaded_image:
        with st.spinner("Extracting text from the image..."):
            image_path = f"temp_image_{uuid.uuid4().hex}.png"
            with open(image_path, "wb") as img_file:
                img_file.write(uploaded_image.getvalue())
            
            # Extract text using EasyOCR
            extracted_text_list = reader.readtext(image_path, detail=0)
            content = " ".join(extracted_text_list)
    print(content)
    file_name = 'example.txt'
    if not os.path.exists(file_name):
        with open(file_name, 'w') as file:
            file.write(content)
        st.sidebar.success(f"'{uploaded_file}' did not exist and has been created.")
    else:
        with open(file_name, 'a') as file:
            file.write(content)
        st.sidebar.success(f"'{uploaded_file}' already exists. Text has been appended.")

    if st.sidebar.button("Process Document"):
        with st.spinner("Processing document..."):
            success = add()
            if success:
                st.sidebar.success("Document processed successfully!")
            else:
                st.sidebar.error("Failed to process document.")

# Chat interface
st.write("## Chat Interface")
user_input = st.text_input("Ask a question:")

if st.button("Submit") and user_input:
    with st.spinner("Generating response..."):
        try:
            response = process_chat(
                user_input,
                st.session_state.chat_history,
                st.session_state.entity_chain
            )
            st.session_state.chat_history.append((user_input, response))
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            response = "I encountered an error while processing your question. Please try again."

    # Display chat history
    for q, a in st.session_state.chat_history:
        st.write(f"**You:** {q}")
        st.write(f"**Assistant:** {a}")
        st.write("---")
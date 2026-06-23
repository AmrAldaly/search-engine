import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_classic.agents import (
    create_tool_calling_agent,
    AgentExecutor
)
from langchain_core.tools import create_retriever_tool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

st.set_page_config(page_title="LangChain Search + RAG", page_icon="🔎")

# --- Web / reference search tools -------------------------------------------
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250))
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250))
search = DuckDuckGoSearchRun(name="Search")


# --- RAG: build a retriever tool from uploaded PDFs -------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Embedding model is heavy to load; cache it across reruns and sessions."""
    return SentenceTransformerEmbeddings(model_name=SENTENCE_TRANSFORMER_MODEL)


def build_retriever_tool(files):
    """Load, split, embed and index the uploaded PDFs, then expose them as a tool."""
    docs = []
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getvalue())
            tmp_path = tmp.name
        try:
            docs.extend(PyPDFLoader(tmp_path).load())
        finally:
            os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vectordb = Chroma.from_documents(chunks, get_embeddings())
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    return create_retriever_tool(
        retriever,
        name="search_uploaded_pdfs",
        description=(
            "Searches and returns relevant excerpts from the PDF documents the user "
            "uploaded. Use this whenever the question is about the uploaded files."
        ),
    )


st.title("🔎 LangChain - Chat with search & your PDFs")
"""
A LangChain agent with web (DuckDuckGo), Arxiv, Wikipedia, and optional RAG over
PDFs you upload. Agent thoughts/actions stream live via `StreamlitCallbackHandler`.
"""

# --- Sidebar settings -------------------------------------------------------
st.sidebar.title("Settings")
api_key = st.sidebar.text_input(
    "Enter your Groq API Key:", type="password", value=os.getenv("GROQ_API_KEY", "")
)
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs for RAG (optional)", type="pdf", accept_multiple_files=True
)

# Rebuild the retriever tool only when the set of uploaded files changes.
retriever_tool = None
if uploaded_files:
    signature = tuple((f.name, f.size) for f in uploaded_files)
    if st.session_state.get("rag_signature") != signature:
        with st.spinner("Indexing uploaded PDFs..."):
            st.session_state["retriever_tool"] = build_retriever_tool(uploaded_files)
            st.session_state["rag_signature"] = signature
    retriever_tool = st.session_state.get("retriever_tool")
    st.sidebar.success(f"{len(uploaded_files)} PDF(s) indexed for RAG.")

# --- Chat state -------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "Hi, I'm a chatbot who can search the web, Arxiv, Wikipedia, and "
                "any PDFs you upload. How can I help you?"
            ),
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not api_key:
        st.info("Please add your Groq API key in the sidebar to continue.")
        st.stop()

    llm = ChatGroq(groq_api_key=api_key, model_name=LLM_MODEL, streaming=True)

    tools = [search, arxiv, wiki]
    if retriever_tool is not None:
        tools.append(retriever_tool)
    

    agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
    )

    search_agent = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(prompt, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)


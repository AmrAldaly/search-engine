import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain.agents import create_agent

from langchain_core.tools import create_retriever_tool, tool
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

# ---------------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="LangChain Search + RAG", page_icon="🔎")
st.title("🔎 LangChain - Chat with search & your PDFs")
st.markdown(
    "A LangChain agent with web (DuckDuckGo), Arxiv, Wikipedia, and optional RAG over "
    "PDFs you upload.  Agent thoughts stream live via `st.status`."
)


@tool
def web_search(query: str) -> str:
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        return DuckDuckGoSearchRun().run(query)
    except Exception as e:
        return f"[Web search unavailable: {type(e).__name__}: {e}]"


@tool
def arxiv_search(query: str) -> str:
    try:
        from langchain_community.utilities import ArxivAPIWrapper
        return ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250).run(query)
    except Exception as e:
        return f"[Arxiv search unavailable: {type(e).__name__}: {e}]"


@tool
def wikipedia_search(query: str) -> str:
    try:
        from langchain_community.utilities import WikipediaAPIWrapper
        return WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250).run(query)
    except Exception as e:
        return f"[Wikipedia search unavailable: {type(e).__name__}: {e}]"


# ---------------------------------------------------------------------------
# RAG: build a retriever tool from uploaded PDFs
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings() -> SentenceTransformerEmbeddings:
    return SentenceTransformerEmbeddings(model_name=SENTENCE_TRANSFORMER_MODEL)


def build_retriever_tool(files):
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
            "uploaded.  Use this whenever the question is about the uploaded files."
        ),
    )


# ---------------------------------------------------------------------------
# Sidebar settings
# ---------------------------------------------------------------------------
st.sidebar.title("Settings")
api_key = st.sidebar.text_input(
    "Enter your Groq API Key:", type="password", value=os.getenv("GROQ_API_KEY", "")
)
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs for RAG (optional)", type="pdf", accept_multiple_files=True
)

retriever_tool = None
if uploaded_files:
    signature = tuple((f.name, f.size) for f in uploaded_files)
    if st.session_state.get("rag_signature") != signature:
        with st.spinner("Indexing uploaded PDFs…"):
            st.session_state["retriever_tool"] = build_retriever_tool(uploaded_files)
            st.session_state["rag_signature"] = signature
    retriever_tool = st.session_state.get("retriever_tool")
    st.sidebar.success(f"{len(uploaded_files)} PDF(s) indexed for RAG.")


# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "Hi, I'm a chatbot who can search the web, Arxiv, Wikipedia, and "
                "any PDFs you upload.  How can I help you?"
            ),
        }
    ]

# Render existing history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# ---------------------------------------------------------------------------
# Helper: convert session history → LangChain message objects
# ---------------------------------------------------------------------------
def _build_lc_history() -> list:
    lc_messages = []
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))
    return lc_messages


# ---------------------------------------------------------------------------
# Chat input handling
# ---------------------------------------------------------------------------
if user_input := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    if not api_key:
        st.info("Please add your Groq API key in the sidebar to continue.")
        st.stop()

    llm = ChatGroq(groq_api_key=api_key, model_name=LLM_MODEL, streaming=True)

    tools = [web_search, arxiv_search, wikipedia_search]
    if retriever_tool is not None:
        tools.append(retriever_tool)

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful research assistant. "
            "Use the available tools to answer the user's question thoroughly. "
            "If a tool returns an '[...unavailable...]' message, try another tool or "
            "answer from your own knowledge. "
            "If the user has uploaded PDFs, prefer 'search_uploaded_pdfs' for questions "
            "about them; otherwise use web_search, arxiv_search, or wikipedia_search."
        ),
    )

    history = _build_lc_history()
    agent_input = {"messages": history + [HumanMessage(content=user_input)]}

    with st.chat_message("assistant"):
        with st.status("Thinking…", expanded=False) as status:
            for event in agent.stream(agent_input, stream_mode="values"):
                last_msg = event.get("messages", [None])[-1]
                if last_msg is None:
                    continue
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        status.write(f"🔧 Calling tool: **{tc['name']}**")
                        if tc.get("args"):
                            status.write(f"   Args: `{tc['args']}`")
            status.update(label="Done", state="complete", expanded=False)

        result = agent.invoke(agent_input)
        final_message = result["messages"][-1]
        response_text = (
            final_message.content
            if hasattr(final_message, "content")
            else str(final_message)
        )
        st.write(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})

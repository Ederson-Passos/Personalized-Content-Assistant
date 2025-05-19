import streamlit as st
import asyncio
import os
from dotenv import load_dotenv

try:
    from rag_application_main import setup_rag_components_and_tools, logger
    from agents.document_agents import QuerySynthesisAgent
    from tasks.document_tasks import create_query_synthesis_tasks
    from crewai import Crew, Process
except ImportError as e:
    st.error(f"Error importing modules from RAG project: {e}."
             "Check if streamlit_app.py is in the root directory of the project "
             "or if the PYTHONPATH is set correctly.")
    st.stop()

st.set_page_config(
    layout="wide",
    page_title="Personalized Content Assistant (Demo)",
    page_icon="🤖"
)

load_dotenv()
logger.info("Streamlit App: Application started and page settings defined.")

DELL_LIKE_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

    body {
        font-family: 'Roboto', sans-serif;
        background-color: #FFFFFF;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    .app-header {
        text-align: center;
        margin-bottom: 2.5rem;
    }
    .app-header img {
        height: 50px;
        margin-bottom: 10px;
    }
    .app-header h1 {
        color: #0076CE; /* Azul Dell */
        font-weight: 700;
        font-size: 2.2em;
    }
    div[data-testid="stTextInput"] > label {
        font-size: 1.1em;
        font-weight: 500;
        color: #333;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stTextInput"] input {
        border: 1px solid #CCCCCC;
        border-radius: 25px;
        padding: 12px 20px;
        font-size: 1.1em;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: border-color 0.3s, box-shadow 0.3s;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #0076CE;
        box-shadow: 0 0 0 0.2rem rgba(0,118,206,0.25);
    }
    div[data-testid="stButton"] > button {
        background-color: #0076CE;
        color: white;
        border: none;
        padding: 12px 25px;
        border-radius: 25px;
        font-size: 1.1em;
        font-weight: 500;
        cursor: pointer;
        transition: background-color 0.3s ease;
        width: 100%;
    }
    div[data-testid="stButton"] > button:hover {
        background-color: #005ea2;
    }
    .response-window {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 25px;
        margin-top: 2.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .response-window h3 {
        color: #0076CE;
        margin-top: 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #e0e0e0;
    }
    .response-window .final-answer {
        font-size: 1.1em;
        line-height: 1.6;
        margin-bottom: 20px;
        padding: 15px;
        background-color: #fff;
        border-left: 4px solid #0076CE;
        border-radius: 4px;
    }
    .response-window .document-sources h4 {
        color: #333;
        margin-bottom: 15px;
    }
    .document-snippet {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 6px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .document-snippet strong {
        color: #0076CE;
    }
    .document-snippet .content-preview {
        color: #555;
        font-size: 0.95em;
        line-height: 1.5;
        margin-top: 8px;
        max-height: 100px;
        overflow-y: auto;
    }
    .document-snippet a {
        color: #0076CE;
        text-decoration: none;
        font-weight: 500;
    }
    .document-snippet a:hover {
        text-decoration: underline;
    }
</style>
"""
st.markdown(DELL_LIKE_CSS, unsafe_allow_html=True)
logger.info("Streamlit App: customized CSS applied.")

# Session State to persist data between reruns
if 'rag_llm' not in st.session_state:
    st.session_state.rag_llm = None
if 'jec_tool' not in st.session_state:
    st.session_state.jec_tool = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'search_triggered' not in st.session_state:
    st.session_state.search_triggered = False
if 'response_data' not in st.session_state:
    st.session_state.response_data = None

logger.info("Streamlit App: Session state initialized.")

# Auxiliar funcs
@st.cache_resource
def load_rag_pipeline_cached():
    """
    Loads and caches RAG components using the function from the main script.
    Returns:
        Tuple[Optional[LLM], Optional[LlamaIndexQueryTool], bool]: llm, jec_tool, success
    """
    logger.info("Streamlit App: Trying to load RAG components (cached if possible)...")
    llm, tool = setup_rag_components_and_tools()
    if not llm or not tool:
        logger.error("Streamlit App: Critical failure while loading LLM or JEC Tool via "
                     "setup_rag_components_and_tools.")
        return None, None, False
    logger.info("Streamlit App: RAG components loaded and ready via setup_rag_components_and_tools.")
    return llm, tool, True

# Initialization of RAG Components
if not st.session_state.initialized:
    with st.spinner("Initializing the wizard... Please wait. This may take a few moments..."):
        logger.info("Streamlit App: Uninitialized state. Loading RAG pipeline...")
        st.session_state.rag_llm, st.session_state.jec_tool, st.session_state.initialized = load_rag_pipeline_cached()

    if not st.session_state.initialized:
        logger.error("Streamlit App: RAG components fail to initialize after attempting to load.")
        st.error("Failed to initialize wizard components. Check console logs and .env configuration.")
        st.stop()
    else:
        logger.info("Streamlit App: Assistant initialized successfully.")
        st.toast("Assistente pronto para suas perguntas!")

async def process_query_and_get_response_streamlit(user_query: str, llm, jec_tool_instance):
    """
    Processes the user query, executes the RAG pipeline, and returns the response and documents.
    This function adapts the logic of your `run_rag_application` to be called by the UI.
    """
    logger.info(f"Streamlit App: Starting query processing: '{user_query}'")
    final_answer = "Sorry, I couldn't process your question at the moment."
    retrieved_docs_for_display = []

    if not llm or not jec_tool_instance:
        logger.error("Streamlit App: LLM or JEC tool are not available for process_query.")
        return "Error: RAG components not initialized correctly.", []

    try:
        logger.info(f"Streamlit App: Running JEC tool (LlamaIndexQueryTool.run) for query: '{user_query}'")

        if not getattr(jec_tool_instance, 'index', None) or not getattr(jec_tool_instance, 'query_engine', None):
            logger.error("Streamlit App: JEC tool does not appear to be fully initialized (no index or query_engine).")
            return "Error: Document search tool is not ready.", []

        jec_results_raw = await asyncio.to_thread(
            jec_tool_instance.run,
            query=user_query,
            top_k=3
        )
        logger.info(f"Streamlit App: JEC tool returned: {jec_results_raw}")

        if not jec_results_raw or \
                not isinstance(jec_results_raw, list) or \
                (jec_results_raw and not all(isinstance(item, dict) for item in jec_results_raw)):
            logger.warning(f"Streamlit App: JEC tool returned result in unexpected format: {jec_results_raw}")
            final_answer = "Unable to find relevant documents or a search error occurred (unexpected format)."
            jec_results = []
        elif jec_results_raw and isinstance(jec_results_raw[0], dict) and "error" in jec_results_raw[0]:
            logger.error(f"Streamlit App: JEC tool returned an explicit error: {jec_results_raw[0]['error']}")
            final_answer = f"Error searching for documents: {jec_results_raw[0]['error']}"
            jec_results = []
        else:
            jec_results = [doc for doc in jec_results_raw if isinstance(doc, dict) and "text_content" in doc]
            retrieved_docs_for_display = jec_results[:3]
            logger.info(f"Streamlit App: JEC tool processed and filtered {len(jec_results)} valid snippets.")

        if not jec_results:
            logger.warning("Streamlit App: No relevant or usable documents found after JEC search.")
            if final_answer.startswith("Sorry."):
                final_answer = "No relevant documents found for your question."

            return final_answer, retrieved_docs_for_display
        else:
            logger.info(f"Streamlit App: Starting response synthesis with CrewAI using {len(jec_results)} snippets.")

            synthesis_task_instance = create_query_synthesis_tasks(
                llm=llm,
                user_query=user_query,
                document_snippets=jec_results
            )

            synthesis_crew = Crew(
                agents=[synthesis_task_instance.agent],
                tasks=[synthesis_task_instance],
                process=Process.sequential,
                verbose=False
            )

            logger.info("Streamlit App: Kicking off Query Synthesis Crew...")
            synthesis_result = await asyncio.to_thread(synthesis_crew.kickoff)

            final_answer = synthesis_result
            logger.info(f"Streamlit App: Response summarized by CrewAI: {final_answer}")

    except Exception as e_process:
        logger.error(f"Streamlit App: Error processing query '{user_query}': {e_process}", exc_info=True)
        final_answer = f"An unexpected error occurred while processing your question: {e_process}"

    return final_answer, retrieved_docs_for_display

# Interface Layout
st.markdown("""
<div class="app-header">
    <h1>Personalized Content Assistant (Demo)</h1>
    <p style="font-size: 1.1em; color: #555;">Searching based on your search history...</p>
</div>
""", unsafe_allow_html=True)

col_spacer1, col_search, col_button, col_spacer2 = st.columns([0.5, 6, 1.5, 0.5])

with col_search:
    user_query = st.text_input(
        "Your search:",
        placeholder="E.g.: Simulating a research and diagnostic history...",
        label_visibility="collapsed",
        key="query_input"
    )

with col_button:
    if st.button("Search", use_container_width=True, key="search_button"):
        if user_query:
            logger.info(f"Streamlit App: 'Search' button clicked with query: '{user_query}'")
            st.session_state.search_triggered = True
            st.session_state.response_data = None
        else:
            logger.warning("Streamlit App: 'Search' button clicked without query.")
            st.warning("Please enter a question before searching.")
            st.session_state.search_triggered = False

# Processing and Displaying Results
if st.session_state.search_triggered and user_query:
    if not st.session_state.initialized or not st.session_state.rag_llm or not st.session_state.jec_tool:
        st.error("The wizard components are not ready. Try reloading the page.")
        logger.error("Streamlit App: Attempted to fetch but RAG components not initialized in session state.")
    else:
        with st.spinner("Looking for the best answer for you... Please wait..."):
            logger.info(f"Streamlit App: Initiating call to process_query_and_get_response_streamlit with "
                        f"query: '{user_query}'")
            try:
                response_tuple = asyncio.run(
                    process_query_and_get_response_streamlit(
                        user_query,
                        st.session_state.rag_llm,
                        st.session_state.jec_tool
                    )
                )
                st.session_state.response_data = response_tuple
                if response_tuple and isinstance(response_tuple[0], str):
                    logger.info(f"Streamlit App: Response received from process_query: {response_tuple[0][:100]}...")
                elif response_tuple and hasattr(response_tuple[0], 'raw'):
                    logger.info(f"Streamlit App: Response received from "
                                f"process_query: {response_tuple[0].raw[:100]}...")
                elif response_tuple: # Fallback para str()
                    logger.info(f"Streamlit App: Response received from "
                                f"process_query: {str(response_tuple[0])[:100]}...")
            except Exception as e_ui_process:
                logger.error(f"Streamlit App: Error calling process_query_and_get_response_streamlit from "
                             f"UI: {e_ui_process}", exc_info=True)
                st.error(f"An unexpected error occurred during processing: {e_ui_process}")
                st.session_state.response_data = (f"Critical error in processing: {e_ui_process}", [])

    st.session_state.search_triggered = False

if st.session_state.response_data:
    final_answer, retrieved_docs = st.session_state.response_data

    st.markdown('<div class="response-window">', unsafe_allow_html=True)
    st.markdown("<h3>Here's what we found for you:</h3>", unsafe_allow_html=True)

    st.markdown(f'<div class="final-answer">{final_answer}</div>', unsafe_allow_html=True)

    if retrieved_docs:
        st.markdown("<hr style='margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)
        st.markdown("<div class='document-sources'><h4>Sources Consulted:</h4></div>", unsafe_allow_html=True)

        for i, doc in enumerate(retrieved_docs):
            if isinstance(doc, dict):
                content = doc.get('text_content', 'Content not available.')
                score = doc.get('score', 0.0)
                metadata = doc.get('metadata', {})

                doc_title = metadata.get('title', metadata.get('file_name', f"Source {i+1}"))

                source_link_md = f"<strong>{doc_title}</strong> (Relevance: {score:.2f})"

                doc_url = metadata.get('url', metadata.get('source_url'))
                if doc_url:
                    source_link_md = (f'<a href="{doc_url}" target="_blank" title="Open source in new '
                                      f'tab">{doc_title}</a> (Relevance: {score:.2f})')
                elif 'file_path' in metadata and metadata['file_path']:
                    source_link_md = (f"File: <strong>{os.path.basename(metadata['file_path'])}</strong> (Relevance"
                                      f": {score:.2f})")

                st.markdown(f"""
                <div class="document-snippet">
                    <p style="margin-bottom: 5px;">{source_link_md}</p>
                    <div class="content-preview">
                        {content[:350]}...
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                logger.warning(f"Streamlit App: Unexpected format for document {i+1} when displaying: {doc}")
                st.warning(f"Unable to display font {i+1} due to unexpected format.")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align: center; color: #777; font-size: 0.9em;'>Powered by Judicious Evocation "
            "of Content Tool (JEC Tool) | Demo Interface</p>", unsafe_allow_html=True)

logger.info("Streamlit App: Main UI rendering completed.")
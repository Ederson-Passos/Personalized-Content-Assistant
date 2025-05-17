import os
import logging
import chromadb
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from crewai import Agent as CrewAIAgent, Task as CrewAITask, Crew, Process, LLM

from core.Authentication import GoogleDriveAPI
from core.DataBaseManager import DataBaseManager
from core.FolderManager import check_directory_existence, cleanup_temp_folder
from utills.logger import setup_logger

logger = setup_logger(
    logger_name="ingestion_pipeline",
    log_file="logs/ingestion_pipeline.log",
    log_level=logging.INFO
)

class IngestionSummarizerAgent(CrewAIAgent):
    """Agent to generate concise summaries of documents during ingestion."""
    def __init__(self, llm_instance: LLM):
        super().__init__(
            role="Local Efficient Document Summarizer for Ingestion",
            goal="Generate a concise and informative summary of the provided document text, suitable for metadata, "
                 "using a locally running LLM. The summary should be between 3 to 7 sentences long.",
            backstory="I specialize in quickly understanding the essence of a document and producing a brief summary "
                      "to aid in its later discovery and relevance assessment. I focus on factual extraction using"
                      " local LLMs.",
            llm=llm_instance,
            verbose=False,
            memory=False,
            tools=[]
        )

DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID")
CHROMA_PERSIST_PATH = os.getenv("CHROMA_PERSIST_PATH", "./chroma_db_store")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "MyKnowledgeBase")
TEMP_INGESTION_DIR = Path(os.getenv("TEMP_INGESTION_DIR", "temp_ingestion_data"))
EMBEDDING_MODEL_NAME_HF = os.getenv("EMBEDDING_MODEL_NAME_HF", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL_FOR_SUMMARY = os.getenv("OLLAMA_MODEL_FOR_SUMMARY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

def run_ingestion():
    load_dotenv()
    logger.info("[green]==================================================[/green]")
    logger.info("[green]Starting knowledge ingestion pipeline[/green]")
    logger.info("[green]==================================================[/green]")

    if not DRIVE_FOLDER_ID:
        logger.error("[bold red]Missing critical environment variable: DRIVE_FOLDER_ID. "
                     "Please check your .env file. Terminating ingestion.[/bold red]")
        return

    # 0. Ensure the ingest temp directory exists and configure the LangChain Ollama instance for CrewAI agent.
    logger.info(f"[green]Verifying/Creating temporary ingest directory: {TEMP_INGESTION_DIR}[/green]")
    check_directory_existence(TEMP_INGESTION_DIR)

    summarizer_agent = None
    try:
        ollama_model_with_provider = f"ollama/{OLLAMA_MODEL_FOR_SUMMARY}"
        logger.info(f"[green]Initializing CrewAI LLM for summarization agent ({OLLAMA_MODEL_FOR_SUMMARY}) "
                    f"at {OLLAMA_BASE_URL}...[/green]")

        crewai_ollama_llm_for_summaries = LLM(
            model=ollama_model_with_provider,
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,
            model_kwargs={
                "repeat_penalty": 1.15,
                "top_k": 40,
                "top_p": 0.9,
                "system": "You are a specialist in creating concise, clear and factual summaries of texts. "
                          "Your goal is to extract the main points and present them objectively, without adding "
                          "personal opinions or information not contained in the original text. "
                          "The summary should be significantly shorter than the original text and present the "
                          "relevant information for reference within the document.",
                "num_ctx": 32768,
                "timeout": 120
            }
        )
        try:
            crewai_ollama_llm_for_summaries.invoke("Respond with just 'OK' if you are working.")
            logger.info(f"[green]Successfully connected to Ollama and model {ollama_model_with_provider}"
                        f" seems responsive.[/green]")
        except Exception as ollama_test_e:
            logger.warning(f"[bold yellow]Could not confirm Ollama model responsiveness: {ollama_test_e}. "
                           "Proceeding, but summarization might fail if Ollama is not running or the model "
                           "is not pulled.[/bold yellow]")

        summarizer_agent = IngestionSummarizerAgent(llm_instance=crewai_ollama_llm_for_summaries)
        logger.info(f"[green]IngestionSummarizerAgent instantiated with Ollama model: "
                    f"{ollama_model_with_provider}.[/green]")

    except Exception as e:
        logger.error(f"[bold red]Failed to initialize Ollama LLM or Agent for summarization: {e}. "
                     "Document summaries will not be generated. Ensure Ollama is running and the model "
                     f"'{OLLAMA_MODEL_FOR_SUMMARY}' is pulled (e.g., "
                     f"'ollama pull {OLLAMA_MODEL_FOR_SUMMARY}').[/bold red]", exc_info=True)

    # 1. Get documents from Google Drive.
    logger.info("[green]Authenticating with Google Drive...[/green]")
    try:
        drive_api = GoogleDriveAPI()
        drive_service = drive_api.service
        if not drive_service:
            logger.error("[bold red]Failed to get Google Drive service. Terminating ingestion.[/bold red]")
            return
        db_manager = DataBaseManager(drive_service)
        logger.info("[green]Authentication and DataBase initialized.[/green]")
    except Exception as e:
        logger.error(f"[bold red]Critical error initializing Google Drive API or DataBaseManager:"
                     f"[/bold red] {e}",
                     exc_info=True)
        return

    logger.info(f"[green]Listing files from Google Drive ID folder: {DRIVE_FOLDER_ID} [/green]")
    drive_files_metadata = db_manager.list_files_recursively(DRIVE_FOLDER_ID)

    if not drive_files_metadata:
        logger.warning("[yellow]No files for ingestion found in Google Drive.[/yellow]")
        return
    logger.info(f"[green]Found {len(drive_files_metadata)} items (files/folders) for potential download.[/green]")

    downloaded_file_info = []
    for file_meta in drive_files_metadata:
        file_id = file_meta.get('id')
        file_name = file_meta.get('name')
        # Skip folders, as list_files_recursively already processes them.
        if file_meta.get('mimeType') == 'application/vnd.google-apps.folder':
            continue

        logger.info(f"[green]Downloading '{file_name}' (ID: {file_id}) to '{TEMP_INGESTION_DIR}'...[/green]")
        downloaded_path_str = db_manager.download_file(file_id, file_name, str(TEMP_INGESTION_DIR))

        if downloaded_path_str:
            downloaded_file_info.append({
                "path": Path(downloaded_path_str),
                "id": file_id,
                "original_name": file_name
            })
            logger.info(f"[green]File '{file_name}' downloaded successfully to: {downloaded_path_str}[/green]")
        else:
            logger.warning(f"[yellow]Failed to download '{file_name}'.Will be skipped in ingestion.[/yellow]")

    if not downloaded_file_info:
        logger.warning("[yellow]No files were successfully downloaded. Terminating ingestion.[/yellow]")
        return

    # 2. Read documents with LlamaIndex and add metadata with local summarization.
    llama_documents = []
    logger.info(f"[green]Reading {len(downloaded_file_info)} downloaded files with LlamaIndex...[/green]")
    for file_info_item in downloaded_file_info:
        file_path = file_info_item["path"]
        original_doc_name = file_info_item["original_name"]
        logger.info(f"[green]Processing document for LlamaIndex: {original_doc_name}[/green]")
        try:
            raw_documents_from_file = SimpleDirectoryReader(input_files=[file_path]).load_data()
            if not raw_documents_from_file:
                logger.warning(f"[yellow]SimpleDirectoryReader returned no documents for {original_doc_name}."
                               f" Skipping.[/yellow]")
                continue

            doc_for_llama_index = raw_documents_from_file[0]
            # Add important metadata to LlamaIndex Document after summarization:
            doc_for_llama_index.metadata["google_drive_id"] = file_info_item["id"]
            doc_for_llama_index.metadata["original_file_name"] = original_doc_name
            doc_for_llama_index.metadata["ingestion_source"] = "google_drive"

            # START OF SUMMARIZATION WITH OLLAMA
            document_summary = "Summary not generated (agent or text unavailable)."
            if summarizer_agent and doc_for_llama_index.text and doc_for_llama_index.text.strip():
                logger.info(f"    [bold green]Summarizing content of: {original_doc_name} using Ollama model "
                            f"{OLLAMA_MODEL_FOR_SUMMARY} (This may take time)...[/bold green]")
                max_chars_for_summary = 30000
                text_to_summarize = doc_for_llama_index.text[:max_chars_for_summary]
                if len(doc_for_llama_index.text) > max_chars_for_summary:
                    logger.warning(f"    [yellow]Text from '{original_doc_name}' truncated for "
                                   f"summarization ({len(text_to_summarize)} chars).[/yellow]")

                summary_task_desc = (
                    f"Provide a concise summary (3 to 7 sentences) of the following document text. "
                    f"Focus on the main topics, purpose, and key takeaways. "
                    f"The document is titled '{original_doc_name}'.\n\n"
                    f"Document Text:\n---\n{text_to_summarize}\n---"
                )
                summary_task = CrewAITask(
                    description=summary_task_desc,
                    agent=summarizer_agent,
                    expected_output="A string containing the concise summary of the document."
                )

                try:
                    summary_crew = Crew(
                        agents=[summarizer_agent],
                        tasks=[summary_task],
                        process=Process.sequential,
                        verbose=False
                    )
                    document_summary_result = summary_crew.kickoff()

                    if isinstance(document_summary_result, str) and document_summary_result.strip():
                        document_summary = document_summary_result.strip()
                        logger.info(f"     [green]Summary generated for {original_doc_name} by Ollama: "
                                    f"{document_summary[:100]}...[/green]")
                    else:
                        logger.warning(f"     [yellow]Ollama summary generation for {original_doc_name} "
                                       f"returned an unexpected result or was empty:"
                                       f" {document_summary_result}[/yellow]")
                        document_summary = "Ollama summary generation failed or returned empty."

                except Exception as summary_e:
                    logger.error(f"     [red]Error during Ollama summary generation task for "
                                 f"{original_doc_name}: {summary_e}[/red]", exc_info=True)
                    document_summary = f"Error during Ollama summary generation: {str(summary_e)[:150]}"
            elif not doc_for_llama_index.text or not doc_for_llama_index.text.strip():
                logger.warning(f"     [yellow]No text content found in {original_doc_name} to summarize.[/yellow]")
                document_summary = "No text content to summarize."

            doc_for_llama_index.metadata["document_summary"] = document_summary
            # END OF SUMMARIZATION

            llama_documents.append(doc_for_llama_index)
            logger.info(f"[green]Document '{original_doc_name}' loaded, summarized (if possible), and "
                        f"metadata added.[/green]")
        except Exception as e:
            logger.error(f"[red]Error loading or summarizing file '{file_path}' with LlamaIndex: "
                         f"{e}[/red]", exc_info=True)

    if not llama_documents:
        logger.warning("[yellow]No documents were successfully processed by LlamaIndex. Terminating "
                       "ingestion.[/yellow]")
        return
    logger.info(f"[green]{len(llama_documents)} LlamaIndex Document objects created and enriched.[/green]")

    # 3. Configure LlamaIndex Global Embeddings (NodeParser).
    logger.info("[green]Configuring LlamaIndex global Settings "
                "(HuggingFace Embeddings & NodeParser)...[/green]")
    try:
        logger.info(f"[green]Initializing HuggingFaceEmbedding with model: {EMBEDDING_MODEL_NAME_HF}[/green]")
        embed_model_instance = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME_HF)
        Settings.embed_model = embed_model_instance
        logger.info("[green]LlamaIndex Settings.embed_model configured with HuggingFaceEmbedding "
                    "({EMBEDDING_MODEL_NAME_HF}).[/green]")
    except Exception as e:
        logger.error(f"[red]Error configuring LlamaIndex HuggingFace embedding model "
                     f"({EMBEDDING_MODEL_NAME_HF}): {e}. Ensure the model name is correct, you have internet "
                     f"to download it if needed, and 'sentence-transformers' library is "
                     f"installed.[/red]", exc_info=True)
        return

    Settings.node_parser = SentenceSplitter(
        chunk_size=256,
        chunk_overlap=25
    )
    Settings.llm = None
    logger.info("[green]LlamaIndex Settings.node_parser (SentenceSplitter) and Settings.llm "
                "(None) configured.[/green]")

    # 4. Prepare Data and save in ChromaDB via LlamaIndex.
    logger.info(f"[green]Initializing ChromaDB client for persistence at: {CHROMA_PERSIST_PATH}[/green]")
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
        logger.info(f"[green]ChromaDB PersistentClient created. Attempting to get/create collection:"
                    f" '{CHROMA_COLLECTION_NAME}'[/green]")
        chroma_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        logger.info(f"[green]ChromaVectorStore configured for collection '{CHROMA_COLLECTION_NAME}'.[/green]")
    except Exception as e:
        logger.error(f"[red]Error initializing ChromaDB or ChromaVectorStore: {e}. "
                     "Ensure ChromaDB is installed and the path is writable.[/red]", exc_info=True)
        return

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    logger.info("[green]StorageContext configured.[/green]")

    logger.info("[green]Creating/Updating VectorStoreIndex with ChromaDB (this may take a while)...[/green]")
    index = VectorStoreIndex.from_documents(
        documents=llama_documents,
        storage_context=storage_context,
        show_progress=True
    )
    logger.info(f"[green]Indexing to ChromaDB collection '{CHROMA_COLLECTION_NAME}' complete. "
                f"Index ID (LlamaIndex internal): '{index.index_id}'[/green]")

    # 5. Cleaning the temporary directory.
    logger.info(f"[green]Cleaning up downloaded files from '{TEMP_INGESTION_DIR}'...[/green]")
    try:
        cleanup_temp_folder(str(TEMP_INGESTION_DIR))
        logger.info(f"[green]Temporary directory '{TEMP_INGESTION_DIR}' cleaned up.[/green]")
    except Exception as e:
        logger.warning(f"[yellow]Error cleaning up temporary directory: {e}.[/yellow]", exc_info=True)

    logger.info("[green]Knowledge Ingestion Pipeline completed[/green]")


if __name__ == "__main__":
    run_ingestion()
import asyncio
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import Settings as LlamaIndexSettings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from crewai import Agent as CrewAIAgent, Task as CrewAITask, Crew, Process, LLM
from core.ingestion_pipeline import run_ingestion
from core.FolderManager import clear_and_ensure_path
from utills.logger import setup_logger
from tools.llama_index_tools import LlamaIndexQueryTool
# from agents.query_agents import UserQueryUnderstandingAgent
# from agents.synthesis_agents import RAGSynthesisAgent
from agents.document_agents import QuerySynthesisAgent
from tasks.document_tasks import create_query_synthesis_tasks

logger = setup_logger(
    logger_name="rag_application_main",
    log_file="logs/rag_application_main.log",
    log_level=logging.INFO
)

def setup_rag_components_and_tools():
    """
    Configure and initialize the central components for the RAG application:
    local LLM (Ollama), local embedding model (HuggingFace), local Reranker and the LlamaIndexQueryTool (JEC).
    Returns:
         Tuple[Optional[Ollama]], Optional[LlamaIndexQueryTool]: crewai_ollama_llm, jec_tool
    """
    logger.info("[bold green]Setting up RAG components and tools...[/bold green]")

    chroma_persist_path = os.getenv("CHROMA_PERSIST_PATH", "./chroma_db_store")
    chroma_collection_name = os.getenv("CHROMA_COLLECTION_NAME", "MyFreeKnowledgeBase")
    embedding_model_name_hf = os.getenv("EMBEDDING_MODEL_NAME_HF", "sentence-transformers/all-MiniLM-L6-v2")
    ollama_model_for_crew = os.getenv("OLLAMA_MODEL_FOR_CREW")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    reranker_model_name = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker_top_n = int(os.getenv("RERANKER_TOP_N", "3"))

    if not all([chroma_persist_path, chroma_collection_name, embedding_model_name_hf,
                ollama_model_for_crew, ollama_base_url, reranker_model_name]):
        logger.error("[bold red]Missing critical environment variables for free RAG setup. "
                     "Check CHROMA_PERSIST_PATH, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME_HF, "
                     "OLLAMA_MODEL_FOR_CREW, OLLAMA_BASE_URL, RERANKER_MODEL_NAME.[/bold red]")
        return None, None

    try:
        # 1. Local LLM (Ollama) for CrewAI agents.
        ollama_model_with_provider = f"ollama/{ollama_model_for_crew}"
        logger.info(f"[green]Initializing Ollama LLM for CrewAI agents ({ollama_model_for_crew}) "
                    f"at {ollama_base_url}...[/green]")
        crewai_ollama_llm = LLM(
            model=ollama_model_with_provider,
            base_url=ollama_base_url,
            model_kwargs={
                "temperature": 0.3,
                "repeat_penalty": 1.15,
                "top_k": 40,
                "top_p": 0.9,
                "num_ctx": 32768,
                "timeout": 240
            }
        )
        try:
            if hasattr(crewai_ollama_llm, 'invoke'):
                crewai_ollama_llm.invoke("Respond with just 'OK' if you are working.")
                logger.info(f"[green]Successfully connected to Ollama and model {ollama_model_with_provider} "
                            f"seems responsive.[/green]")
            else:
                logger.warning("[yellow]LLM object has no 'invoke' method. Cannot confirm responsiveness.[/yellow]")

        except Exception as ollama_test_e:
            logger.warning(f"[yellow]Could not confirm Ollama (for CrewAI) responsiveness: "
                           f"{ollama_test_e}.[/yellow]")
        logger.info("[green]Ollama LLM for CrewAI agents initialized.[/green]")

        # 2. Local embedding model (HuggingFace) for LlamaIndex.
        logger.info(f"[blue]Initializing HuggingFaceEmbedding with model: {embedding_model_name_hf}[/blue]")
        embed_model_instance = HuggingFaceEmbedding(model_name=embedding_model_name_hf)
        logger.info(f"[green]HuggingFaceEmbedding instance ({embedding_model_name_hf}) created.[/green]")

        # 3. Local Reranker (SentenceTransformerRerank).
        logger.info(f"[blue]Initializing SentenceTransformerRerank with model: {reranker_model_name} "
                    f"and top_n={reranker_top_n}[/blue]")
        reranker_instance = SentenceTransformerRerank(
            model=reranker_model_name,
            top_n=reranker_top_n
        )
        logger.info(f"[green]SentenceTransformerRerank instance created.[/green]")

        # 4. LlamaIndexQueryTool (JEC) with ChromaDB and local components.
        LlamaIndexSettings.llm = None
        jec_tool = LlamaIndexQueryTool(
            chroma_persist_path=chroma_persist_path,
            chroma_collection_name=chroma_collection_name,
            embedding_model_instance=embed_model_instance,
            reranker_jec=reranker_instance
        )
        logger.info("[green]LlamaIndexQueryTool (JEC) initialized with local components.[/green]")

        return crewai_ollama_llm, jec_tool

    except Exception as e:
        logger.error(f"[bold red]Error during RAG component setup: {e}[/bold red]", exc_info=True)
        return None, None

async def run_rag_application():
    """
    Asynchronous main function that orchestrates the RAG application.
    """
    logger.info("[bold green]Starting RAG Application Main Logic[/bold green]")
    load_dotenv()

    chroma_persist_path = os.getenv("CHROMA_PERSIST_PATH", "./chroma_db_store")

    # Conditional logic for ingestion
    if os.path.exists(chroma_persist_path) and os.listdir(chroma_persist_path):
        logger.info(f"[yellow]ChromaDB directory '{chroma_persist_path}' found and is not empty. Skipping "
                    f"ingestion.[/yellow]")
    else:
        logger.info(f"[bold blue]ChromaDB directory '{chroma_persist_path}' not found or is empty. Running "
                    f"ingestion pipeline.[/bold blue]")
        run_ingestion()

    crewai_ollama_llm, jec_tool = setup_rag_components_and_tools()

    if not crewai_ollama_llm or not jec_tool:
        logger.error("[bold red]Failed to set up RAG components. Application cannot continue.[/bold red]")
        return

    test_query = "Which five countries have the highest perception of happiness?"
    jec_results = []
    # Directly JEC Tool test (for development).
    if jec_tool and jec_tool.index and jec_tool.query_engine:
        logger.info(f"[blue]Performing a direct test of JEC tool with query: '{test_query}'[/blue]")
        try:
            retrieved_docs = await asyncio.to_thread(
                jec_tool.run,
                query=test_query,
                top_k=3
            )

            if retrieved_docs and not isinstance(retrieved_docs[0],dict) or "error" in retrieved_docs[0]:
                logger.error(f"[bold red]JEC Test - Error retrieving documents: {retrieved_docs}[/bold red]")
            else:
                jec_results = retrieved_docs
                logger.info(f"[bold green]JEC Test - Retrieved {len(jec_results)} document snippets:[/bold green]")
                for i, res_item in enumerate(jec_results):
                    if isinstance(res_item, dict):
                        logger.info(f"  --- Snippet {i+1} ---")
                        logger.info(f"  Score: {res_item.get('score')}")
                        logger.info(f"  Content: {res_item.get('text_content', '')[:200]}...")
                        logger.info(f"  Metadata: {res_item.get('metadata')}")
                        if "document_summary" in res_item.get("metadata", {}):
                            logger.info(f"    Doc Summary: {res_item['metadata']['document_summary'][:100]}...")
                    else:
                        logger.warning(f"  --- JEC Test - Result {i+1} is not a dictionary: {res_item} ---")
        except Exception as e_jec_test:
            logger.error(f"[bold red]Error during direct JEC tool test: "
                         f"{e_jec_test}[/bold red]", exc_info=True)
    else:
        logger.error("[bold red]JEC tool index or query engine not initialized. Cannot perform direct "
                     "test.[/bold red]")

    # Testing the QuerySynthesisAgent and Task if JEC results are available.
    if jec_results and crewai_ollama_llm:
        logger.info(f"\n[bold blue]Proceeding to Query Synthesis with {len(jec_results)} retrieved "
                    f"snippets...[/bold blue]")
        # 1. Instantiate the Agent
        query_synthesis_agent_instance = QuerySynthesisAgent(llm=crewai_ollama_llm)

        # 2. Create the Task
        synthesis_task_instance = create_query_synthesis_tasks(
            llm=crewai_ollama_llm,
            user_query=test_query,
            document_snippets=jec_results
        )

        # 3. Create and run the crew
        synthesis_crew = Crew(
            agents=[synthesis_task_instance.agent],
            tasks=[synthesis_task_instance],
            process=Process.sequential,
            verbose=True
        )

        logger.info("[blue]Kicking off Query Synthesis Crew...[/blue]")
        try:
            synthesis_result = await asyncio.to_thread(synthesis_crew.kickoff)

            logger.info("[bold green]\n--- Query Synthesis Result ---[/bold green]")
            logger.info(f"\n{synthesis_result}\n")
            logger.info("[bold green]----------------------------[/bold green]")
        except Exception as e_synthesis_crew:
            logger.error(f"[bold red]Error during Query Synthesis Crew execution: "
                         f"{e_synthesis_crew}[/bold red]", exc_info=True)
    elif not jec_results:
        logger.warning("[yellow]Skipping Query Synthesis as no documents were retrieved by JEC.[/yellow]")

    logger.info("[bold green]RAG Application main logic finished.[/bold green]")

if __name__ == "__main__":
    try:
        asyncio.run(run_rag_application())
    except KeyboardInterrupt:
        logger.info("\n[bold orange]Application interrupted by user.[/bold orange]")
    except Exception as e_fatal:
        logger.error(f"\n[bold red]Fatal unhandled error in asyncio event loop: "
                     f"{e_fatal}[/bold red]", exc_info=True)
    finally:
        logger.info("[bold blue]RAG Application script finished.[/bold blue]")
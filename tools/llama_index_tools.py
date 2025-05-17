import logging
from typing import Any, List, Optional, Dict, Type

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from crewai.tools import BaseTool
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.schema import MetadataMode, NodeWithScore
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from pydantic import Field, BaseModel, model_validator

from utills.logger import setup_logger

logger = setup_logger(
    logger_name="llama_index_tools",
    log_file="logs/llama_index_tools.log",
    log_level=logging.INFO
)

class LlamaIndexQueryToolInput(BaseModel):
    query: str = Field(description="The natural language query to search for.")
    top_k: int = Field(default=3, description="Optional number of results to return, defaults to 3.")

class LlamaIndexQueryTool(BaseTool):
    name: str = "judicious_evocation_of_content"
    description: str = (
        "Searches a ChromaDB vector database (managed by LlamaIndex) for relevant text chunks "
        "based on a natural language query. Use this to find information within the indexed knowledge base."
        "Input should be the user's query (str) and optionally the number of results (top_k, int)."
    )
    args_schema: Type[BaseModel] = LlamaIndexQueryToolInput

    chroma_persist_path: str
    chroma_collection_name: str
    embedding_model_instance: Any
    reranker_jec: Optional[BaseNodePostprocessor] = None

    query_engine: Optional[BaseQueryEngine] = None
    index: Optional[VectorStoreIndex] = None

    # def __init__(self,
    #              chroma_persist_path: str,
    #              chroma_collection_name: str,
    #              embed_model_instance: Any,
    #              reranker_jec: Optional[BaseNodePostprocessor] = None,
    #              **kwargs):
    #     """
    #     Initializes the tool with ChromaDB connection details, embedding model, and reranker.
    #     Args:
    #         chroma_persist_path: Path to the ChromaDB persistence directory.
    #         chroma_collection_name: Name of the collection in ChromaDB.
    #         embed_model_instance: An instance of the embedding model (e.g., HuggingFaceEmbedding).
    #         reranker_jec: Optional reranker instance (e.g., SentenceTransformerRerank).
    #     """
    #     super().__init__(**kwargs)
    #     self.chroma_persist_path = chroma_persist_path
    #     self.chroma_collection_name = chroma_collection_name
    #     self.embed_model_instance = embed_model_instance
    #     self.reranker_jec = reranker_jec
    #     self._initialize_query_engine()

    @model_validator(mode='after')
    def initialize_tool_components(self) -> 'LlamaIndexQueryTool':
        """Validator called after Pydantic model initialization to set up LlamaIndex components."""
        logger.info(f"LlamaIndexQueryTool: Running @model_validator to initialize components.")
        logger.info(f"  chroma_persist_path: {self.chroma_persist_path}")
        logger.info(f"  chroma_collection_name: {self.chroma_collection_name}")
        logger.info(f"  embedding_model_instance type: {type(self.embedding_model_instance)}")
        logger.info(f"  reranker_jec type: {type(self.reranker_jec)}")
        self._initialize_query_engine()
        return self

    def _initialize_query_engine(self):
        logger.info(f"[green]Tool: Initializing Index and QueryEngine for ChromaDB: "
                    f"path='{self.chroma_persist_path}', collection='{self.chroma_collection_name}'[/green]")
        try:
            # 1. Configure the embedding model globally for LlamaIndex.
            if Settings.embed_model is None or \
                    not isinstance(Settings.embed_model, type(self.embedding_model_instance)) or \
                    (hasattr(Settings.embed_model, 'model_name') and
                    hasattr(self.embedding_model_instance, 'model_name') and
                    Settings.embed_model.model_name != self.embedding_model_instance.model_name):
                Settings.embed_model = self.embedding_model_instance
                logger.info(f"[green]LlamaIndex Settings.embed_model configured with "
                            f"{type(self.embedding_model_instance).__name__} "
                            f"({getattr(self.embedding_model_instance, 'model_name', 'N/A')}).[/green]")
            else:
                logger.info(f"[green]LlamaIndex Settings.embed_model already configured with a compatible "
                            f"model.[/green]")

            # 2. Conecte to ChromaDB.
            chroma_client = chromadb.PersistentClient(path=self.chroma_persist_path)
            logger.info(f"[green]ChromaDB PersistentClient connected to path: {self.chroma_persist_path}[/green]")

            chroma_collection = chroma_client.get_or_create_collection(self.chroma_collection_name)
            logger.info(f"[green]ChromaDB collection '{self.chroma_collection_name}' accessed/created.[/green]")

            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            logger.info("[green]ChromaVectorStore adapter initialized.[/green]")

            # 3. Load the VectorStore index.
            Settings.llm = None
            logger.info("[green]LlamaIndex Settings.llm set to None for JEC tool (focus on retrieval).[/green]")

            self.index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store
            )
            logger.info("[green]VectorStoreIndex loaded from ChromaDB VectorStore.[/green]")

            # 4. Configure the Query Engine (or use retriever directly in run).
            query_engine_params = {
                "similarity_top_k": 10,
                "vector_store_query_mode": VectorStoreQueryMode.DEFAULT,
                "response_mode": "no_text",
            }

            if self.reranker_jec:
                query_engine_params["node_postprocessors"] = [self.reranker_jec]  # type: ignore[assignment]
                logger.info(f"[green]Reranker ({type(self.reranker_jec).__name__}) added to query engine "
                            f"params.[/green]")

            self.query_engine = self.index.as_query_engine(**query_engine_params)
            logger.info("[green]Tool: Default QueryEngine successfully initialized with ChromaDB.[/green]")

        except AttributeError as ae:
            logger.error(f"[red]Tool: AttributeError during QueryEngine initialization: {ae}. "
                         f"This might indicate a Pydantic field was not correctly set.[/red]", exc_info=True)
            self.query_engine = None
            self.index = None

        except Exception as e:
            logger.error(f"[red]Tool: Error initializing QueryEngine with ChromaDB: {e}.[/red]", exc_info=True)
            self.query_engine = None
            self.index = None

    def _run(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.index or not self.query_engine:
            logger.error("[red]Tool: Index or QueryEngine not initialized. Cannot perform search.[/red]")
            return [{"error": "Tool Error: Index or QueryEngine not initialized."}]

        retriever_similarity_top_k_val = "N/A"
        retriever_obj = getattr(self.query_engine, '_retriever', None)
        if retriever_obj:
            retriever_similarity_top_k_val = getattr(retriever_obj, '_similarity_top_k', "N/A")

        if not query:
            logger.warning("[yellow]Tool: Received an empty query.[/yellow]")
            return [{"error": "Query cannot be empty."}]

        logger.info(f"[green]Tool _run initiated with query='{query}', effective top_k for retriever "
                    f"is {retriever_similarity_top_k_val}, "
                    f"reranker will yield up to {getattr(self.reranker_jec, 'top_n', 'N/A')} results.[/green]")

        try:
            response = self.query_engine.query(query)
            retrieved_nodes_with_scores = []
            if  hasattr(response, 'source_nodes'):
                retrieved_nodes_with_scores = response.source_nodes
            elif hasattr(response, 'sources'):
                for source in response.sources:
                    if hasattr(source, 'node') and hasattr(source, 'score'):
                        retrieved_nodes_with_scores.append(source)
                    elif hasattr(source, 'node'):
                        retrieved_nodes_with_scores.append(NodeWithScore(
                            node=source.node, score=source.score if hasattr(source, 'score') else None
                        ))
            else:
                logger.warning("[yellow]Tool: Could not directly extract source_nodes or sources from query "
                               "engine response. Response type: {type(response)}[/yellow]")
                if isinstance(response, list) and all(hasattr(item, 'node') for item in response):
                    retrieved_nodes_with_scores = response

            logger.info(f"[green]Tool: QueryEngine processed query. Retrieved {len(retrieved_nodes_with_scores)} "
                        f"nodes after postprocessing (reranking).[/green]")

            results = []
            final_nodes_to_process = retrieved_nodes_with_scores[:top_k]

            for node_with_score in final_nodes_to_process:
                node_data = {
                    "text_content": node_with_score.node.get_content(metadata_mode=MetadataMode.ALL),
                    "score": node_with_score.score if node_with_score is not None else 0.0,
                    "node_id": node_with_score.node.node_id,
                    "metadata": node_with_score.node.metadata or {}
                }
                results.append(node_data)
                logger.debug(f"[green]Retrieved & Reranked Node ID: {node_data['node_id']}, "
                             f"Score: {node_data['score']:.4f}[/green]")

            return  results[:top_k]
        except Exception as e:
            logger.error(f"[red]Tool: Error during search in _run method: {e}.[/red]", exc_info=True)
            return [{"error": f"Search Error: {str(e)}"}]
import datetime
import os
import traceback

from crewai import Agent
from typing import Any, Dict

from crewai.tools import BaseTool

from tools.document_tools import ExtractTextTool, CountWordsTool


class DocumentAnalysisAgent(Agent):
    """
    Responsável por extrair informações concisas dos documentos: contagem de palavras e resumos.
    Utiliza ferramentas para extração e contagem, e seu próprio LLM para sumarização.
    """
    def __init__(self, llm: Any):
        super().__init__(
            role="Concise Document Analyst",
            goal="Extract key information (like word count) from document content using tools, "
                 "and generate a concise summary using your own reasoning capabilities. "
                 "Prepare the structured data for reporting.",
            backstory="I am a meticulous document analyst. I use tools to extract text and count "
                      "words, and then apply my language understanding to summarize the core message efficiently. "
                      "My goal is to provide structured, actionable insights from documents.",
            tools=[
                ExtractTextTool(),
                CountWordsTool(),
            ],
            memory=False,
            verbose=False,
            llm=llm
        )


class ReportingAgent(Agent):
    """
    Agente para consolidar análises e gerar relatórios em TXT usando uma ferramenta.
    """
    def __init__(self, llm: Any):
        super().__init__(
            role="Document Analysis Consolidator and Reporter",
            goal="Consolidate analysis results (summaries, word counts, etc.) from multiple documents "
                 "(provided as context from previous tasks), structure them into a coherent dictionary, "
                 "and use the 'save_analysis_report' tool to save this dictionary as a text file "
                 "in the specific directory provided in the task instructions.",
            backstory="I specialize in synthesizing information from multiple document analyses. "
                      "I take the structured results from previous steps, organize them into a Python dictionary, "
                      "and then use a dedicated tool to save this data persistently as a text report "
                      "in the designated location.",
            tools=[
                GenerateReportTool() # Ferramenta para salvar em .txt
            ],
            memory=False,
            verbose=False, # Mantenha False para produção
            llm=llm
        )


class GenerateReportTool(BaseTool):
    name: str = "save_analysis_report"
    description: str = (
        "Saves the provided analysis results dictionary to a text file in a specified directory. "
        "Input should be the analysis results (dict) and the target directory path (str) for the report."
    )

    def _run(self, analysis_results: Dict[str, Any], report_directory: str = "reports") -> str:
        """
        Saves the analysis results dictionary to a file within the specified directory.
        Args:
            analysis_results (dict): Dictionary containing analysis results.
            report_directory (str): Directory to save the report. Defaults to 'reports'.
        Returns:
            str: Path to the saved report file or an error message.
        """
        if not isinstance(analysis_results, dict):
            return "Error: Input 'analysis_results' must be a dictionary."
        if not isinstance(report_directory, str) or not report_directory:
            return "Error: Input 'report_directory' must be a non-empty string."

        try:
            # Cria o diretório especificado. Se for uma subpasta, ele criará as pastas pais também.
            os.makedirs(report_directory, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"batch_analysis_report_{timestamp}.txt"
            report_path = os.path.join(report_directory, report_filename)

            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"Document Analysis Report - Generated: {timestamp}\n")
                f.write(f"Saved in directory: {report_directory}\n")
                f.write("========================================\n\n")
                if isinstance(analysis_results, dict):
                    for key, results in analysis_results.items():
                        f.write(f"--- Entry: {str(key)} ---\n")
                        if isinstance(results, dict):
                            # Tenta extrair campos comuns da análise de documento
                            file_path_str = str(results.get('file_path', 'N/A'))
                            word_count_str = str(results.get('word_count', 'N/A'))
                            summary_str = str(results.get('summary', 'N/A'))
                            full_text_preview = str(results.get('full_text', ''))[:100] # Preview
                            f.write(f"  File Path Hint: {file_path_str}\n")
                            f.write(f"  Word Count: {word_count_str}\n")
                            f.write(f"  Summary: {summary_str}\n")
                            if full_text_preview:
                                f.write(f"  Text Preview: {full_text_preview}...\n")
                            # Adiciona outros campos se existirem
                            other_keys = {k: v for k, v in results.items() if k not in ['file_path',
                                                                                        'word_count',
                                                                                        'summary',
                                                                                        'full_text']}
                            if other_keys:
                                f.write(f"  Other Data: {str(other_keys)}\n")
                        else:
                            # Caso o valor não seja um dicionário
                            f.write(f"  Result: {str(results)}\n")
                        f.write("\n")
                else:
                    f.write(f"Raw Results Data:\n{str(analysis_results)}\n")

                f.write("========================================\n")
                f.write(f"End of Report - {timestamp}\n")

            print(f"Report saved by tool to: {report_path}")
            # Retorna o caminho completo para a task
            return f"Report successfully saved to: {report_path}"
        except OSError as e:
            error_message = f"Error creating directory or saving file '{report_path}': {e}"
            print(error_message)
            traceback.print_exc()
            return error_message
        except Exception as e:
            error_message = f"Unexpected error saving report to '{report_directory}': {e}"
            print(error_message)
            traceback.print_exc()
            return error_message


class DataMiningAgent(Agent):
    """
    Encontra padrões e métricas nos dados extraídos.
    """
    def __init__(self, llm: Any):
        super().__init__(
            role="Data Mining Expert",
            goal="Identify patterns, trends, and key metrics within the extracted data.",
            backstory="I am a seasoned data mining expert with a keen eye for detail. "
                      "My mission is to uncover hidden "
                      "patterns and trends within complex datasets, providing valuable insights.",
            tools=[
                CountWordsTool()
            ],
            memory=True,
            verbose=False,
            llm=llm
        )


class QuerySynthesisAgent(Agent):
    """
    Agent responsible for synthesizing an answer from provided document snippets
    and listing the source documents with their Google Drive links.
    """
    def __init__(self, llm: Any):
        super().__init__(
            role="Contextual Answering and Sourcing Specialist",
            goal="Synthesize a comprehensive answer to the user's query using the provided text snippets from "
                 "relevant documents. Clearly list the source documents, including their original names, Google Drive "
                 "IDs, and direct Google Drive view links, to allow the user to reference or download them.",
            backstory="I'm ai AI assistant skilled at understanding user questions and piecing together information "
                      "from multiple text sources to provide a clear, consolidated answer. I always cite my sources "
                      "meticulously, providing actionable links for further exploration.",
            llm=llm,
            verbose=True,
            memory=False,
            tools=[]
        )
import asyncio
import os
import traceback
import math
import datetime

from crewai import LLM

from Agents import DocumentAnalysisAgent, ReportingAgent
from Authentication import GoogleDriveAPI
from DataBaseManager import initialize_apis_and_db, list_drive_files
from dotenv import load_dotenv

from ReportGeneretor import process_batches

DRIVE_FOLDER_ID = "1lXQ7R5z8NGV1YGUncVDHntiOFX35r6WO"
REPORT_DIR = "google_drive_reports"
TEMP_DIR = "temp_drive_files"
BATCH_SIZE = 2  # Define o número de arquivos por lote.


def initialize_agents(llm_instance: LLM):
    """Inicializa os agentes de análise de documentos e relatórios com o LLM fornecido."""
    print("Inicializando Agentes CrewAI (com instância LLM centralizada)...")
    try:
        # Passa a instância LLM recebida
        document_agent = DocumentAnalysisAgent(llm=llm_instance)
        reporting_agent = ReportingAgent(llm=llm_instance)
        print("Agentes inicializados com sucesso.")
        return document_agent, reporting_agent
    except TypeError as e:
        print(f"Erro de Tipo ao inicializar agentes: {e}")
        traceback.print_exc()
        return None, None
    except Exception as e:
        print(f"Erro inesperado ao inicializar agentes: {e}")
        traceback.print_exc()
        return None, None


async def main():
    """Função principal assíncrona que orquestra o processo."""
    print("Carregando variáveis de ambiente...")
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("AVISO URGENTE: GOOGLE_API_KEY não encontrada!")
        exit(1)

    # --- Criação centralizada do crewai.LLM ---
    print("Criando instância centralizada do crewai.LLM...")
    try:
        # Usando o nome do modelo que o LiteLLM/CrewAI deve entender
        my_llm = LLM(
            model="gemini/gemini-1.5-flash-latest",
            api_key=google_api_key
        )
        print("Instância crewai.LLM criada com sucesso.")
    except Exception as e:
        print(f"Erro CRÍTICO ao criar a instância crewai.LLM: {e}")
        traceback.print_exc()
        return

    # Criação do diretório de relatório específico para esta execução
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_specific_report_dir = os.path.join(REPORT_DIR, f"run_{timestamp}")
    print(f"Diretório de relatório para esta execução: {run_specific_report_dir}")
    try:
        os.makedirs(run_specific_report_dir, exist_ok=True)
        os.makedirs(TEMP_DIR, exist_ok=True)
        print(f"Diretórios '{run_specific_report_dir}' e '{TEMP_DIR}' garantidos.")
    except OSError as e:
        print(f"Erro ao criar diretórios: {e}")
        traceback.print_exc()
        return

    print("Autenticando com Google Drive...")
    drive_api = GoogleDriveAPI()
    drive_service = drive_api.service
    if not drive_service:
        print("Falha ao obter o serviço do Google Drive.")
        return
    print("Inicializando Gerenciador de Banco de Dados (Drive)...")
    db_manager = initialize_apis_and_db(drive_service)
    if db_manager is None:
        return

    # Passa a instância my_llm (crewai.LLM) para inicializar os agentes
    document_agent, reporting_agent = initialize_agents(my_llm)
    if document_agent is None or reporting_agent is None:
        print("Falha ao inicializar os agentes. Encerrando.")
        return

    print(f"Listando arquivos da pasta Google Drive ID: {DRIVE_FOLDER_ID}")
    files = list_drive_files(db_manager, DRIVE_FOLDER_ID)
    if files is None:
        print("Falha ao listar arquivos do Google Drive.")
        return

    if not files:
        print("Nenhum arquivo encontrado na pasta especificada do Google Drive.")
        return
    else:
        total_files = len(files)
        if BATCH_SIZE <= 0:
            print("Erro: BATCH_SIZE deve ser um número positivo.")
            return
        total_batches = math.ceil(total_files / BATCH_SIZE)
        print(f"Encontrados {total_files} arquivos. Serão processados em {total_batches} lotes de até {BATCH_SIZE} "
              f"arquivos cada.")

    print("Iniciando processamento em lotes...")
    try:
        await process_batches(
            files=files,
            total_files=total_files,
            total_batches=total_batches,
            batch_size=BATCH_SIZE,
            db_manager=db_manager,
            document_agent=document_agent,
            reporting_agent=reporting_agent,
            temp_dir=TEMP_DIR,
            report_dir=run_specific_report_dir,
            llm_instance=my_llm
        )
        print("Processamento em lotes concluído.")
    except Exception as e:
        print(f"Erro durante a execução de process_batches: {e}")
        traceback.print_exc()


# --- Ponto de Entrada do Script ---
if __name__ == "__main__":
    print("Executando script principal (Main.py)...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExecução interrompida pelo usuário.")
    except Exception as e:
        print(f"\nErro fatal não tratado no loop de eventos asyncio: {e}")
        traceback.print_exc()
    finally:
        print("Script principal finalizado.")
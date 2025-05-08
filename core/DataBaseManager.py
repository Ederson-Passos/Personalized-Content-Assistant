# DataBaseManager.py
import os.path
import io
import os
import time
import traceback
from io import BytesIO
from typing import List, Optional, Tuple, Union

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError
)

from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

from TextExtractor import TEMP_DOWNLOAD_FOLDER

# Diretório onde os arquivos serão baixados temporariamente
DOWNLOAD_FOLDER = TEMP_DOWNLOAD_FOLDER

GOOGLE_DOCS_EXPORT_MIMETYPES = {
    'application/vnd.google-apps.document': 'application/pdf',
    'application/vnd.google-apps.spreadsheet': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.google-apps.presentation': 'application/pdf'
}

EXPORT_EXTENSIONS = {
    'application/pdf': '.pdf',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx'
}

def is_retryable_http_error(exception: BaseException) -> bool:
    """Verifica se uma exceção é um HttpError com status code que justifica retentativa."""
    if isinstance(exception, HttpError):
        # Códigos 5xx (Server errors), 408 (Request Timeout), 429 (Too Many Requests)
        return exception.resp.status >= 500 or exception.resp.status in [408, 429]
    return False


class DataBaseManager:
    """
    Classe para interagir com a API do Google Drive v3, fornecendo serviços como
    listar, baixar e processar arquivos.
    """
    def __init__(self, service):
        """
        Inicializa a classe com o objeto de serviço autenticado do Google Drive API.
        Args:
            service: Objeto de serviço autenticado do Google Drive API v3.
        """
        self.service = service
        self._processed_folders = set()
        os.makedirs(DOWNLOAD_FOLDER, exist_ok=True) # Garante que o diretório temporário exista

    @retry(
        stop=stop_after_attempt(3), # Tenta no máximo 3 vezes (1 original + 2 retries)
        wait=wait_exponential(multiplier=1, min=2, max=10), # Espera 2s, 4s
        retry=retry_if_exception_type((HttpError, TimeoutError)), # Tenta novamente para HttpError ou TimeoutError
        reraise=True # Re-levanta a exceção original se todas as tentativas falharem
    )
    def _get_file_metadata(self, file_id: str) -> Optional[dict]:
        """Busca metadados básicos de um arquivo, incluindo mimeType, com retentativas."""
        print(f"    [Retry Attempt] Buscando metadados para file ID {file_id}...")
        try:
            file_metadata = self.service.files().get(
                fileId=file_id, fields='id, name, mimeType'
            ).execute()
            print(f"    [Retry Success] Metadados obtidos para file ID {file_id}: {file_metadata.get('name')}")
            return file_metadata
        except HttpError as e:
            print(f"    [Retry Attempt Failed] Erro HTTP ao buscar metadados para file ID {file_id}: {e}")
            # Se o erro for retryable (5xx, 408, 429), tenacity tentará novamente.
            # Se não for (ex: 404 Not Found, 403 Forbidden), tenacity não tentará e levantará o erro.
            if not is_retryable_http_error(e):
                 print(f"    Erro HTTP não recuperável: {e.resp.status}. Não tentando novamente.")
                 return None # Ou levante um erro customizado
            raise # Re-levanta para tenacity decidir se tenta novamente baseado na condição 'retry'
        except TimeoutError as te: # Captura TimeoutError genérico se ocorrer
            print(f"    [Retry Attempt Failed] Timeout ao buscar metadados para file ID {file_id}: {te}")
            raise # Re-levanta para tenacity tentar novamente
        except Exception as e:
            # Captura outros erros inesperados que não são HttpError ou TimeoutError
            print(f"    [Retry Attempt Failed] Erro inesperado (não HTTP/Timeout) "
                  f"ao buscar metadados para file ID {file_id}: {e}")
            # Se quiser que o programa continue apesar deste erro, retorne None aqui.
            # return None
            raise # Re-levanta o erro inesperado

    def _prepare_file_download(self, file_id: str, file_name: str) -> Optional[tuple[str, str, str]]:
        """
        Prepara o download de um arquivo, determinando o tipo de exportação necessário e o nome final do arquivo.
        Args:
            file_id: ID do arquivo no Google Drive.
            file_name: Nome original do arquivo.
        Returns:
            Uma tupla contendo (original_mime_type, export_mime_type, final_file_name) ou None em caso de erro.
        """
        print(f"Preparando para baixar/exportar '{file_name}' (ID: {file_id})...")
        try:
            # Chama a função com retry
            metadata = self._get_file_metadata(file_id)
        except RetryError as re:
            # Captura o erro se todas as tentativas de _get_file_metadata falharem
            print(f"Falha ao obter metadados para '{file_name}' (ID: {file_id}) após múltiplas tentativas: {re}")
            return None
        except Exception as e:
            # Captura outros erros não relacionados a retry que _get_file_metadata possa levantar
            print(f"Erro final não tratado ao obter metadados para '{file_name}' (ID: {file_id}): {e}")
            return None

        if not metadata:
            print(f"Não foi possível obter metadados para '{file_name}', download cancelado.")
            return None

        original_mime_type = metadata.get('mimeType')
        export_mime_type = GOOGLE_DOCS_EXPORT_MIMETYPES.get(original_mime_type)

        # Ajusta o nome do arquivo e define o caminho final.
        base_name, _ = os.path.splitext(file_name)
        if export_mime_type:
            extension = EXPORT_EXTENSIONS.get(export_mime_type, '.bin') # Fallback de extensão
            final_file_name = f"{base_name}{extension}"
            print(f"Arquivo '{file_name}' é um Google Doc ({original_mime_type}). Exportando como {export_mime_type} "
                  f"para '{final_file_name}'...")
        else:
            final_file_name = file_name
            print(f"Arquivo '{file_name}' ({original_mime_type}) será baixado diretamente.")

        return original_mime_type, export_mime_type, final_file_name

    @retry(
        stop=stop_after_attempt(2), # Tenta no máximo 2 vezes (1 original + 1 retry)
        wait=wait_exponential(multiplier=1, min=5, max=30), # Espera 5s
        retry=retry_if_exception_type((HttpError, TimeoutError)), # Tenta novamente para erros HTTP ou Timeout
        reraise=True # Re-levanta a exceção original se todas as tentativas falharem
    )
    def _download_or_export_file_attempt(
            self,
            file_id: str,
            export_mime_type: Optional[str],
            final_file_name: str
    ) -> tuple[BytesIO, MediaIoBaseDownload]:
        """
        Tenta iniciar o download/exportação. Esta função é decorada com @retry.
        Retorna (fh, downloader) ou levanta uma exceção.
        """
        print(f"    [Retry Attempt] Iniciando download/exportação de '{final_file_name}' (ID: {file_id})...")
        try:
            if export_mime_type:
                request = self.service.files().export_media(
                    fileId=file_id, mimeType=export_mime_type
                )
            else:
                request = self.service.files().get_media(fileId=file_id)

            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            print(f"    [Retry Success] Conexão para download/exportação de '{final_file_name}' estabelecida.")
            return fh, downloader # Retorna o file handle e o downloader

        except HttpError as error:
            print(f"    [Retry Attempt Failed] Erro HTTP ao iniciar download/exportação de '{final_file_name}':"
                  f" {error}")
            if error.resp.status == 403:
                print("    Detalhes do erro 403:", error.content)
            # Se for retryable, tenacity tentará novamente. Senão, levantará o erro.
            raise # Re-levanta para tenacity
        except TimeoutError as te:
            print(f"    [Retry Attempt Failed] Timeout ao iniciar download/exportação "
                  f"de '{final_file_name}': {te}")
            raise # Re-levanta para tenacity
        except Exception as e:
            print(f"    [Retry Attempt Failed] Erro inesperado (não HTTP/Timeout) ao iniciar "
                  f"download/exportação de '{final_file_name}': {e}")
            traceback.print_exc()
            raise # Re-levanta o erro inesperado

    def download_file(self, file_id: str, file_name: str, destination_path=".") -> Optional[str]:
        """Baixa ou exporta um arquivo específico do Google Drive para um diretório local, com retentativas."""
        # 1. Preparação
        preparation_result = self._prepare_file_download(file_id, file_name)
        if not preparation_result:
            return None # Falha na obtenção de metadados ou preparação
        original_mime_type, export_mime_type, final_file_name = preparation_result

        # 2. Garantir diretório de destino
        if not _ensure_destination_directory(destination_path):
            return None

        file_path = os.path.join(destination_path, final_file_name)
        fh = None
        downloader = None

        # 3. Tentar iniciar o download/exportação (com retry)
        try:
            fh, downloader = self._download_or_export_file_attempt(
                file_id, export_mime_type, final_file_name
            )
        except RetryError as re:
            print(f"Falha ao iniciar download/exportação para '{final_file_name}' (ID: {file_id}) "
                  f"após múltiplas tentativas: {re}")
            return None
        except Exception as e:
            # Captura outros erros não tratados pelo retry que _download_or_export_file_attempt possa levantar
            print(f"Erro final não tratado ao iniciar download/exportação "
                  f"para '{final_file_name}' (ID: {file_id}): {e}")
            return None

        # Se chegou aqui, fh e downloader devem ser válidos
        if fh is None or downloader is None:
            print(f"Erro interno: fh ou downloader são None após tentativa "
                  f"de download bem-sucedida para '{final_file_name}'.")
            return None

        # 4. Baixar os chunks (a lógica de retry dentro do loop de chunks já existe em _download_chunks)
        print(f"Iniciando download dos chunks para '{final_file_name}'...")
        if not _download_chunks(downloader, final_file_name, fh):
            print(f"Download dos chunks falhou para '{final_file_name}'.")
            return None

        # 5. Salvar o arquivo
        print(f"\rDownload dos chunks para '{final_file_name}' concluído. Salvando arquivo...")
        if _save_file(file_path, fh):
            print(f"Arquivo '{final_file_name}' (ID: {file_id}) baixado com sucesso para '{file_path}'.")
            return file_path
        else:
            print(f"Falha ao salvar o arquivo '{final_file_name}' em '{file_path}'.")
            return None

    def list_files(self, folder_id: str, page_size: int = 100) -> List[dict]:
        """Lista os arquivos (não pastas) dentro de uma pasta específica do Google Drive."""
        if not self.service:
            print("Erro: Serviço do Google Drive não inicializado.")
            return []
        try:
            query = f"'{folder_id}' in parents and mimeType != 'application/vnd.google-apps.folder'"
            results = self.service.files().list(
                q=query,
                pageSize=page_size,
                fields="nextPageToken, files(id, name)"
            ).execute()
            items = results.get("files", [])
            if not items:
                print(f"Nenhum arquivo encontrado na pasta com ID: {folder_id}")
                return []
            print(f"Arquivos na pasta (ID: {folder_id}):")
            for item in items:
                print(f"- {item['name']} ({item['id']})")
            return items
        except HttpError as e:
            print(f"Ocorreu um erro ao listar os arquivos da pasta {folder_id}: {e}")
            return []
        except Exception as e:
            print(f"Ocorreu um erro inesperado ao listar arquivos: {e}")
            return []

    def _scan_items(self, items: List[dict], all_files: List[dict]):
        """
        Varre os itens retornados pela API do Google Drive.
        Processa pastas recursivamente e adiciona arquivos a uma lista.
        Args:
            items: Lista de itens retornados pela API do Google Drive.
            all_files: Lista para armazenar os arquivos encontrados.
        """
        for item in items:
            item_id = item.get('id')
            item_name = item.get('name', 'Nome Desconhecido')
            mime_type = item.get('mimeType')

            if mime_type == 'application/vnd.google-apps.folder':
                if item_id not in self._processed_folders:
                    print(f"Recursão -> Entrando na subpasta: '{item_name}' (ID: {item_id})")
                    sub_folder_files = self.list_files_recursively(item_id)
                    all_files.extend(sub_folder_files)
                else:
                    print(f"Aviso: Pasta '{item_name}' (ID: {item_id}) já processada, pulando.")
            else:
                print(f"Encontrado arquivo: '{item_name}' (ID: {item_id}), Mimetype: {mime_type}")
                all_files.append({'id': item_id, 'name': item_name})


    def list_files_recursively(self, folder_id: str) -> List[dict]:
        """Lista todos os arquivos dentro de uma pasta e subpastas de forma recursiva."""
        if not self.service:
            print("Erro: Serviço do Google Drive não inicializado.")
            return []

        # Garante que o atributo exista
        if not hasattr(self, '_processed_folders'):
            self._processed_folders = set()

        all_files = []
        page_token = None

        # Evita recursão infinita se houver links circulares (embora raro no Drive)
        if folder_id in self._processed_folders:
            print(f"Aviso: Pasta '{folder_id}' já visitada nesta recursão, pulando para evitar loop.")
            return []
        self._processed_folders.add(folder_id)

        print(f"--- Buscando na pasta: {folder_id} ---")

        while True:
            try:
                query = f"'{folder_id}' in parents"
                results = self.service.files().list(
                    q=query,
                    pageSize=100, # Aumentar pageSize pode reduzir chamadas API
                    fields="nextPageToken, files(id, name, mimeType)",
                    pageToken=page_token
                ).execute()

                items = results.get("files", [])
                print(f"Itens encontrados nesta página da pasta {folder_id}: {len(items)}")

                # Percorrer todos os itens encontrados.
                self._scan_items(items, all_files) # _scan_items chama list_files_recursively

                page_token = results.get('nextPageToken', None)
                if page_token is None:
                    break

            except HttpError as e:
                print(f"Erro de API ao listar itens na pasta {folder_id}: {e}")
                # Se for um erro 5xx, talvez valesse a pena tentar novamente a página
                print("Interrompendo busca nesta subpasta devido a erro.")
                break
            except Exception as e:
                print(f"Erro inesperado ao processar pasta {folder_id}: {e}")
                break

        print(f"--- Finalizando busca na pasta: {folder_id} ---")
        return all_files


def _ensure_destination_directory(destination_path: str) -> bool:
    """
    Garante que o diretório de destino exista.
    """
    try:
        os.makedirs(destination_path, exist_ok=True)
        return True
    except OSError as e:
        print(f"Erro ao criar/verificar diretório de destino '{destination_path}': {e}")
        return False

def _download_chunks(downloader: MediaIoBaseDownload, file_name: str, fh: BytesIO) -> bool:
    """
    Baixa o arquivo em chunks e lida com erros de forma robusta.
    (A lógica de retry interna do next_chunk já existe, mas podemos adicionar logging)
    """
    done = False
    chunk_num = 0
    last_progress_output_time = 0
    output_interval = 2 # Segundos entre updates de progresso no console

    while not done:
        chunk_num += 1
        status = None
        progress = 0
        try:
            status, done = downloader.next_chunk(num_retries=5)

            current_time = time.time()
            if status:
                progress = int(status.progress() * 100)
                # Limita a frequência de output no console para não poluir
                if current_time - last_progress_output_time > output_interval or done:
                    print(f"\r [Download/Export '{file_name}'] Progresso: {progress}% "
                          f"(Chunk {chunk_num} recebido)...", end="", flush=True)
                    last_progress_output_time = current_time
            else:
                 # Log menos verboso para status None
                 if current_time - last_progress_output_time > output_interval * 2 or done:
                      print(f"\r [Download/Export '{file_name}'] Chunk {chunk_num} recebido "
                            f"(status None). Done={done}", end='', flush=True)
                      last_progress_output_time = current_time

        except HttpError as chunk_error:
            print(f"\n [Download '{file_name}'] Erro HTTP persistente no chunk {chunk_num} "
                  f"(após retries internos): {chunk_error}")
            return False # Falha o download do arquivo
        except Exception as chunk_generic_error:
            print(f"\n [Download '{file_name}'] Erro genérico inesperado no chunk "
                  f"{chunk_num}: {chunk_generic_error}")
            traceback.print_exc()
            return False # Falha o download do arquivo

    # Garante que a linha de progresso seja limpa ou finalizada
    print(f"\r [Download/Export '{file_name}'] Progresso: 100% (Concluído).{' '*20}") # Limpa a linha
    return True # Download bem-sucedido

def _save_file(file_path: str, fh: BytesIO) -> bool:
    """Salva o conteúdo do BytesIO em um arquivo."""
    try:
        with open(file_path, "wb") as f:
            fh.seek(0) # Garante que estamos no início do BytesIO
            f.write(fh.read())
        return True
    except IOError as e:
        print(f"Erro de I/O ao salvar arquivo '{file_path}': {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"Erro inesperado ao salvar arquivo '{file_path}': {e}")
        traceback.print_exc()
        return False


def initialize_apis_and_db(drive_service):
    """Inicializa as APIs e o gerenciador de banco de dados."""
    # ... (código existente) ...
    print("Inicializando APIs e Gerenciador de Banco de Dados...")
    try:
        db_manager = DataBaseManager(drive_service)
        return db_manager
    except Exception as e:
        print(f"Erro durante a inicialização da API do Google Drive ou DataBaseManager: {e}")
        traceback.print_exc()
        return None

def list_drive_files(db_manager, drive_folder_id):
    """Lista os arquivos da pasta especificada no Google Drive."""
    print(f"Listando arquivos da pasta Google Drive ID: {drive_folder_id}")
    try:
        # Limpa o set de pastas processadas antes de iniciar uma nova listagem recursiva
        if hasattr(db_manager, '_processed_folders'):
            db_manager._processed_folders.clear()
        else:
            db_manager._processed_folders = set()

        files = db_manager.list_files_recursively(drive_folder_id)
        return files
    except Exception as e:
        print(f"Erro ao listar arquivos do Google Drive: {e}")
        traceback.print_exc()
        return None

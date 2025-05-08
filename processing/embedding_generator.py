import os
import io
import numpy as np
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from typing import List, Dict, Optional, Any
from TextExtractor import process_and_tokenize_file, TEMP_DOWNLOAD_FOLDER

# Importações relacionadas ao GoogleDriveAPI (necessárias para process_batch)
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError

# Define o tamanho do chunk para dividir documentos grandes antes de gerar embeddings.
# Para obter informações específicas em pequenas passagens, DOCUMENT_CHUNK_SIZE baixo.
# Para obter uma compreensão de seções maiores, DOCUMENT_CHUNK_SIZE alto.
# DOCUMENT_CHUNK_SIZE determina o máximo de tokens dentro de um chunk.
DOCUMENT_CHUNK_SIZE = 50


def _cleanup_temp_file(download_path: str, pid: int):
    """
    Remove o arquivo temporário após o processamento.
    Args:
        download_path (str): O caminho para o arquivo temporário.
        pid (int): O ID do processo atual.
    """
    if os.path.exists(download_path):
        try:
            os.remove(download_path)
            print(f"[Processo {pid}] Arquivo temporário '{download_path}' removido.")
        except Exception as e:
            print(f"[Processo {pid}] Erro ao tentar remover o arquivo temporário '{download_path}': {e}")


def _download_file(drive_service, file_id: str, file_name: str, download_path: str, pid: int):
    """
    Baixa um arquivo do Google Drive para um caminho local.
    Args:
        drive_service: O serviço da API do Google Drive.
        file_id (str): O ID do arquivo no Google Drive.
        file_name (str): O nome do arquivo.
        download_path (str): O caminho local para salvar o arquivo.
        pid (int): O ID do processo atual.
    """
    print(f"[Processo {pid}] Tentando baixar '{file_name}' (ID: {file_id}) para '{download_path}'")
    try:
        # Prepara a requisição para baixar o conteúdo do arquivo
        request = drive_service.files().get_media(fileId=file_id)
        # Usa um buffer em memória para receber os dados do download
        fh = io.BytesIO()
        # Cria o objeto downloader
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while not done:
            # Baixa o próximo chunk do arquivo
            status, done = downloader.next_chunk()
            if status:
                # Exibe o progresso do download
                print(f"\r[Processo {pid}] Baixando '{file_name}': {int(status.progress() * 100)}%...", end='')
        print(f"\r[Processo {pid}] Download de '{file_name}' concluído.")

        # Escreve o conteúdo baixado (do buffer em memória) para o arquivo local
        with open(download_path, "wb") as f:
            f.write(fh.getvalue())
        print(f"[Processo {pid}] Arquivo '{file_name}' salvo em '{download_path}'.")

    except HttpError as error:
        print(f"[Processo {pid}] Erro de API do Google ao processar '{file_name}' (ID: {file_id}): {error}")
        raise
    except Exception as e:
        print(f"[Processo {pid}] Erro inesperado ao baixar '{file_name}' (ID: {file_id}): {e}")
        raise


class EmbeddingGenerator:
    """
    Gera embeddings de texto usando modelos Transformer (BERT) via TensorFlow.
    Projetada para ser usada em um fluxo que baixa arquivos, extrai texto,
    tokeniza e então gera vetores de embedding para chunks de texto.
    """
    def __init__(self, model_name='bert-base-uncased', batch_size=32, output_dir='embeddings_tf'):
        """
        Inicializa o gerador de embeddings com o modelo TensorFlow, gerando embeddings a partir de tokens.
        Args:
            model_name (str): O nome do modelo Transformer pré-treinado a ser usado.
            batch_size (int): O número de sequências a serem processadas por lote.
            output_dir (str): O diretório onde os vetores de embedding serão salvos.
        """
        print(f"[Processo {os.getpid()}] Inicializando EmbeddingGenerator com modelo: {model_name}")

        # Carrega o tokenizador específico do modelo BERT
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # Carrega o modelo BERT pré-treinado
        self.model = TFBertModel.from_pretrained(model_name)
        # self.batch_size = batch_size
        # Diretório para salvar os embeddings gerados
        self.output_dir = output_dir
        # Cria o diretório de saída se ele não existir
        os.makedirs(self.output_dir, exist_ok=True)

        # Verifica se a GPU está disponível e a usa, caso contrário usa a CPU
        if tf.config.list_physical_devices('GPU'):
            print(f"[Processo {os.getpid()}] GPU encontrada. Usando GPU para geração de embeddings.")
            self.device = '/GPU:0'
        else:
            print(f"[Processo {os.getpid()}] Nenhuma GPU encontrada. Usando CPU para geração de embeddings.")
            self.device = '/CPU:0'

        print(f"[Processo {os.getpid()}] EmbeddingGenerator inicializado.")

    def generate_embeddings(self,
                            token_chunk: List[str],
                            filename_prefix: str = "document_chunk") -> Optional[str]:
        """
        Gera embeddings para um único chunk (lista) de tokens usando o modelo BERT.
        Args:
            token_chunk (List[str]): Uma lista de tokens representando um segmento do documento.
            filename_prefix (str): Prefixo para o nome do arquivo .npy onde os embeddings serão salvos.
        Returns:
            Optional[str]: O caminho para o arquivo onde os embeddings foram salvos.
        """
        if not token_chunk:
            print(f"[Processo {os.getpid()}] Aviso: Recebido chunk de tokens vazio para '{filename_prefix}'. Pulando.")
            return None

        # Adiciona os tokens especiais [CLS] no início e [SEP] no final.
        tokens_with_special = ['[CLS]'] + token_chunk + ['[SEP]']

        try:
            # Converte a lista de tokens em IDs de input que o modelo entende.
            inputs = self.tokenizer(tokens_with_special,
                                    return_tensors="tf", # Retorna tensores do TensorFlow
                                    padding=True, # Preenche sequências mais curtas no lote
                                    truncation=True, # Trunca sequências mais longas que max_length.
                                    max_length=512, # Limite comum para o modelo.
                                    is_split_into_words=True)

            # Executa a inferência do modelo dentro do contexto do dispositivo configurado (CPU/GPU)
            with tf.device(self.device):
                # Passa os inputs tokenizados para o modelo.
                outputs = self.model(**inputs, output_hidden_states=True)

                # Pega o embedding do primeiro token ([CLS]) como representação do chunk inteiro.
                cls_embedding = outputs.hidden_states[-1][:, 0, :].numpy()

            # Define o nome do arquivo de saída para o embedding deste chunk
            output_filename = os.path.join(self.output_dir, f"{filename_prefix}_embedding.npy")
            # Salva o embedding (que é um array numpy) no arquivo .npy
            np.save(output_filename, cls_embedding)
            print(f"[Processo {os.getpid()}] Embedding para '{filename_prefix}' salvo em: {output_filename}")
            return output_filename

        except Exception as e:
            print(f"[Processo {os.getpid()}] Erro ao gerar embedding para '{filename_prefix}': {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def _process_document_chunk(self,
                                file_name: str,
                                tokens: List[str],
                                chunk_index: int,
                                num_chunks: int,
                                embeddings_data: List[Dict[str, Any]],
                                pid: int):
        """
        Processa um único chunk de um documento, gerando seu embedding e armazenando os metadados.
        Args:
            file_name (str): O nome do arquivo original.
            tokens (List[str]): A lista de tokens do documento.
            chunk_index (int): O índice do chunk atual.
            num_chunks (int): O número total de chunks no documento.
            embeddings_data (List[Dict[str, Any]]): A lista para armazenar os metadados dos embeddings.
            pid (int): O ID do processo atual.
        """
        # Define os índices de início e fim para o chunk atual
        start_index = chunk_index * DOCUMENT_CHUNK_SIZE
        end_index = min((chunk_index + 1) * DOCUMENT_CHUNK_SIZE, len(tokens))
        # Extrai os tokens para o chunk atual
        chunk_tokens = tokens[start_index:end_index]

        if chunk_tokens:
            # Cria um prefixo de nome de arquivo único para o embedding deste chunk
            embedding_filename_prefix = f"{os.path.splitext(file_name)[0]}_part_{chunk_index}"
            print(f"[Processo {pid}] Gerando embedding para '{file_name}' chunk {chunk_index+1}/{num_chunks}...")

            # Chama o método generate_embeddings para gerar o embedding para o chunk específico.
            embedding_path = self.generate_embeddings(chunk_tokens, embedding_filename_prefix)

            # Se o embedding foi gerado com sucesso, adiciona seus metadados à lista de resultados
            if embedding_path:
                embeddings_data.append({"filename": file_name, "chunk_id": chunk_index, "embedding_path": embedding_path})
        else:
            print(f"[Processo {pid}] Aviso: Chunk {chunk_index} de '{file_name}' está vazio após slicing. Pulando.")

    def _process_downloaded_file(self,
                                 file_name: str,
                                 download_path: str,
                                 embeddings_data: List[Dict[str, Any]],
                                 pid: int):
        """
        Processa um arquivo baixado, extraindo texto, tokenizando e gerando embeddings para cada chunk.
        Args:
            file_name (str): O nome do arquivo original.
            download_path (str): O caminho para o arquivo baixado.
            embeddings_data (List[Dict[str, Any]]): A lista para armazenar os metadados dos embeddings.
            pid (int): O ID do processo atual.
        """
        print(f"[Processo {pid}] Processando e tokenizando '{file_name}'...")
        # Extrai texto e tokeniza usando a função do TextExtractor
        processed_filename, tokens = process_and_tokenize_file(download_path)

        if tokens:
            print(f"[Processo {pid}] Texto extraído e tokenizado de '{processed_filename}'"
                  f"({len(tokens)} tokens). Dividindo em chunks...")
            # Calcula o número de chunks necessários com base no tamanho definido
            num_chunks = (len(tokens) + DOCUMENT_CHUNK_SIZE - 1) // DOCUMENT_CHUNK_SIZE

            # Processa cada chunk do documento
            for i in range(num_chunks):
                self._process_document_chunk(file_name, tokens, i, num_chunks, embeddings_data, pid)
        else:
            # Caso não seja possível extrair texto
            print(f"[Processo {pid}] Não foi possível extrair/tokenizar texto de '{file_name}'.")

    def process_batch(self, batch_files: List[Dict[str, str]]) -> list[Any] | None:
        """
        Processa um lote (batch) de arquivos, extrai texto, tokeniza, divide em chunks e gera embeddings.
        É executada para cada processo filho no pool de multiprocessamento.
        Args:
            batch_files (List[Dict[str, str]]): Uma lista de dicionários, onde cada dicionário
                                                contém 'id' e 'name' de um arquivo.
         Returns:
            List[Dict[str, Any]]: Uma lista de dicionários, cada um contendo informações
                                  sobre um embedding de chunk gerado
        """
        from Authentication import GoogleDriveAPI
        # Obtém o ID do processo atual para logging
        pid = os.getpid()
        print(f"[Processo {pid}] Iniciando processamento de lote com {len(batch_files)} arquivos.")

        # Cria uma instância da API do Google Drive DENTRO do processo filho.
        try:
            drive_api = GoogleDriveAPI()
            drive_service = drive_api.service
            if not drive_service:
                print(f"[Processo {pid}] Erro: Falha ao inicializar o serviço do Google Drive.")
                return [] # Retorna lista vazia se a autenticação falhar
        except Exception as auth_error:
            print(f"[Processo {pid}] Erro crítico ao autenticar Google Drive API: {auth_error}")
            return []

        embeddings_data = [] # Lista para armazenar os resultados do lote

        # Itera sobre cada arquivo no lote atribuído ao processo
        for file_info in batch_files:
            file_id = file_info.get('id')
            file_name = file_info.get('name')

            # Valida as informações do arquivo
            if not file_id or not file_name:
                print(f"[Processo {pid}] Informação de arquivo inválida encontrada: {file_info}. Pulando.")
                continue

            # Caminho local onde o arquivo será baixado temporariamente
            download_path = os.path.join(TEMP_DOWNLOAD_FOLDER, f"{pid}_{file_name}")

            try:
                _download_file(drive_service, file_id, file_name, download_path, pid)

                # Processamento do arquivo baixado
                self._process_downloaded_file(file_name, download_path, embeddings_data, pid)

            except FileNotFoundError:
                print(f"[Processo {pid}] Erro: Arquivo temporário '{download_path}' não encontrado durante processamento.")
            except Exception as e:
                print(f"[Processo {pid}] Erro inesperado ao processar '{file_name}' (ID: {file_id}): {e}")
                import traceback
                print(traceback.format_exc())
            finally:
                _cleanup_temp_file(download_path, pid)
                return None

        # Retorna a lista de metadados dos embeddings gerados neste lote
        print(f"[Processo {pid}] Finalizado processamento do lote. {len(embeddings_data)} embeddings gerados.")
        return embeddings_data
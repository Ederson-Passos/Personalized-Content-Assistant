import os
from pathlib import Path

def check_directory_existence(temp_dir):
    """
    Checks if a directory exists and creates it if it doesn't.

    Args:
        temp_dir (Path): caminho para checar/criar o diretório.
    Raises:
        OSError: If there is an error creating the directory.
    """
    if not temp_dir.exists():  # Checa se o diretório existe.
        print(f"Diretório '{temp_dir}' não encontrado. Criando...")
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)  # Criado o diretório, incluindo pais se necessário.
            print(f"Diretório '{temp_dir}' criado com sucesso.")
        except OSError as e:
            raise OSError(f"Erro ao criar o diretório '{temp_dir}': {e}") from e
    elif not temp_dir.is_dir():  # Checa se não é um diretório.
        print(f"Erro: '{temp_dir}' existe, mas não é um diretório.")
    else:
        print(f"Diretório '{temp_dir}' já existe.")

def cleanup_temp_files(file_paths, temp_dir):
    """
    Função auxiliar para limpar arquivos e diretório temporários.
    """
    print("\nIniciando limpeza dos arquivos temporários...")
    if file_paths is None:
        file_paths = []

    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"  Arquivo temporário removido: {file_path}")
        except Exception as e:
            print(f"  Erro ao deletar arquivo temporário {file_path}: {e}")
    try:
        # Tenta remover o diretório apenas se ele estiver vazio
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
            print(f"Diretório temporário removido: {temp_dir}")
        elif os.path.exists(temp_dir):
            remaining_files = os.listdir(temp_dir)
            if remaining_files:
                print(f"Diretório temporário {temp_dir} não está vazio, não será removido.")
            else:
                os.rmdir(temp_dir)
                print(f"Diretório temporário removido: {temp_dir}")

    except Exception as e:
        print(f"Erro ao remover diretório temporário {temp_dir}: {e}")
    print("Limpeza concluída.")


def cleanup_temp_folder(download_folder):
    """Limpa o diretório temporário de download."""
    for filename in os.listdir(download_folder):
        file_path = os.path.join(download_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Erro ao remover {file_path}: {e}")
    if os.path.exists(download_folder):
        os.rmdir(download_folder)
        print(f"Diretório temporário '{download_folder}' limpo.")

def create_directories(report_dir, temp_dir):
    """Cria os diretórios necessários para relatórios e arquivos temporários."""
    print("Criando diretórios necessários...")
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
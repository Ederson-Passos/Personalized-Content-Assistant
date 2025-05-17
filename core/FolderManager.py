import os
import shutil

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


def cleanup_temp_folder(download_folder_path: Path):
    """Limpa o diretório temporário de download. Remove arquivos e subdiretórios."""
    if not download_folder_path.exists():
        print(f"Diretório '{download_folder_path}' não encontrado. Nada a limpar.")
        return

    if not download_folder_path.is_dir():
        print(f"'{download_folder_path}' não é um diretório. Não é possível limpar.")
        return

    print(f"Limpando conteúdo do diretório: {download_folder_path}")
    for item in download_folder_path.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
                print(f"  Arquivo removido: {item}")
            elif item.is_dir():
                shutil.rmtree(item)
                print(f"  Subdiretório removido: {item}")
        except Exception as e:
            print(f"  Erro ao remover {item}: {e}")

    try:
        download_folder_path.rmdir()
        print(f"Diretório '{download_folder_path}' removido (após limpeza de conteúdo).")
    except OSError:
        print(f"Diretório '{download_folder_path}' não está vazio ou erro ao remover (após limpeza de conteúdo).")

def create_directories(report_dir, temp_dir):
    """Cria os diretórios necessários para relatórios e arquivos temporários."""
    print("Criando diretórios necessários...")
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

def clear_and_ensure_path(directory_path_str: str):
    """
    Garante que um diretório esteja limpo e exista.
    Se o diretório existir, todo o seu conteúdo (arquivos e subdiretórios) será removido.
    Se o diretório não existir, ele será criado.
    Args:
        directory_path_str (str): O caminho para o diretório.
    """
    directory_path = Path(directory_path_str)

    if directory_path.exists():
        if not directory_path.is_dir():
            print(f"Erro: '{directory_path}' existe, mas não é um diretório. Não é possível limpar.")
            return

        print(f"Diretório '{directory_path}' encontrado. Limpando conteúdo...")
        for item in directory_path.iterdir():
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                print(f"  Erro ao remover {item}: {e}")
        print(f"Conteúdo do diretório '{directory_path}' limpo.")
    else:
        print(f"Diretório '{directory_path}' não encontrado. Criando...")
        try:
            directory_path.mkdir(parents=True, exist_ok=True)
            print(f"Diretório '{directory_path}' criado com sucesso.")
        except OSError as e:
            print(f"Erro ao criar o diretório '{directory_path}': {e}")
            raise
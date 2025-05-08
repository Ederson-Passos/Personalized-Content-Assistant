import os.path

from google.auth.transport.requests import Request
from google.oauth2 import credentials
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Define os escopos de permissão necessários para acessar o Google Drive.
SCOPES = ["https://www.googleapis.com/auth/drive"]
# Nome do arquivo que contém as credenciais da API do Google Cloud (OAuth 2.0 client ID).
CREDENTIALS_FILE = "credentials.json"
# Nome do arquivo onde o token de acesso do usuário será armazenado após a autenticação.
TOKEN_FILE = "token.json"


def get_service():
    creds = None
    # O arquivo token.json armazena os tokens de acesso e atualização do usuário,
    # e é criado automaticamente quando o fluxo de autorização é concluído
    # pela primeira vez.
    if os.path.exists('token.json'):
        creds = credentials.Credentials.from_authorized_user_file('token.json', ['https://www.googleapis.com/auth/drive'])
    # Se não houver credenciais (ou se forem inválidas), deixe o usuário fazer o login.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())  # A biblioteca tenta atualizar o token automaticamente.
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', ['https://www.googleapis.com/auth/drive'])
            creds = flow.run_local_server(port=0)
        # Salva as credenciais para a próxima execução
        with open('token.json', 'w') as token:
            token.write(creds.to_json())


    service = build('drive', 'v3', credentials=creds)
    return service

def _authenticate():
    """
    Autentica o usuário e retorna um objeto de serviço do Google Drive API.
    Retorna um objeto "resource" do Google API Client Library para interagir com o Drive.
    """
    creds = None
    # Verifica se o arquivo token.json existe e carrega as credenciais.
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    # Se não houver credenciais válidas inicia o fluxo de autenticação.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        # Salva as credenciais para futuras execuções.
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())

    # Constrói e retorna o objeto de serviço do Google Drive API.
    return build("drive", "v3", credentials=creds)


class GoogleDriveAPI:
    """
    Classe para autenticar e obter o serviço da API do Google Drive v3.
    """
    def __init__(self):
        """Inicializa a classe e inicia o processo de autenticação."""
        self.service = _authenticate()


if __name__ == "__main__":
    drive_api = GoogleDriveAPI()
    print("Serviço do Google Drive autenticado com sucesso (dentro de Authentication.py).")
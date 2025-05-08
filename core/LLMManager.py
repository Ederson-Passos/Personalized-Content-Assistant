import os
import time
import traceback
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai


class GeminiLLM:
    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise ValueError("A variável de ambiente GOOGLE_API_KEY não está definida.")
        # Configura a API do Google Generative AI (necessário para contagem de tokens também)
        try:
            genai.configure(api_key=self.google_api_key)
            print("API Google Generative AI configurada com sucesso.")
        except Exception as e:
            print(f"Erro ao configurar a API Google Generative AI: {e}")
            raise RuntimeError("Falha ao configurar a API do Google. Verifique a chave GOOGLE_API_KEY.")

        try:
            self.chat_llm = ChatGoogleGenerativeAI(
                model="gemini/gemini-1.5-flash-latest",
                google_api_key=self.google_api_key,
                safety_settings={
                     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                 },
                convert_system_message_to_human=True
            )
            # Cria um modelo separado apenas para contagem de tokens
            self._token_counter_model = genai.GenerativeModel('gemini-1.5-flash-latest')
            print("Modelo ChatGoogleGenerativeAI (gemini-1.5-flash-latest) inicializado.")
        except Exception as e:
            print(f"Erro ao inicializar o ChatGoogleGenerativeAI: {e}")
            traceback.print_exc()
            raise RuntimeError("Não foi possível inicializar o modelo Gemini.")

        # Lógica de limite de taxa baseada em requisições por minuto
        self.requests_feitos_no_minuto = 0
        self.inicio_do_minuto = time.time()
        # Limite da API Gemini Gratuita: 15 requisições por minuto
        self.limite_requisicoes_por_minuto = 15
        self.max_input_tokens_model = 60000

    def contador_de_tokens(self, texto: str) -> int:
        """Realiza a contagem de tokens usando a API Gemini."""
        if not texto:
            return 0
        try:
            # Certifica que _token_counter_model foi inicializado
            if hasattr(self, '_token_counter_model') and self._token_counter_model:
                response = self._token_counter_model.count_tokens(texto)
                return response.total_tokens
            else:
                print("Aviso: Modelo de contagem de tokens não inicializado. Usando contagem de palavras.")
                return len(texto.split())
        except Exception as e:
            print(f"Erro ao contar tokens com a API Gemini: {e}. Usando contagem de palavras.")
            return len(texto.split())

    def truncate_content(self, content: str, current_tokens: int, max_tokens: int, batch_number: int) -> str:
        """
        Trunca o conteúdo se a contagem de tokens exceder o máximo permitido para a entrada do modelo.
        Como não temos um tokenizer local preciso para Gemini, usamos uma estratégia baseada em caracteres.
        Args:
            content: O texto completo do arquivo.
            current_tokens: A contagem de tokens já calculada para o conteúdo (via API Gemini).
            max_tokens: O número máximo de tokens que a LLM deve receber na entrada.
            batch_number: O número do lote atual (para fins de log).
        Returns:
            O conteúdo original ou uma versão truncada com um aviso.
        """
        if current_tokens > max_tokens:
            print(f"      [Lote {batch_number}] ALERTA: Conteúdo excedeu o limite de tokens de entrada "
                  f"({current_tokens} > {max_tokens}). Truncando...")

            # Estima caracteres por token. Se current_tokens for 0, usa um fallback.
            estimated_chars_per_token = len(content) / current_tokens if current_tokens > 0 else 3.5 # Média para PT/EN
            # Calcula o número máximo de caracteres, com uma margem de segurança (95%)
            max_chars = int(max_tokens * estimated_chars_per_token * 0.95)
            truncated_content = content[:max_chars]

            # Recalcula os tokens do conteúdo truncado para log (opcional, mas útil)
            new_token_count = self.contador_de_tokens(truncated_content)
            print(f"      [Lote {batch_number}] Conteúdo truncado para aproximadamente {new_token_count} tokens "
                  f"(baseado em {max_chars} caracteres).")
            print(f"      [Lote {batch_number}] Conteúdo truncado baseado em caracteres ({max_chars} caracteres).")

            # Adiciona um aviso claro no final do conteúdo truncado
            truncation_warning = "\n\n[... CONTEÚDO TRUNCADO DEVIDO AO LIMITE DE TOKENS DE ENTRADA ...]"
            # Garante que o aviso caiba, removendo parte do final se necessário
            if len(truncated_content) + len(truncation_warning) > max_chars: # Evita crescer o conteúdo
                # Remove espaço suficiente para o aviso
                truncated_content = truncated_content[:max_chars - len(truncation_warning) - 1]

            truncated_content += truncation_warning

            return truncated_content
        else:
            # Se não excedeu o limite, retorna o conteúdo original
            return content

    def executar_solicitacao(self, prompt: str) -> Optional[str]:
        """Executa uma solicitação ao modelo Gemini, gerenciando o limite de taxa."""

        # 1. Verificar Limite de Taxa (antes de fazer a chamada)
        tempo_para_esperar = self.verificar_limite_de_taxa()
        if tempo_para_esperar > 0:
            print(f"Limite de requisições por minuto ({self.limite_requisicoes_por_minuto} RPM) atingido. Esperando"
                  f" {tempo_para_esperar:.1f} segundos.")
            time.sleep(tempo_para_esperar)
            # Após esperar, o contador é resetado dentro de verificar_limite_de_taxa se necessário.

        # 2. Contar tokens de entrada (após a espera, para log e possível truncamento final)
        tokens_na_solicitacao = self.contador_de_tokens(prompt)
        print(f"Estimativa de tokens de entrada (prompt): {tokens_na_solicitacao}")

        try:
            print(f"Enviando prompt para Gemini ({tokens_na_solicitacao} tokens estimados)...")
            start_time = time.time()
            resposta = self.chat_llm.invoke(prompt)
            end_time = time.time()
            print(f"Resposta recebida do Gemini em {end_time - start_time:.2f} segundos.")

            # 3. Atualizar o contador de requisições (após chamada bem-sucedida)
            self.atualizar_contador_requisicoes()

            # Verifica se a resposta e o conteúdo são válidos
            if resposta and isinstance(resposta.content, str):
                response_content = resposta.content
                # Estima os tokens da resposta para log
                tokens_resposta_estimados = self.contador_de_tokens(response_content)
                print(f"Resposta contém {tokens_resposta_estimados} tokens estimados.")
                print(f"Total estimado (prompt+resposta): {tokens_na_solicitacao + tokens_resposta_estimados} tokens.")
                return response_content # Retorna o conteúdo da string
            else:
                # Caso onde resposta.content não é uma string (pode ser None, etc.)
                print(f"Aviso: A resposta do LLM Gemini não continha conteúdo de texto válido ou foi None. "
                      f"Resposta: {resposta}")
                return None

        except Exception as e:
            print(f"Erro ao executar solicitação ao Gemini: {e}")
            if "ResourceExhaustedError" in str(e):
                print("Erro de 'ResourceExhaustedError': Pode ter excedido um limite de cota (RPM, TPM?). "
                      "Verifique o console do Google Cloud.")
            elif "response was blocked" in str(e).lower() or "safety settings" in str(e).lower():
                print("Erro de Segurança: A resposta do Gemini foi bloqueada devido às configurações de segurança. "
                      "Considere ajustar 'safety_settings' na inicialização do ChatGoogleGenerativeAI se apropriado.")

            traceback.print_exc()
            return None

    def verificar_limite_de_taxa(self) -> float:
        """Verifica se o limite de requisições por minuto foi atingido."""
        tempo_atual = time.time()
        tempo_decorrido = tempo_atual - self.inicio_do_minuto

        if tempo_decorrido >= 60:  # Passou um minuto?
            print(f"Resetando contador de RPM. {self.requests_feitos_no_minuto} reqs no último minuto.")
            self.inicio_do_minuto = tempo_atual
            self.requests_feitos_no_minuto = 0
            return 0.0  # Não precisa esperar

        # Verifica se a próxima requisição ultrapassaria o limite
        if self.requests_feitos_no_minuto >= self.limite_requisicoes_por_minuto:
            # Calcula quanto tempo falta para completar o minuto atual
            tempo_para_esperar = 60.0 - tempo_decorrido
            return max(0.1, tempo_para_esperar) # Espera pelo menos um pouco
        else:
            # Ainda há espaço no limite para esta solicitação
            return 0.0

    def atualizar_contador_requisicoes(self):
        """Atualiza o contador de requisições feitas no minuto atual."""
        tempo_atual = time.time()
        # Só incrementa se ainda estiver dentro da janela de 1 minuto
        if tempo_atual - self.inicio_do_minuto < 60:
            self.requests_feitos_no_minuto += 1
            print(f"Reqs neste minuto: {self.requests_feitos_no_minuto}/{self.limite_requisicoes_por_minuto}")
        else:
            # Se passou mais de um minuto entre a verificação e a atualização,
            # reinicia o contador com 1 (esta requisição atual).
            print(f"Resetando contador de RPM (atrasado). 1 req neste novo minuto.")
            self.inicio_do_minuto = tempo_atual
            self.requests_feitos_no_minuto = 1


# Função de setup
def setup_llm_manager():
    """Configura e retorna o gerenciador do LLM (agora Gemini)."""
    try:
        llm_manager = GeminiLLM()
        print("Gerenciador LLM (Gemini) configurado com sucesso.")
        return llm_manager
    except ValueError as e:
        print(f"Erro de configuração: {e}")
        return None
    except Exception as e:
        print(f"Ocorreu um erro inesperado ao inicializar o LLM Gemini: {e}")
        traceback.print_exc()
        return None

# Função de inicialização adaptada
def initialize_llm():
    """Inicializa o LLM Manager e retorna a instância do LLM Gemini."""
    print("Inicializando o LLM Manager (Gemini)...")
    llm_manager = setup_llm_manager() # Chama a função de setup
    if llm_manager is None:
        print("Falha ao inicializar o LLM Manager. Encerrando.")
        return None, None

    try:
        # Verifica se o atributo llm existe e é válido
        if hasattr(llm_manager, 'chat_llm') and llm_manager.chat_llm:
            gemini_chat_llm = llm_manager.chat_llm
            print("Instância do ChatGoogleGenerativeAI LLM obtida com sucesso.")
            # Retorna o manager e o llm.
            return llm_manager, gemini_chat_llm
        else:
            print("Erro: O LLM Manager foi inicializado, mas a instância do LLM (ChatGoogleGenerativeAI) "
                  "não está disponível.")
            traceback.print_exc()
            return None, None
    except AttributeError:
        print("Erro: O objeto retornado por setup_llm_manager() não possui o atributo 'llm'.")
        traceback.print_exc()
        return None, None
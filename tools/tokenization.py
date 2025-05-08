import re
import nltk
from nltk.tokenize import word_tokenize

# Verificando se o tokenizador punkt está baixado
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Verificando se o recurso punkt_tab está baixado
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

def remove_special_characters(text):
    """Remove caracteres especiais do texto, preservando letras (incluindo acentuadas),
    números, espaços e sinais diacríticos comuns em inglês."""
    text = re.sub(r"[^a-zA-Z0-9áàâãéèêíïóôõúüçñÁÀÂÃÉÈÊÍÏÓÔÕÚÜÇÑ\s\-\']", "", text)
    return text

def convert_to_lowercase(text):
    """Converte o texto para minúsculas."""
    return text.lower()

def tokenize_text(text):
    """Tokeniza o texto em palavras."""
    return word_tokenize(text)

def preprocess_text(text):
    """Realiza o pré-processamento completo do texto."""
    tokens = tokenize_text(convert_to_lowercase(remove_special_characters(text)))
    return tokens

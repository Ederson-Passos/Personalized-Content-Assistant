def create_document_analysis_tasks(file_path: str, llm: Any) -> List[Task]:
    """
    Cria tarefas para contar palavras e resumir o CONTEÚDO FORNECIDO de um documento.
    Args:
        file_path (str): O caminho original do arquivo (usado para referência no output).
        llm (Any): A instância do LLM a ser usada pelo agente.
    Returns:
        List[Task]: Uma lista contendo a tarefa de análise do documento.
    """
    document_analyzer = DocumentAnalysisAgent(llm=llm)

    analysis_task = Task(
        description=(
            f"Analise o conteúdo do documento fornecido (originalmente de '{file_path}').\n"
            f"1. Use a ferramenta 'count_words' para contar o número total de palavras no texto fornecido.\n"
            f"2. Use suas capacidades de raciocínio (LLM) para gerar um resumo conciso "
            f"(5-10 linhas) da mensagem principal do texto fornecido.\n"
            "Certifique-se de que o resultado final seja um dicionário Python estruturado."
        ),
        agent=document_analyzer,
        expected_output=(
            "Um dicionário Python contendo:\n"
            f"- 'file_path': O caminho original do arquivo '{file_path}' (string).\n"
            "- 'word_count': A contagem total de palavras do texto fornecido (integer ou string de erro).\n"
            "- 'summary': Um resumo conciso do texto fornecido ou uma mensagem de erro se a análise falhou (string)."
        )
    )
    return [analysis_task]


def create_reporting_tasks(llm: Any) -> List[Task]:
    """
    Cria tarefas para consolidar resultados e instruir o ReportingAgent a salvar um relatório.
    O diretório específico será injetado dinamicamente na descrição/expected_output.
    Args:
        llm (Any): A instância do LLM a ser usada pelo agente.
    Returns:
        List[Task]: Uma lista contendo a tarefa de geração de relatório.
    """
    # Instancie o agente passando o LLM
    reporting_agent = ReportingAgent(llm=llm)

    # Tarefa para gerar e salvar o relatório consolidado
    generate_save_report_task = Task(
        description=(
            "Consolide TODAS as descobertas das tarefas anteriores de análise de documentos "
            "(fornecidas como contexto). Para cada documento analisado, extraia as informações relevantes "
            "(como nome do arquivo implícito na descrição da tarefa de análise, contagem de palavras e resumo) "
            "do resultado da respectiva tarefa de análise.\n"
            "IMPORTANTE: Ao processar os resumos ('summary') de cada tarefa de análise, certifique-se de que "
            "eles mantenham a formatação de parágrafos original. Se um resumo parecer um bloco único, "
            "tente adicionar quebras de parágrafo (duplo newline '\\n\\n') onde fizer sentido semanticamente.\n"
            "Estruture todas essas descobertas consolidadas em um único dicionário Python. As chaves podem ser "
            "nomes de arquivos ou identificadores, e os valores devem ser os dicionários de resultados de cada análise "
            "(contendo 'file_path', 'word_count', 'summary' formatado, etc.).\n"
            "Finalmente, use a ferramenta 'save_analysis_report' para salvar este dicionário consolidado.\n"
            "Ao usar a ferramenta 'save_analysis_report':\n"
            "1. O primeiro argumento DEVE ser o dicionário Python com os resultados consolidados e formatados.\n"
            "2. O segundo argumento, chamado 'report_directory', DEVE ser a seguinte string de caminho: "
            "'{specific_report_dir_placeholder}'." # Placeholder para o caminho específico
        ),
        agent=reporting_agent,
        expected_output=(
            "A confirmação textual retornada pela ferramenta 'save_analysis_report' indicando se o relatório "
            "foi salvo com sucesso e o caminho completo do arquivo .txt gerado.\n"
            "Exemplo de sucesso: 'Report successfully saved to: "
            "google_drive_reports/run_.../batch_analysis_report_....txt'\n"
            "A resposta DEVE incluir o caminho completo do arquivo salvo retornado pela ferramenta, "
            "que deve estar dentro do diretório: '{specific_report_dir_placeholder}'."
        )
    )
    return [generate_save_report_task]


def create_data_mining_tasks(llm: Any) -> List[Task]:
    """
    Cria tarefas para analisar os resultados das análises de documentos (passados como contexto).
    Args:
        llm (Any): A instância do LLM a ser usada pelo agente.
    Returns:
        List[Task]: Uma lista contendo a tarefa de mineração de dados.
    """
    # Instancie o agente passando o LLM
    data_mining_agent = DataMiningAgent(llm=llm)

    # Tarefa para analisar resultados agregados (passados via contexto pela Crew)
    mining_task = Task(
        description=(
            "Analise os resultados coletados (contagens de palavras, resumos) das tarefas anteriores de "
            "análise de documentos (fornecidos como contexto). "
            "Identifique quaisquer tendências, padrões, anomalias ou métricas chave notáveis entre os documentos. "
            "Concentre-se em comparar contagens de palavras e resumir temas comuns ou discrepâncias "
            "encontradas nos resumos. "
            "Use a ferramenta 'count_words' se precisar recontar ou verificar partes específicas "
            "do texto no contexto."
        ),
        agent=data_mining_agent,
        expected_output=(
            "Um relatório de análise textual detalhando:\n"
            "- Tendências ou padrões observados nas contagens de palavras entre documentos.\n"
            "- Anomalias ou outliers nas contagens de palavras.\n"
            "- Temas comuns ou descobertas chave sintetizadas a partir dos resumos dos documentos.\n"
            "- Quaisquer outras métricas ou insights relevantes derivados dos dados combinados."
        )
    )
    return [mining_task]
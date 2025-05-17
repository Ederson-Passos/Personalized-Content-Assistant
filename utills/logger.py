import logging
import os
from rich.logging import RichHandler

FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
RICH_HANDLER = RichHandler(
    rich_tracebacks=True,
    tracebacks_show_locals=True,
    markup=True,
    keywords=[
        "INFO", "WARNING", "ERROR", "CRITICAL", "DEBUG",
        "Starting", "Finished", "Error", "Success",
        "Initializing", "Configuring", "Downloading",
        "Processing", "Indexing", "Query", "Tool",
        "Agent", "Task", "Crew", "Result", "Timeout",
        "Missing", "Failed", "Skipping", "Found",
        "Retrieved", "Summarizing", "Running", "Cleaning",
        "Batch", "Lote", "Thread", "WARNING", "ERROR"
    ]
)

def setup_logger(logger_name: str, log_file: str, log_level=logging.INFO):
    """
    Configures a logger with output to console (Rich) and file.
    Deletes the existing log file before configuring the FileHandler.
    """
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    if os.path.exists(log_file):
        try:
            os.remove(log_dir)

        except OSError as e:
            print(f"Warning: Could not remove existing log file {log_file}: {e}")

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(FORMAT))
    logger.addHandler(file_handler)

    if not any(isinstance(h, RichHandler) for h in logger.handlers):
        logger.addHandler(RICH_HANDLER)

    return logger
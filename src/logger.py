"""
Project: Stock Price Predictor v2
Package: src
Filename: logger.py
Description: Centralized system telemetry and logging module. Configures customized 
             stream handlers with ANSI terminal color formatting alongside persistent 
             asynchronous thread-safe file logging contexts to capture application execution paths.
Author: Kartik Kant (AI/ML Engineer)
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

# ==========================================
# TELEMETRY LOG FORMAT SCHEMAS & ANSI COLORS
# ==========================================
CONSOLE_FORMAT: str = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)

FILE_FORMAT: str = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(funcName)s | %(message)s"
)

COLORS: dict = {
    "DEBUG": "\033[36m",     
    "INFO": "\033[32m",      
    "WARNING": "\033[33m",   
    "ERROR": "\033[31m",     
    "CRITICAL": "\033[35m",  
    "RESET": "\033[0m"       
}


# ==========================================
# CUSTOM CONSOLE STREAM FORMATTER
# ==========================================
class ColoredFormatter(logging.Formatter):
    """
    Custom stream formatter extension that maps ANSI color escapes onto log level names.
    Enhances visibility of operational tracking thresholds within local interactive terminal outputs.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Intercepts incoming logging records to inject color-coded string buffers before streaming to standard output.
        """
        color = COLORS.get(record.levelname, COLORS["RESET"])
        reset = COLORS["RESET"]

        # Enclose the standard log level text within ANSI color codes
        record.levelname = f"{color}{record.levelname}{reset}"

        return super().format(record)


# ==========================================
# LOGGER SUBSYSTEM CORE FACTORY
# ==========================================
def get_logger(
    name: str,
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_dir: Optional[str] = None
) -> logging.Logger:
    """
    Constructs or retrieves a singleton tracking stream instance mapped to a specific namespace scope.
    Instantiates synchronized console stream interfaces and atomic local disk output handlers.

    Args:
        name (str): Unique identifying key tracking the system origin module instance.
        level (int): Central parameter setting the baseline visibility filter threshold.
        log_to_file (bool): Conditional flag checking whether to mirror console logs onto file systems.
        log_dir (Optional[str]): Target string filepath map targeting the destination log output directory.

    Returns:
        logging.Logger: Fully configured context telemetry tracker pipeline object.
    """
    logger = logging.getLogger(name)

    # Check for pre-existing handler registrations to safeguard against duplicate message pipelines
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Configure stdout stream pipeline routes for runtime container tracking
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = ColoredFormatter(CONSOLE_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Core synchronization routines processing secondary file backup writes
    if log_to_file:
        if log_dir is None:
            log_dir = str(Path(__file__).parent.parent / "logs")

        # Guarantee operational workspace existence by spinning up directories on demand
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # Append isolated date keys onto logging files to establish historical file structures
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = Path(log_dir) / f"stock_predictor_{timestamp}.log"

        # Instantiate persistent disk file context handlers using robust encoding sets
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(FILE_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")

    return logger


# ==========================================
# SYSTEM SINGLETON REFERENCE EXPOSURES
# ==========================================
# Instantiate system baseline tracking scope allocation mapping
default_logger: logging.Logger = get_logger("stock_predictor")


def get_default_logger() -> logging.Logger:
    """
    Exposes a global, un-instantiated baseline channel for high-level package diagnostics.
    """
    return default_logger
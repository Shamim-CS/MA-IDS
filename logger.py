"""
Logging to console and to a timestamped log file.

Author: Md Shamimul Islam
Project: MA-IDS: Multi-Agent RAG Framework for IoT Network Intrusion Detection with an Experience Library
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


class IDSLogger:
    """Custom logger for IDS system with file and console output"""
    
    def __init__(self, log_dir: str = "logs", prefix: str = "ids_run"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / f"{prefix}_log_{self.timestamp}.txt"
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Configure logging with file and console handlers"""
        logger = logging.getLogger(f"IDS_{self.timestamp}")
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def separator(self, length: int = 60):
        """Log a separator line"""
        self.info("=" * length)
    
    def section_header(self, title: str, length: int = 60):
        """Log a section header"""
        self.separator(length)
        self.info(title)
        self.separator(length)
    
    def get_log_file(self) -> Path:
        """Return the path to the log file"""
        return self.log_file
    
    def get_timestamp(self) -> str:
        """Return the timestamp for this logger instance"""
        return self.timestamp

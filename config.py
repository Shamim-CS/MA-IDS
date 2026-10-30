"""
Configuration for paths, model settings and IDS parameters.

Author: Md Shamimul Islam
Project: MA-IDS: Multi-Agent RAG Framework for IoT Network Intrusion Detection with an Experience Library
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv()


@dataclass
class PathConfig:
    """File and directory path configurations"""
    # Set these in your .env file (see .env.example) or edit the defaults here
    brain_path: str = field(default_factory=lambda: os.getenv(
        "BRAIN_PATH", "brain/experience_library"))
    online_file: str = field(default_factory=lambda: os.getenv(
        "DATA_FILE", "data/train.csv"))
    log_dir: str = "logs"
    output_dir: str = "output"
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """LLM and embedding model configurations"""
    llm_model: str = "openai/gpt-4o"
    llm_temperature: float = 0.0
    llm_max_retries: int = 3
    base_url: str = "https://openrouter.ai/api/v1"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))


@dataclass
class IDSConfig:
    """IDS system configurations"""
    #valid_labels: List[str] = field(default_factory=lambda: [
    #    "Benign", "Theft", "DDoS", "DoS", "Reconnaissance"
    #])
    #valid_labels: List[str] = field(default_factory=lambda: [
    #    "Benign", "Fuzzers", "Analysis", "Backdoor", "DoS", "Exploits", "Generic", "Reconnaissance", "Shellcode", "Worms"
    #])
    valid_labels: List[str] = field(default_factory=lambda: [
        "Benign", "scanning", "ddos", "Backdoor", "dos", "injection", "password", "xss", "ransomware", "mitm"
    ])
    sample_size: int = 50000
    retrieval_top_k: int = 1
    protocol_map: dict = field(default_factory=lambda: {
        6: "TCP", 
        17: "UDP", 
        1: "ICMP"
    })




@dataclass
class Config:
    """Main configuration container"""
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ids: IDSConfig = field(default_factory=IDSConfig)
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.model.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        if not Path(self.paths.online_file).exists():
            raise FileNotFoundError(f"Data file not found: {self.paths.online_file}")
        
        return True


# Global config instance
config = Config()


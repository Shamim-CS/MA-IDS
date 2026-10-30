"""
Main MA-IDS pipeline: loads data, classifies flows, learns rules from
mistakes and evaluates the results.

Author: Md Shamimul Islam
Project: MA-IDS: Multi-Agent RAG Framework for IoT Network Intrusion Detection with an Experience Library
"""
import time
from typing import List, Tuple
from langchain_openai import ChatOpenAI

from config import Config
from logger import IDSLogger
from memory import MemoryManager
from classifier import FlowClassifier
from data_loader import DataLoader
from evaluator import Evaluator


class IDSSystem:
    """Main Intrusion Detection System coordinating all components"""
    
    def __init__(self, config: Config):
        """
        Initialize IDS System
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Initialize logger
        self.logger = IDSLogger(
            log_dir=config.paths.log_dir,
            prefix="ids_run"
        )
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize components
        self.memory = MemoryManager(
            brain_path=config.paths.brain_path,
            embedding_model=config.model.embedding_model,
            embedding_device=config.model.embedding_device,
            logger=self.logger
        )
        
        self.classifier = FlowClassifier(
            llm=self.llm,
            memory_manager=self.memory,
            valid_labels=config.ids.valid_labels,
            protocol_map=config.ids.protocol_map,
            logger=self.logger
        )
        
        self.data_loader = DataLoader(logger=self.logger)
        
        self.evaluator = Evaluator(
            valid_labels=config.ids.valid_labels,
            output_dir=config.paths.output_dir,
            logger=self.logger
        )
        
        # Track predictions over time for learning curves
        self.predictions_over_time = []
        self.original_df = None
    
    def _initialize_llm(self) -> ChatOpenAI:
        """Initialize the LLM"""
        self.logger.info("Initializing LLM...")
        return ChatOpenAI(
            model=self.config.model.llm_model,
            temperature=self.config.model.llm_temperature,
            max_retries=self.config.model.llm_max_retries,
            api_key=self.config.model.api_key,
            base_url=self.config.model.base_url
        )
    
    def run(self) -> dict:
        """
        Run the complete IDS pipeline
        
        Returns:
            Dictionary with results and metrics
        """
        self.logger.section_header("Starting IDS System")
        self.logger.info(f"Logs saved to: {self.logger.get_log_file()}")
        
        # Load data
        df, label_col = self.data_loader.load_data(
            file_path=self.config.paths.online_file,
            sample_size=self.config.ids.sample_size
        )
        
        # Store original dataframe for feature analysis
        self.original_df = df.copy()
        
        # Get data info
        data_info = self.data_loader.get_data_info(df, label_col)
        
        # Process flows
        y_true, y_pred = self._process_flows(df, label_col)
        
        # Generate evaluation metrics
        results = self._evaluate_results(y_true, y_pred, df)
        
        self.logger.info("\nExecution finished. Check the log file for detailed analysis.")
        
        return {
            "data_info": data_info,
            "results": results,
            "log_file": str(self.logger.get_log_file())
        }
    
    def _process_flows(self, df, label_col: str) -> Tuple[List[str], List[str]]:
        """
        Process all flows in the dataset
        
        Args:
            df: DataFrame with flow data
            label_col: Name of label column
            
        Returns:
            Tuple of (true_labels, predicted_labels)
        """
        y_true = []
        y_pred = []
        
        start_time = time.time()
        self.logger.info("\nProcessing Flows...")
        
        for index, row in df.iterrows():
            true_label = str(row[label_col]).strip()
            
            # Log separator
            self.logger.info(f"\n--- Packet {index} (True Label: {true_label}) ---")
            
            # Classify
            pred_label, features = self.classifier.classify(row)
            
            y_true.append(true_label)
            y_pred.append(pred_label)
            
            # Check correctness
            is_correct = (pred_label.lower() == true_label.lower())
            icon = "correct" if is_correct else "wrong"
            self.logger.info(f"   Pred: {pred_label} | True: {true_label} | {icon}")
            
            # Track for learning curves
            self.predictions_over_time.append((index, true_label, pred_label, is_correct))
            
            # Learn from mistakes
            if not is_correct:
                self._learn_from_mistake(features, true_label, pred_label)
        
        elapsed = time.time() - start_time
        self.logger.info(f"\n⏱️  Processing completed in {elapsed:.2f}s")
        
        return y_true, y_pred
    
    def _learn_from_mistake(self, features: dict, true_label: str, pred_label: str):
        """
        Learn from a classification mistake
        
        Args:
            features: Flow features
            true_label: Correct label
            pred_label: Incorrect prediction
        """
        self.logger.info("   Mistake detected! Distilling experience...")
        
        exp = self.classifier.distill_experience(features, true_label, pred_label)
        
        if exp and "strategy_signature" in exp:
            import json
            self.memory.add_rule(json.dumps(features), exp)
            self.memory.save()
    
    def _evaluate_results(self, y_true: List[str], y_pred: List[str], df) -> dict:
        """
        Evaluate and visualize results
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            df: Original dataframe
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Generate summary report
        summary = self.evaluator.generate_summary_report(
            y_true, 
            y_pred,
            elapsed_time=0  # Already logged
        )
        
        # Generate classification report
        self.evaluator.generate_classification_report(y_true, y_pred)
        
        # Generate basic confusion matrix
        cm_file = self.evaluator.generate_confusion_matrix(
            y_true, 
            y_pred,
            self.logger.get_timestamp()
        )
        
        # Generate all research paper figures
        self.logger.section_header("GENERATING RESEARCH FIGURES")
        
        # Identify feature columns for correlation analysis
        feature_columns = [
            'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS',
            'FLOW_DURATION_MILLISECONDS', 'SRC_TO_DST_IAT_AVG',
            'DST_TO_SRC_IAT_AVG', 'SRC_TO_DST_AVG_THROUGHPUT',
            'DST_TO_SRC_AVG_THROUGHPUT', 'TCP_FLAGS'
        ]
        
        # Filter to only existing columns
        available_features = [col for col in feature_columns if col in df.columns]
        
        research_figures = self.evaluator.generate_research_figures(
            y_true=y_true,
            y_pred=y_pred,
            timestamp=self.logger.get_timestamp(),
            predictions_over_time=self.predictions_over_time,
            df=df,
            feature_columns=available_features if available_features else None
        )
        
        self.logger.info(f"\nGenerated {len(research_figures)} research figures")
        
        # Save results for potential future figure regeneration
        self._save_results(y_true, y_pred)
        
        return {
            "summary": summary,
            "confusion_matrix_file": str(cm_file) if cm_file else None,
            "research_figures": {name: str(path) for name, path in research_figures.items()}
        }
    
    def _save_results(self, y_true: List[str], y_pred: List[str]):
        """Save prediction results for later analysis"""
        import pickle
        from pathlib import Path
        
        output_path = Path(self.config.paths.output_dir)
        results_file = output_path / f"results_{self.logger.get_timestamp()}.pkl"
        
        data = {
            'y_true': y_true,
            'y_pred': y_pred,
            'predictions_over_time': self.predictions_over_time,
            'valid_labels': self.config.ids.valid_labels,
            'timestamp': self.logger.get_timestamp()
        }
        
        try:
            with open(results_file, 'wb') as f:
                pickle.dump(data, f)
            self.logger.info(f"Results saved to: {results_file}")
        except Exception as e:
            self.logger.warning(f"Could not save results: {e}")

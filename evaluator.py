"""
Evaluation metrics, classification report and confusion matrix.

Author: Md Shamimul Islam
Project: MA-IDS: Multi-Agent RAG Framework for IoT Network Intrusion Detection with an Experience Library
"""
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from logger import IDSLogger


class Evaluator:
    """Handles evaluation and visualization of IDS performance"""
    
    def __init__(
        self, 
        valid_labels: List[str],
        output_dir: str,
        logger: IDSLogger
    ):
        self.valid_labels = valid_labels
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_classification_report(
        self, 
        y_true: List[str], 
        y_pred: List[str]
    ) -> str:
        """
        Generate sklearn classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report as string
        """
        try:
            report = classification_report(
                y_true, 
                y_pred, 
                labels=self.valid_labels,
                zero_division=0
            )
            self.logger.info("\n--- Classification Report ---")
            self.logger.info(report)
            return report
        
        except Exception as e:
            self.logger.error(f"Could not generate classification report: {e}")
            return ""
    
    def generate_confusion_matrix(
        self, 
        y_true: List[str], 
        y_pred: List[str],
        timestamp: str
    ) -> Path:
        """
        Generate and save confusion matrix heatmap
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            timestamp: Timestamp for filename
            
        Returns:
            Path to saved confusion matrix image
        """
        try:
            cm_file = self.output_dir / f"confusion_matrix_{timestamp}.png"
            self.logger.info(f"Generating Confusion Matrix Heatmap -> {cm_file}")
            
            cm = confusion_matrix(y_true, y_pred, labels=self.valid_labels)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=self.valid_labels,
                yticklabels=self.valid_labels
            )
            plt.title('IDS Confusion Matrix')
            plt.ylabel('Actual Label')
            plt.xlabel('Predicted Label')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(cm_file)
            plt.close()
            
            self.logger.info("Heatmap saved successfully.")
            return cm_file
        
        except Exception as e:
            self.logger.error(f"Error generating heatmap: {e}")
            return None
    
    def calculate_accuracy(self, y_true: List[str], y_pred: List[str]) -> float:
        """Calculate overall accuracy"""
        correct = sum(1 for t, p in zip(y_true, y_pred) if t.lower() == p.lower())
        total = len(y_true)
        accuracy = (correct / total) * 100 if total > 0 else 0
        return accuracy
    
    def generate_summary_report(
        self,
        y_true: List[str],
        y_pred: List[str],
        elapsed_time: float
    ) -> dict:
        """
        Generate a comprehensive summary report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            elapsed_time: Processing time in seconds
            
        Returns:
            Dictionary with summary statistics
        """
        accuracy = self.calculate_accuracy(y_true, y_pred)
        
        # Count correct and incorrect predictions
        correct = sum(1 for t, p in zip(y_true, y_pred) if t.lower() == p.lower())
        incorrect = len(y_true) - correct
        
        summary = {
            "total_samples": len(y_true),
            "correct_predictions": correct,
            "incorrect_predictions": incorrect,
            "accuracy": accuracy,
            "elapsed_time": elapsed_time,
            "avg_time_per_sample": elapsed_time / len(y_true) if len(y_true) > 0 else 0
        }
        
        self.logger.section_header("FINAL PERFORMANCE REPORT")
        self.logger.info(f"Total Samples: {summary['total_samples']}")
        self.logger.info(f"Correct Predictions: {summary['correct_predictions']}")
        self.logger.info(f"Incorrect Predictions: {summary['incorrect_predictions']}")
        self.logger.info(f"Accuracy: {summary['accuracy']:.2f}%")
        self.logger.info(f"Total Time: {summary['elapsed_time']:.2f}s")
        self.logger.info(f"Avg Time per Sample: {summary['avg_time_per_sample']:.3f}s")
        
        return summary
    
    def generate_research_figures(
        self,
        y_true: List[str],
        y_pred: List[str],
        timestamp: str,
        predictions_over_time: Optional[List[Tuple[int, str, str, bool]]] = None,
        df: Optional[pd.DataFrame] = None,
        feature_columns: Optional[List[str]] = None,
        model_comparison: Optional[Dict[str, float]] = None
    ) -> Dict[str, Path]:
        """
        Generate all research paper figures
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            timestamp: Timestamp for filenames
            predictions_over_time: Temporal prediction data
            df: Original dataframe for feature analysis
            feature_columns: List of feature column names
            model_comparison: Dictionary of model names to accuracies
            
        Returns:
            Dictionary mapping figure names to file paths
        """
        try:
            from visualizer import ResearchVisualizer
            
            visualizer = ResearchVisualizer(
                output_dir=str(self.output_dir),
                valid_labels=self.valid_labels,
                logger=self.logger,
                timestamp=timestamp,
                dpi=300
            )
            
            figures = visualizer.generate_all_figures(
                y_true=y_true,
                y_pred=y_pred,
                predictions_over_time=predictions_over_time,
                df=df,
                feature_columns=feature_columns
            )
            
            # Generate model comparison if data provided
            if model_comparison:
                figures['accuracy_comparison'] = visualizer.plot_accuracy_comparison(
                    model_comparison
                )
            
            return figures
        
        except ImportError as e:
            self.logger.error(f"Could not import visualizer: {e}")
            self.logger.info("Make sure visualizer.py is in the same directory")
            return {}
        
        except Exception as e:
            self.logger.error(f"Error generating research figures: {e}")
            import traceback
            traceback.print_exc()
            return {}

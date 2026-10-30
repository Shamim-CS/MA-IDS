"""
Combines the results of all batches and generates the final statistics and figures.
Run this after batch_processor.py has finished.

Author: Md Shamimul Islam
Project: MA-IDS: Multi-Agent RAG Framework for IoT Network Intrusion Detection with an Experience Library
"""
import pickle
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

from config import Config
from logger import IDSLogger
from visualizer import ResearchVisualizer
from evaluator import Evaluator


class ResultAnalyzer:
    """Analyze and visualize results from batch processing"""
    
    def __init__(self, batch_dir: str, config: Config = None):
        """
        Initialize result analyzer
        
        Args:
            batch_dir: Directory containing batch result files
            config: Optional config (will create default if not provided)
        """
        self.batch_dir = Path(batch_dir)
        self.config = config or Config()
        self.logger = IDSLogger(
            log_dir=self.config.paths.log_dir,
            prefix="analysis"
        )
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def load_all_batches(self) -> Dict:
        """
        Load and combine all batch results
        
        Returns:
            Dictionary with combined results
        """
        self.logger.section_header("LOADING BATCH RESULTS")
        
        # Find all batch files
        batch_files = sorted(self.batch_dir.glob("results_batch_*.pkl"))
        
        if not batch_files:
            raise FileNotFoundError(f"No batch files found in {self.batch_dir}")
        
        self.logger.info(f"Found {len(batch_files)} batch files")
        
        # Combine results
        all_y_true = []
        all_y_pred = []
        all_predictions_over_time = []
        batch_stats = []
        
        for batch_file in batch_files:
            self.logger.info(f"   Loading: {batch_file.name}")
            
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f)
            
            all_y_true.extend(batch_data['y_true'])
            all_y_pred.extend(batch_data['y_pred'])
            
            # Adjust indices for predictions_over_time
            if 'predictions_over_time' in batch_data:
                offset = batch_data['start_idx']
                adjusted_predictions = [
                    (idx + offset, true, pred, correct)
                    for idx, true, pred, correct in batch_data['predictions_over_time']
                ]
                all_predictions_over_time.extend(adjusted_predictions)
            
            # Calculate batch accuracy
            correct = sum(1 for t, p in zip(batch_data['y_true'], batch_data['y_pred']) 
                         if t.lower() == p.lower())
            accuracy = (correct / len(batch_data['y_true'])) * 100
            
            batch_stats.append({
                'batch_num': batch_data['batch_num'],
                'start_idx': batch_data['start_idx'],
                'end_idx': batch_data['end_idx'],
                'samples': len(batch_data['y_true']),
                'accuracy': accuracy,
                'correct': correct,
                'incorrect': len(batch_data['y_true']) - correct
            })
        
        self.logger.info(f"\nLoaded {len(all_y_true)} total predictions")
        
        return {
            'y_true': all_y_true,
            'y_pred': all_y_pred,
            'predictions_over_time': all_predictions_over_time,
            'batch_stats': batch_stats,
            'num_batches': len(batch_files),
            'timestamp': self.timestamp
        }
    
    def calculate_statistics(self, results: Dict) -> Dict:
        """
        Calculate comprehensive statistics
        
        Args:
            results: Combined results dictionary
            
        Returns:
            Dictionary with statistics
        """
        self.logger.section_header("CALCULATING STATISTICS")
        
        y_true = results['y_true']
        y_pred = results['y_pred']
        
        # Overall accuracy
        correct = sum(1 for t, p in zip(y_true, y_pred) if t.lower() == p.lower())
        total = len(y_true)
        overall_accuracy = (correct / total) * 100
        
        # Per-class statistics
        from collections import defaultdict, Counter
        
        true_counts = Counter(y_true)
        pred_counts = Counter(y_pred)
        
        per_class_correct = defaultdict(int)
        per_class_total = defaultdict(int)
        
        for t, p in zip(y_true, y_pred):
            per_class_total[t] += 1
            if t.lower() == p.lower():
                per_class_correct[t] += 1
        
        per_class_accuracy = {
            label: (per_class_correct[label] / per_class_total[label] * 100)
            for label in per_class_total
        }
        
        # Batch-level statistics
        batch_accuracies = [b['accuracy'] for b in results['batch_stats']]
        
        stats = {
            'total_samples': total,
            'correct_predictions': correct,
            'incorrect_predictions': total - correct,
            'overall_accuracy': overall_accuracy,
            'per_class_accuracy': per_class_accuracy,
            'true_distribution': dict(true_counts),
            'pred_distribution': dict(pred_counts),
            'batch_accuracies': batch_accuracies,
            'num_batches': results['num_batches'],
            'avg_batch_accuracy': sum(batch_accuracies) / len(batch_accuracies),
            'min_batch_accuracy': min(batch_accuracies),
            'max_batch_accuracy': max(batch_accuracies)
        }
        
        # Log statistics
        self.logger.info(f"Total Samples: {stats['total_samples']:,}")
        self.logger.info(f"Correct: {stats['correct_predictions']:,}")
        self.logger.info(f"Incorrect: {stats['incorrect_predictions']:,}")
        self.logger.info(f"Overall Accuracy: {stats['overall_accuracy']:.2f}%")
        
        self.logger.info("\nPer-Class Accuracy:")
        for label, acc in sorted(per_class_accuracy.items()):
            count = per_class_total[label]
            self.logger.info(f"   {label:20s}: {acc:6.2f}% ({count:,} samples)")
        
        self.logger.info(f"\nBatch Statistics:")
        self.logger.info(f"   Number of Batches: {stats['num_batches']}")
        self.logger.info(f"   Avg Batch Accuracy: {stats['avg_batch_accuracy']:.2f}%")
        self.logger.info(f"   Min Batch Accuracy: {stats['min_batch_accuracy']:.2f}%")
        self.logger.info(f"   Max Batch Accuracy: {stats['max_batch_accuracy']:.2f}%")
        
        return stats
    
    def generate_all_figures(self, results: Dict, stats: Dict) -> Dict[str, Path]:
        """
        Generate all research figures
        
        Args:
            results: Combined results
            stats: Statistics dictionary
            
        Returns:
            Dictionary mapping figure names to paths
        """
        self.logger.section_header("GENERATING RESEARCH FIGURES")
        
        # Initialize visualizer
        visualizer = ResearchVisualizer(
            output_dir=str(self.config.paths.output_dir),
            valid_labels=self.config.ids.valid_labels,
            logger=self.logger,
            timestamp=self.timestamp,
            dpi=300
        )
        
        # Generate all standard figures
        figures = visualizer.generate_all_figures(
            y_true=results['y_true'],
            y_pred=results['y_pred'],
            predictions_over_time=results['predictions_over_time']
        )
        
        # Generate batch-specific figures
        self.logger.info("Generating batch analysis figures...")
        figures['batch_accuracy'] = self._plot_batch_accuracy(
            results['batch_stats'], visualizer
        )
        figures['cumulative_accuracy'] = self._plot_cumulative_accuracy(
            results['predictions_over_time'], visualizer
        )
        
        self.logger.info(f"\nGenerated {len(figures)} figures")
        
        return figures
    
    def _plot_batch_accuracy(self, batch_stats: List[Dict], visualizer) -> Path:
        """Plot accuracy per batch"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        batch_nums = [b['batch_num'] + 1 for b in batch_stats]
        accuracies = [b['accuracy'] for b in batch_stats]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(batch_nums, accuracies, marker='o', linewidth=2, markersize=4)
        ax.axhline(y=np.mean(accuracies), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(accuracies):.2f}%')
        
        ax.set_xlabel('Batch Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy Across Batches', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        return visualizer.save_figure(fig, 'batch_accuracy_progression', 'learning')
    
    def _plot_cumulative_accuracy(self, predictions_over_time: List, visualizer) -> Path:
        """Plot cumulative accuracy over all samples"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        if not predictions_over_time:
            self.logger.warning("No temporal data for cumulative accuracy plot")
            return None
        
        indices = [p[0] for p in predictions_over_time]
        correct = [1 if p[3] else 0 for p in predictions_over_time]
        
        # Calculate cumulative accuracy
        cumulative_correct = np.cumsum(correct)
        cumulative_total = np.arange(1, len(correct) + 1)
        cumulative_accuracy = (cumulative_correct / cumulative_total) * 100
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(indices, cumulative_accuracy, linewidth=2)
        ax.axhline(y=cumulative_accuracy[-1], color='r', linestyle='--',
                   label=f'Final: {cumulative_accuracy[-1]:.2f}%')
        
        ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Cumulative Accuracy Over All Samples', 
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        return visualizer.save_figure(fig, 'cumulative_accuracy', 'learning')
    
    def save_final_report(self, results: Dict, stats: Dict, figures: Dict):
        """Save comprehensive final report"""
        report_file = Path(self.config.paths.output_dir) / f"final_report_{self.timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("IDS SYSTEM - FINAL ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Batch Directory: {self.batch_dir}\n")
            f.write(f"Number of Batches: {stats['num_batches']}\n\n")
            
            f.write("="*80 + "\n")
            f.write("OVERALL PERFORMANCE\n")
            f.write("="*80 + "\n")
            f.write(f"Total Samples Processed: {stats['total_samples']:,}\n")
            f.write(f"Correct Predictions: {stats['correct_predictions']:,}\n")
            f.write(f"Incorrect Predictions: {stats['incorrect_predictions']:,}\n")
            f.write(f"Overall Accuracy: {stats['overall_accuracy']:.2f}%\n\n")
            
            f.write("="*80 + "\n")
            f.write("PER-CLASS PERFORMANCE\n")
            f.write("="*80 + "\n")
            for label in sorted(stats['per_class_accuracy'].keys()):
                acc = stats['per_class_accuracy'][label]
                count = stats['true_distribution'][label]
                f.write(f"{label:20s}: {acc:6.2f}% ({count:,} samples)\n")
            f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("BATCH STATISTICS\n")
            f.write("="*80 + "\n")
            f.write(f"Number of Batches: {stats['num_batches']}\n")
            f.write(f"Average Batch Accuracy: {stats['avg_batch_accuracy']:.2f}%\n")
            f.write(f"Min Batch Accuracy: {stats['min_batch_accuracy']:.2f}%\n")
            f.write(f"Max Batch Accuracy: {stats['max_batch_accuracy']:.2f}%\n\n")
            
            f.write("="*80 + "\n")
            f.write("GENERATED FIGURES\n")
            f.write("="*80 + "\n")
            for name, path in sorted(figures.items()):
                if path:
                    f.write(f"{name:40s}: {path}\n")
            f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        self.logger.info(f"Final report saved to: {report_file}")
        return report_file
    
    def export_results_csv(self, results: Dict, stats: Dict):
        """Export results to CSV for further analysis"""
        output_dir = Path(self.config.paths.output_dir)
        
        # Export predictions
        predictions_file = output_dir / f"all_predictions_{self.timestamp}.csv"
        df_predictions = pd.DataFrame({
            'index': range(len(results['y_true'])),
            'true_label': results['y_true'],
            'predicted_label': results['y_pred'],
            'correct': [t.lower() == p.lower() for t, p in zip(results['y_true'], results['y_pred'])]
        })
        df_predictions.to_csv(predictions_file, index=False)
        self.logger.info(f"Predictions exported to: {predictions_file}")
        
        # Export batch statistics
        batch_stats_file = output_dir / f"batch_statistics_{self.timestamp}.csv"
        df_batch_stats = pd.DataFrame(results['batch_stats'])
        df_batch_stats.to_csv(batch_stats_file, index=False)
        self.logger.info(f"Batch stats exported to: {batch_stats_file}")
        
        # Export per-class statistics
        per_class_file = output_dir / f"per_class_statistics_{self.timestamp}.csv"
        df_per_class = pd.DataFrame([
            {
                'class': label,
                'accuracy': stats['per_class_accuracy'][label],
                'true_count': stats['true_distribution'][label],
                'pred_count': stats['pred_distribution'].get(label, 0)
            }
            for label in sorted(stats['per_class_accuracy'].keys())
        ])
        df_per_class.to_csv(per_class_file, index=False)
        self.logger.info(f"Per-class stats exported to: {per_class_file}")


def main():
    """Main analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze batch processing results and generate figures"
    )
    parser.add_argument(
        '--batch-dir',
        type=str,
        default='output/batches',
        help='Directory containing batch result files'
    )
    parser.add_argument(
        '--no-figures',
        action='store_true',
        help='Skip figure generation (statistics only)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = ResultAnalyzer(batch_dir=args.batch_dir)
        
        # Load all batch results
        results = analyzer.load_all_batches()
        
        # Calculate statistics
        stats = analyzer.calculate_statistics(results)
        
        # Generate figures (unless skipped)
        figures = {}
        if not args.no_figures:
            figures = analyzer.generate_all_figures(results, stats)
        
        # Save reports
        analyzer.save_final_report(results, stats, figures)
        analyzer.export_results_csv(results, stats)
        
        # Summary
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"Total Samples: {stats['total_samples']:,}")
        print(f"Overall Accuracy: {stats['overall_accuracy']:.2f}%")
        print(f"Figures Generated: {len(figures)}")
        print(f"Output Directory: {analyzer.config.paths.output_dir}")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

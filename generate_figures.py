"""
Regenerates figures from saved result files.

Author: Md Shamimul Islam
Project: MA-IDS: Multi-Agent RAG Framework for IoT Network Intrusion Detection with an Experience Library
"""
import argparse
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime

from visualizer import ResearchVisualizer
from logger import IDSLogger


def load_results(results_file: str):
    """Load saved prediction results"""
    with open(results_file, 'rb') as f:
        data = pickle.load(f)
    return data


def save_results(y_true, y_pred, predictions_over_time, output_file: str):
    """Save prediction results for later visualization"""
    data = {
        'y_true': y_true,
        'y_pred': y_pred,
        'predictions_over_time': predictions_over_time,
        'timestamp': datetime.now().isoformat()
    }
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"Results saved to: {output_file}")


def generate_figures_from_results(
    results_file: str,
    output_dir: str = "figures",
    valid_labels: list = None
):
    """Generate all figures from saved results"""
    
    if valid_labels is None:
        valid_labels = ["Benign", "Theft", "DDoS", "DoS", "Reconnaissance"]
    
    # Load results
    print(f"Loading results from: {results_file}")
    data = load_results(results_file)
    
    y_true = data['y_true']
    y_pred = data['y_pred']
    predictions_over_time = data.get('predictions_over_time', None)
    
    # Initialize logger and visualizer
    logger = IDSLogger(log_dir=output_dir, prefix="figure_generation")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    visualizer = ResearchVisualizer(
        output_dir=output_dir,
        valid_labels=valid_labels,
        logger=logger,
        timestamp=timestamp,
        dpi=300
    )
    
    # Generate all figures
    figures = visualizer.generate_all_figures(
        y_true=y_true,
        y_pred=y_pred,
        predictions_over_time=predictions_over_time
    )
    
    logger.section_header("FIGURE GENERATION COMPLETE")
    logger.info(f"Generated {len(figures)} figures in: {output_dir}")
    for name, path in figures.items():
        logger.info(f"  - {name}: {path}")
    
    return figures


def generate_comparison_figures(
    results_dict: dict,
    output_dir: str = "comparison_figures",
    valid_labels: list = None
):
    """
    Generate comparison figures from multiple model results
    
    Args:
        results_dict: Dictionary like {'Model1': 'results1.pkl', 'Model2': 'results2.pkl'}
    """
    if valid_labels is None:
        valid_labels = ["Benign", "Theft", "DDoS", "DoS", "Reconnaissance"]
    
    logger = IDSLogger(log_dir=output_dir, prefix="comparison")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    visualizer = ResearchVisualizer(
        output_dir=output_dir,
        valid_labels=valid_labels,
        logger=logger,
        timestamp=timestamp
    )
    
    # Load all results and calculate accuracies
    model_accuracies = {}
    for model_name, results_file in results_dict.items():
        data = load_results(results_file)
        y_true = data['y_true']
        y_pred = data['y_pred']
        
        correct = sum(1 for t, p in zip(y_true, y_pred) if t.lower() == p.lower())
        accuracy = (correct / len(y_true)) * 100
        model_accuracies[model_name] = accuracy
    
    # Generate comparison figure
    comparison_fig = visualizer.plot_accuracy_comparison(model_accuracies)
    logger.info(f"Generated comparison figure: {comparison_fig}")
    
    return comparison_fig


def main():
    parser = argparse.ArgumentParser(
        description="Generate research figures from IDS results"
    )
    parser.add_argument(
        '--results',
        type=str,
        help='Path to pickled results file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='research_figures',
        help='Output directory for figures'
    )
    parser.add_argument(
        '--compare',
        nargs='+',
        help='Compare multiple results: --compare Model1:file1.pkl Model2:file2.pkl'
    )
    
    args = parser.parse_args()
    
    if args.compare:
        # Parse comparison arguments
        results_dict = {}
        for item in args.compare:
            model_name, results_file = item.split(':')
            results_dict[model_name] = results_file
        
        print("Generating comparison figures...")
        generate_comparison_figures(results_dict, args.output)
    
    elif args.results:
        print("Generating figures from results...")
        generate_figures_from_results(args.results, args.output)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

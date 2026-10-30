"""
Examples of running MA-IDS and generating figures from code.

Author: Md Shamimul Islam
Project: MA-IDS: Multi-Agent RAG Framework for IoT Network Intrusion Detection with an Experience Library
"""
# EXAMPLE 1: Basic Usage - Generate All Figures

def example_basic_usage():
    """
    Most common use case: Run IDS and generate all figures
    """
    print("="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60)
    
    from config import Config
    from ids_system import IDSSystem
    
    # Initialize with default config
    config = Config()
    
    # Run system (generates all figures automatically)
    ids = IDSSystem(config)
    results = ids.run()
    
    print(f"\nComplete! Generated {len(results['results']['research_figures'])} figures")
    print(f"Check output directory: {config.paths.output_dir}")


# EXAMPLE 2: Custom Configuration

def example_custom_config():
    """
    Run with custom configuration settings
    """
    print("="*60)
    print("EXAMPLE 2: Custom Configuration")
    print("="*60)
    
    from config import Config, PathConfig, IDSConfig
    from ids_system import IDSSystem
    
    # Create custom configuration
    config = Config()
    
    # Customize paths
    config.paths.output_dir = "my_custom_output"
    config.paths.log_dir = "my_custom_logs"
    
    # Customize IDS settings
    config.ids.sample_size = 500  # Process only 500 samples
    config.ids.valid_labels = ["Benign", "Attack"]  # Binary classification
    
    # Run system
    ids = IDSSystem(config)
    results = ids.run()
    
    print(f"Results saved to: {config.paths.output_dir}")


# EXAMPLE 3: Generate Specific Figures Only

def example_specific_figures():
    """
    Generate only specific figures you need
    """
    print("="*60)
    print("EXAMPLE 3: Generate Specific Figures")
    print("="*60)
    
    from visualizer import ResearchVisualizer
    from logger import IDSLogger
    from datetime import datetime
    
    # Sample data (replace with your actual data)
    y_true = ["Benign", "DDoS", "DoS", "Benign", "Theft"] * 20
    y_pred = ["Benign", "DDoS", "Benign", "Benign", "Theft"] * 20
    
    # Setup
    logger = IDSLogger(log_dir="logs", prefix="custom")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    visualizer = ResearchVisualizer(
        output_dir="custom_figures",
        valid_labels=["Benign", "Theft", "DDoS", "DoS", "Reconnaissance"],
        logger=logger,
        timestamp=timestamp,
        dpi=300
    )
    
    # Generate specific figures
    print("\nGenerating confusion matrices...")
    visualizer.plot_confusion_matrix_standard(y_true, y_pred)
    visualizer.plot_confusion_matrix_normalized(y_true, y_pred)
    
    print("\nGenerating performance metrics...")
    visualizer.plot_per_class_metrics(y_true, y_pred)
    visualizer.plot_metrics_radar(y_true, y_pred)
    
    print("\nDone! Check custom_figures/ directory")


# EXAMPLE 4: Regenerate Figures with Different Styles

def example_regenerate_figures():
    """
    Regenerate figures from saved results with different styling
    """
    print("="*60)
    print("EXAMPLE 4: Regenerate with Custom Styling")
    print("="*60)
    
    import pickle
    from pathlib import Path
    from visualizer import ResearchVisualizer
    from logger import IDSLogger
    import matplotlib.pyplot as plt
    
    # Find the most recent results file
    results_dir = Path("output")
    result_files = list(results_dir.glob("results_*.pkl"))
    
    if not result_files:
        print("No results files found. Run main.py first!")
        return
    
    latest_results = max(result_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading results from: {latest_results}")
    
    # Load results
    with open(latest_results, 'rb') as f:
        data = pickle.load(f)
    
    # Setup with custom style
    logger = IDSLogger(log_dir="logs", prefix="regenerate")
    
    # Change matplotlib style
    plt.style.use('ggplot')  # Different style!
    
    visualizer = ResearchVisualizer(
        output_dir="regenerated_figures",
        valid_labels=data['valid_labels'],
        logger=logger,
        timestamp=data['timestamp'] + "_regen",
        dpi=600  # Higher resolution!
    )
    
    # Regenerate all figures
    figures = visualizer.generate_all_figures(
        y_true=data['y_true'],
        y_pred=data['y_pred'],
        predictions_over_time=data.get('predictions_over_time')
    )
    
    print(f"\nRegenerated {len(figures)} figures with new styling!")


# EXAMPLE 5: Compare Multiple Models

def example_compare_models():
    """
    Generate comparison figures for multiple models
    """
    print("="*60)
    print("EXAMPLE 5: Model Comparison")
    print("="*60)
    
    from visualizer import ResearchVisualizer
    from logger import IDSLogger
    from datetime import datetime
    
    # Simulated results for different models
    # In real usage, these would come from different runs
    model_results = {
        "Baseline SVM": 78.5,
        "Random Forest": 82.3,
        "LSTM": 85.7,
        "Our IDS (GPT-4)": 92.4,
        "Our IDS + Memory": 95.8
    }
    
    # Also simulate detailed metrics
    model_details = {
        "Baseline SVM": {"Accuracy": 78.5, "Precision": 0.76, "Recall": 0.74, "F1-Score": 0.75},
        "Random Forest": {"Accuracy": 82.3, "Precision": 0.81, "Recall": 0.79, "F1-Score": 0.80},
        "LSTM": {"Accuracy": 85.7, "Precision": 0.84, "Recall": 0.83, "F1-Score": 0.83},
        "Our IDS (GPT-4)": {"Accuracy": 92.4, "Precision": 0.91, "Recall": 0.90, "F1-Score": 0.91},
        "Our IDS + Memory": {"Accuracy": 95.8, "Precision": 0.95, "Recall": 0.94, "F1-Score": 0.95}
    }
    
    # Setup
    logger = IDSLogger(log_dir="logs", prefix="comparison")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    visualizer = ResearchVisualizer(
        output_dir="comparison_figures",
        valid_labels=["Benign", "Theft", "DDoS", "DoS", "Reconnaissance"],
        logger=logger,
        timestamp=timestamp
    )
    
    # Generate comparison figures
    print("\nGenerating accuracy comparison...")
    visualizer.plot_accuracy_comparison(model_results)
    
    print("\nGenerating detailed comparison table...")
    visualizer.plot_model_comparison_table(model_details)
    
    print("\nComparison figures generated!")


# EXAMPLE 6: High-Resolution Figures for Print

def example_high_res_print():
    """
    Generate ultra-high resolution figures for print journals
    """
    print("="*60)
    print("EXAMPLE 6: High-Resolution Print Figures")
    print("="*60)
    
    from config import Config
    from ids_system import IDSSystem
    from visualizer import ResearchVisualizer
    
    # Create config
    config = Config()
    config.paths.output_dir = "print_quality_figures"
    config.ids.sample_size = 200  # Smaller sample for quick demo
    
    # Run IDS
    ids = IDSSystem(config)
    results = ids.run()
    
    # Note: To make figures truly print-ready, edit visualizer.py:
    # Change: dpi=300 to dpi=600
    # Change font sizes to be larger
    
    print("\nTIP: For print journals:")
    print("   - Set dpi=600 in visualizer.py")
    print("   - Increase font sizes by 2-3 points")
    print("   - Save as PDF in addition to PNG")
    print("   - Test figures at actual print size")


# EXAMPLE 7: Batch Process Multiple Datasets

def example_batch_processing():
    """
    Process multiple datasets and generate figures for each
    """
    print("="*60)
    print("EXAMPLE 7: Batch Processing")
    print("="*60)
    
    from config import Config
    from ids_system import IDSSystem
    from pathlib import Path
    
    # List of datasets to process
    datasets = [
        "/path/to/dataset1.csv",
        "/path/to/dataset2.csv",
        "/path/to/dataset3.csv"
    ]
    
    results_summary = {}
    
    for i, dataset_path in enumerate(datasets, 1):
        if not Path(dataset_path).exists():
            print(f"Skipping {dataset_path} (file not found)")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing Dataset {i}/{len(datasets)}: {Path(dataset_path).name}")
        print(f"{'='*60}")
        
        # Create custom config for this dataset
        config = Config()
        config.paths.online_file = dataset_path
        config.paths.output_dir = f"output_dataset_{i}"
        
        # Run IDS
        ids = IDSSystem(config)
        results = ids.run()
        
        # Store summary
        dataset_name = Path(dataset_path).stem
        results_summary[dataset_name] = results['results']['summary']['accuracy']
    
    # Generate comparison of all datasets
    print("\n" + "="*60)
    print("SUMMARY OF ALL DATASETS")
    print("="*60)
    for name, accuracy in results_summary.items():
        print(f"{name:30s}: {accuracy:.2f}%")


# EXAMPLE 8: Custom Feature Analysis

def example_feature_analysis():
    """
    Generate feature-specific visualizations
    """
    print("="*60)
    print("EXAMPLE 8: Feature Analysis")
    print("="*60)
    
    import pandas as pd
    from visualizer import ResearchVisualizer
    from logger import IDSLogger
    from datetime import datetime
    
    # Load your dataset
    # df = pd.read_csv("your_dataset.csv")
    # For demo, create sample data
    df = pd.DataFrame({
        'IN_BYTES': [100, 200, 150, 300] * 25,
        'OUT_BYTES': [50, 100, 75, 150] * 25,
        'IN_PKTS': [10, 20, 15, 30] * 25,
        'OUT_PKTS': [5, 10, 8, 15] * 25,
        'FLOW_DURATION_MILLISECONDS': [1000, 2000, 1500, 3000] * 25
    })
    
    # Setup
    logger = IDSLogger(log_dir="logs", prefix="features")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    visualizer = ResearchVisualizer(
        output_dir="feature_analysis",
        valid_labels=["Benign", "Theft", "DDoS", "DoS", "Reconnaissance"],
        logger=logger,
        timestamp=timestamp
    )
    
    # Generate feature correlation heatmap
    print("\nGenerating feature correlation analysis...")
    feature_columns = ['IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS']
    visualizer.plot_feature_correlation_heatmap(df, feature_columns)
    
    print("\nFeature analysis complete!")


# MAIN MENU

def main():
    """Run example demonstrations"""
    print("\n" + "="*60)
    print("IDS VISUALIZATION - EXAMPLE USAGE")
    print("="*60)
    
    examples = {
        "1": ("Basic Usage (Run IDS + Generate All Figures)", example_basic_usage),
        "2": ("Custom Configuration", example_custom_config),
        "3": ("Generate Specific Figures Only", example_specific_figures),
        "4": ("Regenerate with Different Styling", example_regenerate_figures),
        "5": ("Compare Multiple Models", example_compare_models),
        "6": ("High-Resolution for Print", example_high_res_print),
        "7": ("Batch Process Multiple Datasets", example_batch_processing),
        "8": ("Custom Feature Analysis", example_feature_analysis),
    }
    
    print("\nAvailable Examples:")
    for key, (description, _) in examples.items():
        print(f"  {key}. {description}")
    
    print("\n" + "="*60)
    choice = input("Select example to run (1-8, or 'all' for all, 'q' to quit): ").strip().lower()
    
    if choice == 'q':
        print("Goodbye!")
        return
    
    if choice == 'all':
        for key, (description, func) in examples.items():
            print(f"\n\n{'='*60}")
            print(f"Running Example {key}: {description}")
            print(f"{'='*60}")
            try:
                func()
            except Exception as e:
                print(f"Error in example {key}: {e}")
    elif choice in examples:
        _, func = examples[choice]
        try:
            func()
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()

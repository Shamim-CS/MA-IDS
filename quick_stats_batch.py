#!/usr/bin/env python3
"""
Prints the progress and accuracy of a running batch job.

Author: Md Shamimul Islam
Project: MA-IDS: Multi-Agent RAG Framework for IoT Network Intrusion Detection with an Experience Library
"""
import pickle
from pathlib import Path
from collections import Counter
import argparse


def check_batch_progress(batch_dir: str = "output/batches"):
    """Check progress of batch processing"""
    batch_path = Path(batch_dir)
    
    if not batch_path.exists():
        print(f"Batch directory not found: {batch_dir}")
        return
    
    # Find all batch files
    batch_files = sorted(batch_path.glob("results_batch_*.pkl"))
    
    if not batch_files:
        print(f"No batch files found in {batch_dir}")
        return
    
    print("="*80)
    print("BATCH PROCESSING PROGRESS")
    print("="*80)
    
    total_samples = 0
    total_correct = 0
    all_true = []
    all_pred = []
    batch_accuracies = []
    
    print(f"\nBatch | Samples | Correct | Accuracy")
    print("-" * 50)
    
    for batch_file in batch_files:
        try:
            with open(batch_file, 'rb') as f:
                data = pickle.load(f)
            
            samples = len(data['y_true'])
            correct = sum(1 for t, p in zip(data['y_true'], data['y_pred']) 
                         if t.lower() == p.lower())
            accuracy = (correct / samples * 100) if samples > 0 else 0
            
            total_samples += samples
            total_correct += correct
            all_true.extend(data['y_true'])
            all_pred.extend(data['y_pred'])
            batch_accuracies.append(accuracy)
            
            print(f"{data['batch_num']:5d} | {samples:7d} | {correct:7d} | {accuracy:6.2f}%")
        
        except Exception as e:
            print(f"Error reading {batch_file.name}: {e}")
    
    # Overall statistics
    print("-" * 50)
    overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
    print(f"TOTAL | {total_samples:7d} | {total_correct:7d} | {overall_accuracy:6.2f}%")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Batches Completed: {len(batch_files)}")
    print(f"Total Samples: {total_samples:,}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    
    if batch_accuracies:
        print(f"Average Batch Accuracy: {sum(batch_accuracies)/len(batch_accuracies):.2f}%")
        print(f"Min Batch Accuracy: {min(batch_accuracies):.2f}%")
        print(f"Max Batch Accuracy: {max(batch_accuracies):.2f}%")
    
    # Class distribution
    print("\n" + "="*80)
    print("CLASS DISTRIBUTION")
    print("="*80)
    
    true_counts = Counter(all_true)
    pred_counts = Counter(all_pred)
    
    all_labels = sorted(set(all_true) | set(all_pred))
    
    print(f"\n{'Class':20s} | {'True':>8s} | {'Predicted':>10s} | {'Accuracy':>10s}")
    print("-" * 60)
    
    for label in all_labels:
        true_count = true_counts.get(label, 0)
        pred_count = pred_counts.get(label, 0)
        
        # Calculate per-class accuracy
        class_correct = sum(1 for t, p in zip(all_true, all_pred) 
                           if t == label and t.lower() == p.lower())
        class_accuracy = (class_correct / true_count * 100) if true_count > 0 else 0
        
        print(f"{label:20s} | {true_count:8,d} | {pred_count:10,d} | {class_accuracy:9.2f}%")
    
    print("\n" + "="*80)
    
    # Estimate completion
    if len(batch_files) > 0:
        first_batch = batch_files[0]
        last_batch = batch_files[-1]
        
        with open(first_batch, 'rb') as f:
            first_data = pickle.load(f)
        with open(last_batch, 'rb') as f:
            last_data = pickle.load(f)
        
        samples_per_batch = samples  # From last batch
        completed_batches = len(batch_files)
        
        print(f"Estimated samples per batch: ~{samples_per_batch}")
        print(f"Completed: {completed_batches} batches")
        
        # If we can estimate total
        print(f"\nTo generate figures, run:")
        print(f"   python analyze_results.py --batch-dir {batch_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Check batch processing progress"
    )
    parser.add_argument(
        '--batch-dir',
        type=str,
        default='output/batches',
        help='Directory containing batch files'
    )
    
    args = parser.parse_args()
    check_batch_progress(args.batch_dir)


if __name__ == "__main__":
    main()

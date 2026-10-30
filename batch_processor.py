"""
Runs MA-IDS over a large dataset in batches (to stay within API rate limits)
and saves the results of each batch so a run can be resumed.

Author: Md Shamimul Islam
Project: MA-IDS: Multi-Agent RAG Framework for IoT Network Intrusion Detection with an Experience Library
"""
import pickle
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

from config import Config
from ids_system import IDSSystem
from logger import IDSLogger


class BatchProcessor:
    """Process large datasets in batches to handle API rate limits"""
    
    def __init__(self, config: Config, batch_size: int = 1000):
        """
        Initialize batch processor
        
        Args:
            config: System configuration
            batch_size: Number of samples per batch (default 1000)
        """
        self.config = config
        self.batch_size = batch_size
        self.logger = IDSLogger(log_dir=config.paths.log_dir, prefix="batch")
        self.batch_dir = Path(config.paths.output_dir) / "batches"
        self.batch_dir.mkdir(parents=True, exist_ok=True)
    
    def process_in_batches(
        self, 
        total_samples: int,
        start_batch: int = 0,
        delay_between_batches: int = 0
    ) -> List[Path]:
        """
        Process dataset in batches
        
        Args:
            total_samples: Total number of samples to process
            start_batch: Batch number to start from (for resuming)
            delay_between_batches: Seconds to wait between batches
            
        Returns:
            List of batch result file paths
        """
        import pandas as pd
        
        self.logger.section_header("STARTING BATCH PROCESSING")
        
        # Load full dataset once
        self.logger.info(f"Loading full dataset: {self.config.paths.online_file}")
        df_full = pd.read_csv(self.config.paths.online_file)
        
        # Determine label column
        label_col = 'Attack' if 'Attack' in df_full.columns else df_full.columns[-1]
        
        # Calculate number of batches
        num_batches = (total_samples + self.batch_size - 1) // self.batch_size
        
        self.logger.info(f"Total samples: {total_samples}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Number of batches: {num_batches}")
        
        batch_files = []
        
        for batch_num in range(start_batch, num_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_samples)
            
            self.logger.separator()
            self.logger.info(f"BATCH {batch_num + 1}/{num_batches}")
            self.logger.info(f"   Samples: {start_idx} to {end_idx}")
            
            # Extract batch
            df_batch = df_full.iloc[start_idx:end_idx].copy()
            
            # Create temporary config for this batch
            batch_config = Config()
            batch_config.paths = self.config.paths
            batch_config.model = self.config.model
            batch_config.ids = self.config.ids
            batch_config.ids.sample_size = None  # Process all in batch
            
            # Save batch to temporary file
            batch_csv = self.batch_dir / f"batch_{batch_num:04d}.csv"
            df_batch.to_csv(batch_csv, index=False)
            batch_config.paths.online_file = str(batch_csv)
            
            try:
                # Process batch
                ids = IDSSystem(batch_config)
                
                # Disable figure generation for intermediate batches
                ids.generate_figures = False
                
                # Process (this will create y_true, y_pred)
                from data_loader import DataLoader
                data_loader = DataLoader(ids.logger)
                df, label_col_batch = data_loader.load_data(str(batch_csv), None)
                
                y_true, y_pred = ids._process_flows(df, label_col_batch)
                
                # Save batch results
                batch_result = {
                    'batch_num': batch_num,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'predictions_over_time': ids.predictions_over_time,
                    'timestamp': datetime.now().isoformat()
                }
                
                batch_file = self.batch_dir / f"results_batch_{batch_num:04d}.pkl"
                with open(batch_file, 'wb') as f:
                    pickle.dump(batch_result, f)
                
                batch_files.append(batch_file)
                
                self.logger.info(f"Batch {batch_num + 1} completed")
                self.logger.info(f"Saved to: {batch_file}")
                
                # Cleanup temporary CSV
                batch_csv.unlink()
                
            except Exception as e:
                self.logger.error(f"Error processing batch {batch_num + 1}: {e}")
                import traceback
                traceback.print_exc()
                
                # Save error info
                error_file = self.batch_dir / f"error_batch_{batch_num:04d}.txt"
                with open(error_file, 'w') as f:
                    f.write(f"Error in batch {batch_num}\n")
                    f.write(traceback.format_exc())
                
                continue
            
            # Delay between batches to avoid rate limits
            if delay_between_batches > 0 and batch_num < num_batches - 1:
                self.logger.info(f"⏳ Waiting {delay_between_batches}s before next batch...")
                time.sleep(delay_between_batches)
        
        self.logger.section_header("BATCH PROCESSING COMPLETE")
        self.logger.info(f"Processed {len(batch_files)}/{num_batches} batches successfully")
        
        return batch_files
    
    def resume_processing(self) -> List[Path]:
        """
        Resume processing from last completed batch
        
        Returns:
            List of all batch result files
        """
        # Find existing batch files
        existing_batches = sorted(self.batch_dir.glob("results_batch_*.pkl"))
        
        if not existing_batches:
            self.logger.info("No existing batches found. Starting from beginning.")
            return []
        
        # Get last batch number
        last_batch_file = existing_batches[-1]
        last_batch_num = int(last_batch_file.stem.split('_')[-1])
        
        self.logger.info(f"Found {len(existing_batches)} existing batches")
        self.logger.info(f"Resuming from batch {last_batch_num + 2}")
        
        return existing_batches


def main():
    """Main function for batch processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process large datasets in batches")
    parser.add_argument('--total', type=int, default=100000, help='Total samples to process')
    parser.add_argument('--batch-size', type=int, default=1000, help='Samples per batch')
    parser.add_argument('--delay', type=int, default=60, help='Seconds between batches')
    parser.add_argument('--resume', action='store_true', help='Resume from last batch')
    parser.add_argument('--start-batch', type=int, default=0, help='Batch to start from')
    
    args = parser.parse_args()
    
    # Load config
    config = Config()
    
    # Create processor
    processor = BatchProcessor(config, batch_size=args.batch_size)
    
    # Resume or start fresh
    if args.resume:
        existing = processor.resume_processing()
        if existing:
            last_num = int(existing[-1].stem.split('_')[-1])
            args.start_batch = last_num + 1
    
    # Process batches
    batch_files = processor.process_in_batches(
        total_samples=args.total,
        start_batch=args.start_batch,
        delay_between_batches=args.delay
    )
    
    print(f"\nProcessing complete!")
    print(f"Processed batches: {len(batch_files)}")
    print(f"Results saved in: {processor.batch_dir}")
    print(f"\nNext step: Run analysis to combine results and generate figures")
    print(f"   python analyze_results.py --batch-dir {processor.batch_dir}")


if __name__ == "__main__":
    main()

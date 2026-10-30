"""
Entry point for running MA-IDS.

Author: Md Shamimul Islam
Project: MA-IDS: Multi-Agent RAG Framework for IoT Network Intrusion Detection with an Experience Library
"""
import sys
from config import Config
from ids_system import IDSSystem


def main():
    """Main execution function"""
    try:
        # Load and validate configuration
        config = Config()
        config.validate()
        
        # Initialize and run IDS system
        ids = IDSSystem(config)
        results = ids.run()
        
        # Print summary
        print("\n" + "="*60)
        print("IDS System completed successfully!")
        print(f"Accuracy: {results['results']['summary']['accuracy']:.2f}%")
        print(f"Log file: {results['log_file']}")
        if results['results']['confusion_matrix_file']:
            print(f"Confusion Matrix: {results['results']['confusion_matrix_file']}")
        print("="*60)
        
        return 0
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    except ValueError as e:
        print(f"Configuration Error: {e}")
        return 1
    
    except Exception as e:
        print(f"Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

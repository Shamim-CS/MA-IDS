# MA-IDS: Multi-Agent RAG Framework for IoT Network Intrusion Detection with an Experience Library

**Author:** Md Shamimul Islam, Florida Polytechnic University

Official code for the paper **"MA-IDS: Multi-Agent RAG Framework for IoT Network Intrusion Detection with an Experience Library"** (IEEE SMC 2026). [arXiv:2604.05458](https://arxiv.org/abs/2604.05458)

MA-IDS combines Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) for reasoning-driven network intrusion detection. When the classifier makes a mistake, an error analysis step turns it into a human-readable rule and stores it in a persistent **Experience Library** (a FAISS vector store). Later flows retrieve the most relevant rules, so the system keeps improving without retraining. Evaluated on NF-BoT-IoT and NF-ToN-IoT.

## Features

- **Experience Library**: FAISS vector store of learned rules, retrieved for each new flow
- **LLM-Powered Classification**: GPT-4o (through OpenRouter by default) reasons over flow features plus retrieved rules
- **Continual Learning**: Misclassifications are distilled into new interpretable rules
- **Evaluation**: Classification reports, confusion matrices and research figures
- **Detailed Logging**: Full record of every decision and learning event

## Repository Structure

```
├── main.py                   # Entry point
├── config.py                 # Configuration (paths, model, labels)
├── ids_system.py             # Pipeline orchestrator
├── classifier.py             # Feature extraction, prompting, rule distillation
├── memory.py                 # Experience Library (FAISS) management
├── data_loader.py            # CSV loading and cleaning
├── evaluator.py              # Metrics and confusion matrix
├── logger.py                 # Logging
├── visualizer.py             # Research figures
├── generate_figures.py       # Regenerate figures from saved results
├── batch_processor.py        # Large runs in batches (API rate limits)
├── analyze_result_batch.py   # Combine and analyze batch results
├── quick_stats_batch.py      # Check progress of a batch run
├── standalone_pipeline.py    # Single-file version of the pipeline
├── example_usage.py          # Programmatic usage examples
├── requirements.txt
└── .env.example              # Template for your .env file
```

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/shamim-38/MA-IDS.git
cd MA-IDS
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env: add your API key and set DATA_FILE / BRAIN_PATH
```

5. **Get the data**

Datasets are not included in this repository. Download the NetFlow versions of the datasets (NF-ToN-IoT, NF-BoT-IoT) from the [University of Queensland NIDS datasets page](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) and point `DATA_FILE` in `.env` to the CSV.

## Configuration

Paths and the API key come from `.env`. Everything else is in `config.py`.

### Paths (`.env`)
- `BRAIN_PATH`: Location of the Experience Library (FAISS index). Created on first run if missing
- `DATA_FILE`: Path to your network flow CSV
- Logs go to `logs/` and figures to `output/`

### Model Configuration (`config.py`)
- `llm_model`: LLM model to use (default: openai/gpt-4o)
- `base_url`: API endpoint (default: OpenRouter)
- `llm_temperature`: Temperature for LLM (default: 0.0)
- `embedding_model`: Embedding model for the vector store (default: all-MiniLM-L6-v2)

### IDS Configuration (`config.py`)
- `valid_labels`: Attack classes. The default list is for NF-ToN-IoT; a commented list for NF-BoT-IoT is included
- `sample_size`: Number of flows to process
- `retrieval_top_k`: Number of rules to retrieve per flow

## Usage

### Basic Usage

```bash
python main.py
```

### Large Runs in Batches

```bash
python batch_processor.py --total 50000 --batch-size 1000 --delay 60
python quick_stats_batch.py          # check progress at any time
python analyze_result_batch.py       # combine results and make figures
```

### Programmatic Usage

```python
from config import Config
from ids_system import IDSSystem

config = Config()
config.paths.online_file = "data/train.csv"
config.ids.sample_size = 500

ids = IDSSystem(config)
results = ids.run()

print(f"Accuracy: {results['results']['summary']['accuracy']:.2f}%")
```

## Data Format

Your CSV file should contain the following columns:

### Required Columns
- `IPV4_SRC_ADDR`: Source IP address
- `IPV4_DST_ADDR`: Destination IP address
- `L4_DST_PORT`: Destination port
- `PROTOCOL`: Protocol number (6=TCP, 17=UDP, 1=ICMP)
- `IN_BYTES`: Incoming bytes
- `OUT_BYTES`: Outgoing bytes
- `TCP_FLAGS`: TCP flags
- `FLOW_DURATION_MILLISECONDS`: Flow duration
- `IN_PKTS`: Incoming packets
- `OUT_PKTS`: Outgoing packets
- `SRC_TO_DST_IAT_AVG`: Average inter-arrival time (source to dest)
- `DST_TO_SRC_IAT_AVG`: Average inter-arrival time (dest to source)
- `SRC_TO_DST_AVG_THROUGHPUT`: Average throughput (source to dest)
- `DST_TO_SRC_AVG_THROUGHPUT`: Average throughput (dest to source)
- `Attack`: Label column (or specify in config)

## Output

The system generates:

1. **Log File**: `logs/ids_run_log_<timestamp>.txt`
   - Detailed flow-by-flow analysis
   - Retrieved memory rules
   - Learning events
   - Performance metrics

2. **Confusion Matrix**: `output/confusion_matrix_<timestamp>.png`
   - Visual representation of classification performance

3. **Classification Report**: Console and log output
   - Precision, recall, F1-score per class
   - Overall accuracy

## Attack Categories

**NF-ToN-IoT** (default in `config.py`): Benign, scanning, ddos, Backdoor, dos, injection, password, xss, ransomware, mitm

**NF-BoT-IoT**: Benign, Theft, DDoS, DoS, Reconnaissance

## How It Works

1. **Feature Extraction**: Extracts relevant features from network flows
2. **Memory Retrieval**: Queries FAISS for similar past experiences
3. **LLM Classification**: The LLM analyzes features with memory context
4. **Feedback Loop**: On mistakes, creates a rule and stores in memory
5. **Continuous Improvement**: Future similar flows benefit from learned rules

## Troubleshooting

**`Configuration Error: OPENAI_API_KEY not found in environment`**
Make sure `.env` exists (copy `.env.example`) and has a valid key.

**`Error: Data file not found`**
Set `DATA_FILE` in `.env` to the correct CSV path.

**Long runs stop because of API rate limits**
Use `batch_processor.py` with a delay between batches, and `--resume` to continue.

## Contributing

Issues and pull requests are welcome.

## License

MIT License, see [LICENSE](LICENSE).

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{islam2026maids,
  title={MA-IDS: Multi-Agent RAG Framework for IoT Network Intrusion Detection with an Experience Library},
  author={Islam, Md Shamimul and Jaimes, Luis G. and Dina, Ayesha S.},
  booktitle={IEEE International Conference on Systems, Man, and Cybernetics (SMC)},
  year={2026},
  note={arXiv:2604.05458}
}
```

## Contact

Md Shamimul Islam, Florida Polytechnic University
- Email: shamimul435@gmail.com, mislam3051@floridapoly.edu
- Website: [shamim-cs.github.io](https://shamim-cs.github.io)
- Issues: [GitHub Issues](https://github.com/shamim-cs/MA-IDS/issues)

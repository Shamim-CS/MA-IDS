"""
Single-file version of the MA-IDS pipeline (inference with a pre-built
Experience Library).

Author: Md Shamimul Islam
Project: MA-IDS: Multi-Agent RAG Framework for IoT Network Intrusion Detection with an Experience Library
"""
import os
import json
import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dotenv import load_dotenv
from sklearn.metrics import classification_report, confusion_matrix

# LangChain imports
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. SETUP & CONFIGURATION

load_dotenv()

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"ids_test_log_{TIMESTAMP}.txt"
CM_FILE = f"confusion_matrix_{TIMESTAMP}.png"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def log(msg):
    logging.info(msg)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    log("WARNING: OPENAI_API_KEY not found.")

BRAIN_PATH = os.getenv("BRAIN_PATH", "brain/experience_library")
ONLINE_FILE = os.getenv("DATA_FILE", "data/train.csv")

VALID_LABELS = ["Benign", "Theft", "DDoS", "DoS", "Reconnaissance"]

# 2. MODEL & EMBEDDINGS

llm = ChatOpenAI(
    model="openai/gpt-4o",
    temperature=0.0,
    max_retries=3,
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

log("Loading HuggingFace Embeddings (CPU)...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# 3. HELPER FUNCTIONS

def get_protocol_name(proto_num):
    proto_map = {6: "TCP", 17: "UDP", 1: "ICMP"}
    return proto_map.get(proto_num, str(proto_num))


def normalize_label(prediction):
    clean_pred = prediction.strip().replace(".", "").replace('"', '')
    for label in VALID_LABELS:
        if clean_pred.lower() == label.lower():
            return label
    return "Benign"


def load_brain():
    if os.path.exists(BRAIN_PATH):
        log(f"Loading Experience Library from: {BRAIN_PATH}")
        return FAISS.load_local(
            BRAIN_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        log("Brain path not found.")
        exit()


def retrieve_similar_rules(vectorstore, flow_str):
    try:
        docs = vectorstore.similarity_search(flow_str, k=3)
        rules = [
            d.metadata["strategy_signature"]
            for d in docs
            if "strategy_signature" in d.metadata
        ]

        if rules:
            log(f"   Retrieved Experience: {rules}")
            return "\n".join(rules)
        else:
            log("   No relevant experience found.")
            return "No relevant experience found."
    except Exception:
        return "No relevant experience found."


# 4. CLASSIFICATION LOGIC (RAG ONLY)

def classify_flow(row, vectorstore):

    features = {
        "Source IP": row.get('IPV4_SRC_ADDR', 'Unknown'),
        "Dest IP": row.get('IPV4_DST_ADDR', 'Unknown'),
        "Dest Port": row.get('L4_DST_PORT', 0),
        "Protocol": get_protocol_name(row.get('PROTOCOL', 0)),
        "Bytes In": row.get('IN_BYTES', 0),
        "Bytes Out": row.get('OUT_BYTES', 0),
        "TCP Flags": row.get('TCP_FLAGS', 0),
        "Duration (ms)": row.get('FLOW_DURATION_MILLISECONDS', 0),
        "Packets In": row.get('IN_PKTS', 0),
        "Packets Out": row.get('OUT_PKTS', 0),
        "IAT Avg (Src->Dst)": row.get('SRC_TO_DST_IAT_AVG', 0),
        "IAT Avg (Dst->Src)": row.get('DST_TO_SRC_IAT_AVG', 0),
        "Throughput (Src->Dst)": row.get('SRC_TO_DST_AVG_THROUGHPUT', 0),
        "Throughput (Dst->Src)": row.get('DST_TO_SRC_AVG_THROUGHPUT', 0)
    }

    flow_str = json.dumps(features)

    memory_context = retrieve_similar_rules(vectorstore, flow_str)

    template = f"""
    You are a Network Intrusion Detection Expert.

    CURRENT FLOW DATA:
    {{flow_data}}

    RETRIEVED EXPERIENCE LIBRARY:
    {{memory}}

    INSTRUCTIONS:
    1. Compare the flow with retrieved signatures.
    2. If a signature matches, apply it.
    3. Otherwise, classify based on flow statistics.
    4. If traffic appears normal, classify as "Benign".

    VALID LABELS (Choose EXACTLY one):
    {', '.join(VALID_LABELS)}

    OUTPUT:
    Return ONLY the label name.
    """

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    try:
        raw_pred = chain.invoke({
            "flow_data": flow_str,
            "memory": memory_context
        })
        return normalize_label(raw_pred)
    except Exception:
        return "Benign"


# 5. MAIN EXECUTION

if __name__ == "__main__":

    log("Starting RAG-Based IDS (Inference Mode Only)...")
    log(f"Log File: {LOG_FILE}")

    try:
        df = pd.read_csv(ONLINE_FILE)
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        label_col = 'Attack' if 'Attack' in df.columns else df.columns[-1]

        if len(df) > 1000:
            log("   -> Sampling 1000 rows for evaluation...")
            df = df.sample(n=1000, random_state=42)

    except FileNotFoundError:
        log("Dataset not found.")
        exit()

    vectorstore = load_brain()

    y_true = []
    y_pred = []

    start_time = time.time()

    log("\nProcessing flows...\n")

    for index, row in df.iterrows():

        true_label = str(row[label_col]).strip()
        pred_label = classify_flow(row, vectorstore)

        y_true.append(true_label)
        y_pred.append(pred_label)

        is_correct = (pred_label.lower() == true_label.lower())
        icon = "correct" if is_correct else "wrong"

        log(f"Packet {index} | Pred: {pred_label} | True: {true_label} | {icon}")

    end_time = time.time()
    elapsed = end_time - start_time

    log("\n" + "="*60)
    log("FINAL PERFORMANCE REPORT")
    log("="*60)
    log(f"Time: {elapsed:.2f}s")

    # Classification Report
    report = classification_report(
        y_true,
        y_pred,
        labels=VALID_LABELS,
        zero_division=0
    )
    log("\n--- Classification Report ---")
    log(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=VALID_LABELS)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=VALID_LABELS,
        yticklabels=VALID_LABELS
    )
    plt.title("IDS Confusion Matrix (RAG Only)")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(CM_FILE)

    log(f"Confusion Matrix saved to: {CM_FILE}")
    log("\nExecution finished.")

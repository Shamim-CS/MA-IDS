"""
Flow classifier: feature extraction, LLM classification with retrieved rules,
and rule distillation from misclassified flows.

Author: Md Shamimul Islam
Project: MA-IDS: Multi-Agent RAG Framework for IoT Network Intrusion Detection with an Experience Library
"""
import json
from typing import Dict, Tuple, List
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from logger import IDSLogger
from memory import MemoryManager


class FlowClassifier:
    """Classifies network flows using LLM with memory-augmented reasoning"""
    
    def __init__(
        self,
        llm: ChatOpenAI,
        memory_manager: MemoryManager,
        valid_labels: List[str],
        protocol_map: Dict[int, str],
        logger: IDSLogger
    ):
        self.llm = llm
        self.memory = memory_manager
        self.valid_labels = valid_labels
        self.protocol_map = protocol_map
        self.logger = logger
    
    def get_protocol_name(self, proto_num: int) -> str:
        """Map protocol number to name"""
        return self.protocol_map.get(proto_num, str(proto_num))
    
    def normalize_label(self, prediction: str) -> str:
        """
        Normalize LLM output to match valid labels
        
        Args:
            prediction: Raw prediction from LLM
            
        Returns:
            Normalized label from valid_labels
        """
        clean_pred = prediction.strip().replace(".", "").replace('"', '')
        
        # Check for direct or case-insensitive match
        for label in self.valid_labels:
            if clean_pred.lower() == label.lower():
                return label
        
        # Handle common hallucinations/misclassifications
        if "unauthorized" in clean_pred.lower() or "access" in clean_pred.lower():
            return "Theft"
        
        # Default fallback
        return "Benign"
    
    def extract_features(self, row: Dict) -> Dict:
        """
        Extract relevant features from network flow data
        
        Args:
            row: Dictionary containing flow data
            
        Returns:
            Dictionary of extracted features
        """
        return {
            "Source IP": row.get('IPV4_SRC_ADDR', 'Unknown'),
            "Dest IP": row.get('IPV4_DST_ADDR', 'Unknown'),
            "Dest Port": row.get('L4_DST_PORT', 0),
            "Protocol": self.get_protocol_name(row.get('PROTOCOL', 0)),
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
    
    def build_prompt(self) -> PromptTemplate:
        """Build the classification prompt template"""
        template = f"""
You are a Network Security Expert (IDS). Analyze the following network flow data.

CURRENT FLOW DATA:
{{flow_data}}

PAST MEMORY/RULES (CRITICAL SIGNATURES from Past Mistakes):
{{memory}}

INSTRUCTIONS:
1. Analyze all features, paying close attention to the Temporal features (IAT, Throughput) for rate analysis.
2. **CRITICAL:** If the 'PAST MEMORY/RULES' contains a specific, relevant signature (e.g., a rule matching low IAT for DDoS), **you must prioritize that rule's conclusion**. These rules are high-confidence anomaly detectors created from verified past errors.
3. If the 'PAST MEMORY/RULES' contains a signature, you must apply its RULE conclusion directly, even if implied.
4. If the traffic looks like normal application traffic AND no specific memory rule applies, label it "Benign".
5. Choose the best classification from the list below.

VALID LABELS (Choose EXACTLY one): {', '.join(self.valid_labels)}

OUTPUT:
Return ONLY the label name. No explanation.
"""
        return PromptTemplate.from_template(template)
    
    def classify(self, row: Dict) -> Tuple[str, Dict]:
        """
        Classify a network flow
        
        Args:
            row: Dictionary containing flow data
            
        Returns:
            Tuple of (predicted_label, extracted_features)
        """
        # Extract features
        features = self.extract_features(row)
        flow_str = json.dumps(features)
        
        # Retrieve memory context
        memory_context = self.memory.retrieve_similar_rules(flow_str)
        
        # Build and execute chain
        prompt = self.build_prompt()
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            raw_prediction = chain.invoke({
                "flow_data": flow_str,
                "memory": memory_context
            })
            clean_pred = self.normalize_label(raw_prediction)
            return clean_pred, features
        
        except Exception as e:
            self.logger.error(f"Error in LLM call: {e}")
            return "Benign", features
    
    def distill_experience(
        self, 
        features: Dict, 
        true_label: str, 
        pred_label: str
    ) -> Dict:
        """
        Create a rule from a classification mistake
        
        Args:
            features: Flow features that were misclassified
            true_label: Actual correct label
            pred_label: Incorrect predicted label
            
        Returns:
            Dictionary with strategy_signature or None
        """
        prompt = f"""
You are a Senior Network Security Analyst conducting a Post-Mortem on a classification error.

--- MISTAKE CONTEXT ---
Input Flow: {json.dumps(features)}
INCORRECT Prediction: {pred_label}
ACTUAL Label: {true_label}

--- ANALYSIS INSTRUCTIONS ---
1. COMPARE: Why did this look like '{pred_label}'? 
2. ISOLATE: What specific feature proves it is actually '{true_label}'? 
   - Check TEMPORAL: Is the IAT (Inter-Arrival Time) unusually high or low?
   - Check THROUGHPUT: Is the byte rate distinct?
   - Check FLAGS/SIZE: Are packet sizes or flags suspicious?

--- OUTPUT TASK ---
Generate a high-precision "Signature Summary" that explains the distinction.

OUTPUT FORMAT:
"Mistaken for [Pred] likely due to [Common Features], but identified as [True] because [Distinguishing Features]. RULE: If [Specific Condition], classify as [True]."
"""
        
        try:
            response = self.llm.invoke(prompt)
            rule = response.content
            self.logger.info(f"   LEARNED NEW RULE: {rule}")
            return {"strategy_signature": rule}
        
        except Exception as e:
            self.logger.error(f"Error distilling experience: {e}")
            return None

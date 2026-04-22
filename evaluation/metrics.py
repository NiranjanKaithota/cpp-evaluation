# evaluation/metrics.py
import os

def calculate_compression_ratio(raw_size: int, compressed_size: int) -> float:
    if raw_size == 0:
        return 0.0
    return float(raw_size) / compressed_size if compressed_size > 0 else float(raw_size)

def calculate_compression_percentage(raw_size: int, compressed_size: int) -> float:
    if raw_size == 0:
        return 0.0
    return (1.0 - (float(compressed_size) / float(raw_size))) * 100.0

def get_file_size(path: str) -> int:
    if os.path.exists(path):
         return os.path.getsize(path)
    return 0
    


def evaluate_rcm(predicted: dict, ground_truth: dict) -> dict:
    gt_entity = str(ground_truth.get("root_cause_entity", "")).lower()
    gt_type = str(ground_truth.get("root_cause_type", "")).lower()
    
    pred_entity = str(predicted.get("root_cause_entity", "")).lower()
    pred_type = str(predicted.get("root_cause_type", "")).lower()
    
    entity_match = (gt_entity in pred_entity) or (pred_entity in gt_entity) if gt_entity and pred_entity else False
    type_match = gt_type == pred_type
    
    passed = entity_match
    
    return {
        "entity_match": entity_match,
        "type_match": type_match,
        "passed": passed,
        "expected_entity": gt_entity,
        "predicted_entity": pred_entity,
        "expected_type": gt_type,
        "predicted_type": pred_type
    }

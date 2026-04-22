# evaluation/loader.py
import os
import json

class TestCaseLoader:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def load_bundles(self):
        bundles = []
        if not os.path.exists(self.dataset_path):
            return bundles
            
        # Check if the dataset_path is a single bundle
        if os.path.exists(os.path.join(self.dataset_path, "metadata.json")):
            bundle_dir = os.path.basename(os.path.normpath(self.dataset_path))
            bundles.append(self._load_single_bundle(self.dataset_path, bundle_dir))
            return bundles
        
        # Original logic for multiple bundles
        for bundle_dir in os.listdir(self.dataset_path):
            bundle_path = os.path.join(self.dataset_path, bundle_dir)
            if not os.path.isdir(bundle_path):
                continue
            if os.path.exists(os.path.join(bundle_path, "metadata.json")):
                bundles.append(self._load_single_bundle(bundle_path, bundle_dir))
            
        return bundles

    def _load_single_bundle(self, bundle_path: str, bundle_name: str):
        with open(os.path.join(bundle_path, "metadata.json"), 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        return {
            "name": bundle_name,
            "path": bundle_path,
            "messages_path": os.path.join(bundle_path, "messages.log"),
            "showtech_path": os.path.join(bundle_path, "showtech.txt"),
            "routeinfo_path": os.path.join(bundle_path, "routeinfo.txt"),
            "metadata": metadata
        }

    def get_raw_text(self, bundle: dict):
        text = ""
        for key in ["messages_path", "showtech_path", "routeinfo_path"]:
            path = bundle[key]
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    text += f"--- {os.path.basename(path)} ---\n"
                    text += f.read() + "\n\n"
        return text

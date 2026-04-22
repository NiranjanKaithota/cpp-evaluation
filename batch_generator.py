#!/usr/bin/env python3
"""
Batch Support Bundle Generator
Creates multiple support bundles for evaluation testing
"""

import argparse
import yaml
from pathlib import Path
from bundle_generator import SupportBundleGenerator
import json


class BatchGenerator:
    """Generate multiple support bundles for test datasets"""
    
    def __init__(self, config_file: str = "scenario_config.yaml"):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def generate_test_suite(self, suite_name: str, output_base: Path):
        """Generate all bundles in a test suite"""
        
        if suite_name not in self.config.get("test_suites", {}):
            raise ValueError(f"Unknown test suite: {suite_name}")
        
        suite = self.config["test_suites"][suite_name]
        scenarios = suite["scenarios"]
        duration = suite.get("duration_minutes", 60)
        noise = suite.get("noise_level", "medium")
        
        output_base = Path(output_base)
        output_base.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating test suite: {suite_name}")
        print(f"Description: {suite['description']}")
        print(f"Scenarios: {len(scenarios)}")
        print(f"Duration: {duration} minutes")
        print(f"Noise level: {noise}\n")
        
        bundle_info = []
        
        for idx, scenario_name in enumerate(scenarios, 1):
            if scenario_name not in self.config.get("scenarios", {}):
                print(f"⚠ Warning: Unknown scenario '{scenario_name}', skipping...")
                continue
            
            scenario_config = self.config["scenarios"][scenario_name]
            
            # Create output directory for this bundle
            bundle_dir = output_base / f"bundle_{idx:03d}_{scenario_name}"
            
            print(f"[{idx}/{len(scenarios)}] Generating {scenario_name}...")
            
            # Map scenario class name to actual scenario type
            scenario_type = self._map_scenario_class(scenario_config["class"])
            
            # Generate bundle
            generator = SupportBundleGenerator(bundle_dir)
            generator.generate(
                scenario_name=scenario_type,
                duration_minutes=duration,
                noise_level=noise,
                **scenario_config.get("params", {})
            )
            
            bundle_info.append({
                "bundle_id": f"bundle_{idx:03d}",
                "scenario_name": scenario_name,
                "scenario_type": scenario_type,
                "path": str(bundle_dir),
                "description": scenario_config["description"],
                "severity": scenario_config.get("severity", "unknown")
            })
        
        # Write test suite manifest
        manifest_path = output_base / "test_suite_manifest.json"
        manifest = {
            "suite_name": suite_name,
            "suite_description": suite["description"],
            "duration_minutes": duration,
            "noise_level": noise,
            "bundles": bundle_info,
            "total_bundles": len(bundle_info)
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n✓ Test suite generated successfully!")
        print(f"  Location: {output_base}")
        print(f"  Bundles: {len(bundle_info)}")
        print(f"  Manifest: {manifest_path}")
    
    def generate_single_scenario(self, scenario_name: str, output_dir: Path,
                                duration: int = 60, noise: str = "medium"):
        """Generate a single scenario bundle"""
        
        if scenario_name not in self.config.get("scenarios", {}):
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario_config = self.config["scenarios"][scenario_name]
        scenario_type = self._map_scenario_class(scenario_config["class"])
        
        print(f"Generating scenario: {scenario_name}")
        print(f"Description: {scenario_config['description']}")
        print(f"Severity: {scenario_config.get('severity', 'unknown')}\n")
        
        generator = SupportBundleGenerator(output_dir)
        generator.generate(
            scenario_name=scenario_type,
            duration_minutes=duration,
            noise_level=noise,
            **scenario_config.get("params", {})
        )
    
    def list_scenarios(self):
        """List all available scenarios"""
        print("Available Scenarios:")
        print("=" * 80)
        
        for name, config in self.config.get("scenarios", {}).items():
            severity = config.get("severity", "unknown").upper()
            print(f"\n{name} [{severity}]")
            print(f"  {config['description']}")
            print(f"  Class: {config['class']}")
            if config.get("params"):
                print(f"  Params: {config['params']}")
    
    def list_test_suites(self):
        """List all available test suites"""
        print("Available Test Suites:")
        print("=" * 80)
        
        for name, suite in self.config.get("test_suites", {}).items():
            print(f"\n{name}")
            print(f"  {suite['description']}")
            print(f"  Duration: {suite.get('duration_minutes', 60)} minutes")
            print(f"  Noise: {suite.get('noise_level', 'medium')}")
            print(f"  Scenarios ({len(suite['scenarios'])}): {', '.join(suite['scenarios'])}")
    
    def _map_scenario_class(self, class_name: str) -> str:
        """Map scenario class name to CLI scenario type"""
        mapping = {
            "PortFlapScenario": "port_flap",
            "BGPNeighborDownScenario": "bgp_neighbor_down",
            "VLANMismatchScenario": "vlan_mismatch",
            "MissingRouteScenario": "missing_route"
        }
        return mapping.get(class_name, class_name.lower())


def main():
    parser = argparse.ArgumentParser(
        description="Batch generate support bundles for testing"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Test suite generation
    suite_parser = subparsers.add_parser("suite", 
                                         help="Generate entire test suite")
    suite_parser.add_argument("--name", required=True,
                             help="Test suite name (from config)")
    suite_parser.add_argument("--output", required=True,
                             help="Output base directory")
    suite_parser.add_argument("--config", default="scenario_config.yaml",
                             help="Configuration file")
    
    # Single scenario generation
    single_parser = subparsers.add_parser("single",
                                          help="Generate single scenario")
    single_parser.add_argument("--scenario", required=True,
                              help="Scenario name (from config)")
    single_parser.add_argument("--output", required=True,
                              help="Output directory")
    single_parser.add_argument("--duration", type=int, default=60,
                              help="Duration in minutes")
    single_parser.add_argument("--noise", default="medium",
                              choices=["low", "medium", "high"],
                              help="Noise level")
    single_parser.add_argument("--config", default="scenario_config.yaml",
                              help="Configuration file")
    
    # List scenarios
    list_parser = subparsers.add_parser("list",
                                       help="List available scenarios/suites")
    list_parser.add_argument("--type", choices=["scenarios", "suites"],
                            default="scenarios",
                            help="What to list")
    list_parser.add_argument("--config", default="scenario_config.yaml",
                            help="Configuration file")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    batch_gen = BatchGenerator(args.config)
    
    if args.command == "suite":
        batch_gen.generate_test_suite(args.name, Path(args.output))
    
    elif args.command == "single":
        batch_gen.generate_single_scenario(
            args.scenario,
            Path(args.output),
            args.duration,
            args.noise
        )
    
    elif args.command == "list":
        if args.type == "scenarios":
            batch_gen.list_scenarios()
        else:
            batch_gen.list_test_suites()


if __name__ == "__main__":
    main()

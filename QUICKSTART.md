# Quick Start Guide - Support Bundle Generator

## What You Have

A complete support bundle generator that creates realistic HPE AOS-CX switch logs with injected network failures for testing your LLM-based debugging pipeline.

## Immediate Next Steps

### 1. Generate Your First Test Dataset

```bash
# Create a comprehensive test suite (8 bundles with different failures)
python batch_generator.py suite --name regression_test --output ./evaluation_dataset

# This creates:
# - 8 realistic support bundles
# - Each with messages.log, showtech.txt, routeinfo.txt
# - metadata.json with ground truth for each
```

### 2. Understand the Output Structure

Each bundle contains:
```
bundle_001_port_flap_fiber/
├── messages.log       # Event logs (this is your main signal)
├── showtech.txt       # Switch state at failure time
├── routeinfo.txt      # Network/routing information
└── metadata.json      # Ground truth (what your evaluation will compare against)
```

**Key fields in metadata.json:**
- `root_cause_entity`: The exact component that failed (e.g., "1/1/3")
- `root_cause_type`: Category of failure
- `expected_symptoms`: List of symptoms that should be detected
- `recommended_action`: The correct remediation step
- `failure_timestamp`: When the failure occurred

### 3. Verify a Generated Bundle

```bash
# Look at one bundle
cd evaluation_dataset/bundle_001_port_flap_fiber

# Check the ground truth
cat metadata.json

# See the failure in the logs (grep for ERR or WARN)
grep -E "(ERR|WARN|Down)" messages.log

# Check interface state in showtech
grep -A5 "show interface dom" showtech.txt | grep "1/1/3"
```

### 4. Run Your Compression Pipeline

Now you can:
1. Feed `messages.log`, `showtech.txt`, `routeinfo.txt` through your 5-stage pipeline
2. Get compressed output
3. Compare both raw and compressed against ground truth

## Available Test Scenarios

### Already Generated in test_suites/smoke/
- ✓ Port flap (physical layer failure)
- ✓ BGP neighbor down (routing protocol failure)
- ✓ VLAN mismatch (configuration error)
- ✓ Missing route (routing table error)

### Generate Custom Scenarios

```bash
# High-noise port flap for stress testing
python bundle_generator.py \
  --scenario port_flap \
  --output ./custom_tests/stress_port_flap \
  --interface 1/1/49 \
  --duration 120 \
  --noise-level high

# BGP failure with specific peer
python bundle_generator.py \
  --scenario bgp_neighbor_down \
  --output ./custom_tests/bgp_peer_down \
  --neighbor-ip 192.168.1.1 \
  --duration 60 \
  --noise-level medium
```

## Token Count Estimation

Based on your requirements (300K+ tokens for raw bundles):

- **30 min duration, low noise**: ~50K tokens
- **60 min duration, medium noise**: ~200K tokens
- **120 min duration, high noise**: ~500K tokens

**Recommendation for initial testing:**
Use 60-90 minute duration with medium noise to hit your 300K+ target.

```bash
python bundle_generator.py \
  --scenario port_flap \
  --output ./large_bundle \
  --duration 90 \
  --noise-level medium
```

## Integration with Your Evaluation Pipeline

### Expected Workflow

```
┌─────────────────────┐
│ Generate Bundles    │
│ (This tool)         │
└──────────┬──────────┘
           │
           ├─── Raw Logs (300K+ tokens)
           │
           ↓
┌─────────────────────┐
│ Your 5-Stage        │
│ Compression         │
│ Pipeline            │
└──────────┬──────────┘
           │
           ├─── Compressed Logs (10-30K tokens)
           │
           ↓
┌─────────────────────┐
│ Dual LLM Inference  │
│ (Baseline vs        │
│  Compressed)        │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│ Evaluation Metrics  │
│ - RCM (Root Cause)  │
│ - Compression Ratio │
│ - Diagnostic Match  │
└─────────────────────┘
           │
           ↓
Compare LLM outputs
against metadata.json
```

### What to Extract from metadata.json for Evaluation

```python
import json

# Load ground truth
with open('bundle_001/metadata.json') as f:
    gt = json.load(f)['ground_truth']

# Key fields for evaluation:
expected_entity = gt['root_cause_entity']        # e.g., "1/1/3"
expected_type = gt['root_cause_type']            # e.g., "physical_layer_failure"
expected_action = gt['recommended_action']        # e.g., "Check cable on 1/1/3"
failure_time = gt['failure_timestamp']           # For temporal ordering check

# Now compare LLM output:
# - Did it identify "1/1/3" as the problem?
# - Did it recommend checking the cable?
# - Did it maintain correct temporal order?
```

## Common Patterns in Generated Logs

### Normal Background Noise
```
2026-04-17T19:20:20.926000+05:30 PSCSCTLEAFB03 acctsyslogd[387]: AUDIT|REST URI executed...
2026-04-17T19:20:21.029000+05:30 PSCSCTLEAFB03 kernel: net_ratelimit: 85 callbacks suppressed
```

### Failure Signal (Port Flap)
```
2026-04-17T19:20:42.058000+05:30 PSCSCTLEAFB03 kernel: provision 0000:04:00.0 1/1/3: NIC Link is Down
2026-04-17T19:20:43.310000+05:30 PSCSCTLEAFB03 hpe-routing[4714]: ovs|17708|interface_mgr|ERR|Interface 1/1/3 state changed to down
```

### Failure Signal (BGP)
```
2026-04-17T19:25:12.123000+05:30 PSCSCTLEAFB03 hpe-routing[4714]: ovs|12345|bgp|WARN|BGP peer 10.233.255.1 connection lost
2026-04-17T19:25:12.234000+05:30 PSCSCTLEAFB03 hpe-routing[4714]: ovs|12346|bgp|ERR|BGP session to 10.233.255.1 went down - Hold Timer Expired
```

## Troubleshooting

### "No module named 'yaml'"
```bash
pip install pyyaml
```

### Generated logs too short
Increase `--duration` parameter (each minute ≈ 3-5K tokens with medium noise)

### Need more realistic failures
1. Add custom scenarios to `scenario_config.yaml`
2. Increase `--noise-level` to `high` for production-like chaos

## Next: Build the Evaluation Pipeline

Once you have test bundles, the next step is building the evaluation harness:

1. **Dual Execution Engine**: Runs LLM on both raw and compressed bundles
2. **Metrics Calculator**: Computes RCM, compression ratio, etc.
3. **Judge Integration**: LLM-as-a-Judge for qualitative metrics
4. **Report Generator**: Aggregates results into scorecard

Would you like me to build that next?

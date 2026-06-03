# Compressed Causal Narrative

## Event Template Dictionary
### Clusters
- **Cluster -1**: [Uncategorized_Noise] Keywords: 
- **Cluster 0**: [Noise_Domain] Keywords: [bridge, ovs, vlan]
- **Cluster 1**: [Uri_Domain] Keywords: [uri, rest, server]
- **Cluster 2**: [Time_Domain] Keywords: [time, disabling, enabling]

### Templates
- **T1 [C0]**: < ts>+05 pscsctleafb03 ovs vswitchd < pid > ovs|08151|bridge|info|bridge br0 interface statistics queried status ok
- **T2 [C1]**: < ts>+05 pscsctleafb03 acctsyslogd < pid > uri executed user afc admin address < ip > rest session uri uri > method response code payload resulted success timezone asia calcutta
- **T3 [C-1]**: < ts>+05 pscsctleafb03 kernel net ratelimit callbacks suppressed
- **T4 [C1]**: < ts>+05 pscsctleafb03 ovsdb server < pid > ovs|03527|ovsdb server|info|transact success transaction completed rows inserted
- **T5 [C2]**: < ts>+05 pscsctleafb03 kernel provision < time>.0 enabling interrupts disabling napi polling
- **T6 [C2]**: < ts>+05 pscsctleafb03 kernel provision < time>.0 pvnet0 enabling interrupts disabling napi polling
- **T7 [C0]**: < ts>+05 pscsctleafb03 hpe routing < pid > ovs|17515|vlan mgr|err|vlan mismatch expected received
- **T8 [C0]**: < ts>+05 pscsctleafb03 ovs vswitchd < pid > ovs|17516|bridge|err|bridge dropping frame unexpected tag received interface
- **T9 [C-1]**: < ts>+05 pscsctleafb03 kernel dropping frame unexpected tag

## Causal Sequences
## Dynamic Token Reduction Strategies Applied:
- None (Log volume within limits).

| ID | Causal Path | Occurrences | Time Span | Entities |
|---|---|---|---|---|

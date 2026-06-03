# Compressed Causal Narrative

## Event Template Dictionary
### Clusters
- **Cluster -1**: [Uncategorized_Noise] Keywords: 
- **Cluster 0**: [Noise_Domain] Keywords: [uri, success, rest]
- **Cluster 1**: [Time_Domain] Keywords: [time, provision, polling]
- **Cluster 2**: [Bridge_Domain] Keywords: [bridge, ovs, interface]

### Templates
- **T1 [C0]**: < ts>+05 pscsctleafb03 acctsyslogd < pid > uri executed user afc admin address < ip > rest session uri uri > method response code payload resulted success timezone asia calcutta
- **T2 [C-1]**: < ts>+05 pscsctleafb03 kernel net ratelimit callbacks suppressed
- **T3 [C1]**: < ts>+05 pscsctleafb03 kernel provision < time>.0 enabling interrupts disabling napi polling
- **T4 [C0]**: < ts>+05 pscsctleafb03 ovsdb server < pid > ovs|09263|ovsdb server|info|transact success transaction completed rows inserted
- **T5 [C2]**: < ts>+05 pscsctleafb03 ovs vswitchd < pid > ovs|04569|bridge|info|bridge br0 interface statistics queried status ok
- **T6 [C1]**: < ts>+05 pscsctleafb03 kernel provision < time>.0 pvnet1 enabling interrupts disabling napi polling
- **T7 [C-1]**: < ts>+05 pscsctleafb03 hpe routing < pid > expected received
- **T8 [C2]**: < ts>+05 pscsctleafb03 ovs vswitchd < pid > ovs|19935|bridge|err|bridge br0 dropping frame unexpected tag received interface
- **T9 [C-1]**: < ts>+05 pscsctleafb03 kernel dropping frame unexpected tag

## Causal Sequences
## Dynamic Token Reduction Strategies Applied:
- None (Log volume within limits).

| ID | Causal Path | Occurrences | Time Span | Entities |
|---|---|---|---|---|

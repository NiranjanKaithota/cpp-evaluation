# Compressed Causal Narrative

## Event Template Dictionary
### Clusters
- **Cluster -1**: [Uncategorized_Noise] Keywords: 
- **Cluster 0**: [Noise_Domain] Keywords: [time, napi, enabling]
- **Cluster 1**: [Ovsdb_Domain] Keywords: [ovsdb, server, transact]
- **Cluster 2**: [Bgp_Domain] Keywords: [bgp, hpe, routing]

### Templates
- **T1 [C0]**: < ts>+05 pscsctleafb03 kernel provision < time>.0 enabling interrupts disabling napi polling
- **T2 [C0]**: < ts>+05 pscsctleafb03 kernel provision < time>.0 pvnet1 enabling interrupts disabling napi polling
- **T3 [C-1]**: < ts>+05 pscsctleafb03 ovs vswitchd < pid > ovs|03353|bridge|info|bridge br0 interface statistics queried status ok
- **T4 [C-1]**: < ts>+05 pscsctleafb03 kernel net ratelimit callbacks suppressed
- **T5 [C-1]**: < ts>+05 pscsctleafb03 acctsyslogd < pid > uri executed user afc admin address < ip > rest session uri uri > method response code payload resulted success timezone asia calcutta
- **T6 [C1]**: < ts>+05 pscsctleafb03 ovsdb server < pid > ovs|05330|ovsdb server|info|transact success transaction completed rows inserted
- **T7 [C2]**: < ts>+05 pscsctleafb03 hpe routing < pid > ovs|16082|bgp|warn|bgp peer < ip > connection lost
- **T8 [C2]**: < ts>+05 pscsctleafb03 hpe routing < pid > ovs|16083|bgp|err|bgp session < ip went hold timer expired
- **T9 [C1]**: < ts>+05 pscsctleafb03 ovsdb server < pid > ovs|16085|ovsdb server|info|transact success rows updated table withdrew peer < ip >

## Causal Sequences
## Dynamic Token Reduction Strategies Applied:
- None (Log volume within limits).

| ID | Causal Path | Occurrences | Time Span | Entities |
|---|---|---|---|---|
| 1 | [C-1] (T5) (2)→ →[C0] (T1) (3)→ ⏳ 1.0s →[C-1] (T3) (4)→ →[C2] (T7) (2) | 1 | 05-27 03:32:32 to 05-27 03:32:53 | [10.233.131.22, 10.233.255.1, 505, None] |
| 2 | [C-1] (T5) (2)→ →[C0] (T1) (3)→ ⏳ 1.0s →[C-1] (T3) (4)→ →[C2] (T7) (2)→ →[C2] (T8) | 1 | 05-27 03:32:32 to 05-27 03:32:57 | [10.233.131.22, 10.233.255.1, 505, None] |
